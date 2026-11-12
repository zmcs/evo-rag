"""Build a retriever from a PipelineConfig: dense, BM25 or hybrid search over chunks.

Chunks and embeddings are cached on disk, keyed only by the genes that affect them,
so configs that differ in other genes (top_k, generator, ...) reuse the same index.
"""

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from evo_rag.chunking import Chunk, EmbedFn, chunk_corpus
from evo_rag.config import PipelineConfig

CORPUS_PATH = Path("data/processed/corpus.jsonl")
CACHE_DIR = Path("data/cache")

# Hugging Face model name, query prefix, passage prefix (each model was trained with these).
EMBEDDERS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "bge-base": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "e5-base": ("intfloat/e5-base-v2", "query: ", "passage: "),
}

TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens for BM25 ('Article 6(3)' -> ['article', '6', '3'])."""
    return TOKEN_RE.findall(text.lower())


class Embedder:
    """Wraps a sentence-transformers model; returns L2-normalised vectors."""

    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer  # slow import, only when needed

        model_name, self.query_prefix, self.passage_prefix = EMBEDDERS[name]
        self.model = SentenceTransformer(model_name)
        self._memo: dict[str, np.ndarray] = {}  # text -> vector, reused across configs in one run

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts, computing only those not seen before in this process."""
        new = list(dict.fromkeys(t for t in texts if t not in self._memo))
        if new:
            vectors = self.model.encode(new, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
            self._memo.update(zip(new, vectors))
        return np.array([self._memo[t] for t in texts])

    def passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])

    def queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])


@lru_cache(maxsize=3)
def get_embedder(name: str) -> Embedder:
    """Load each embedding model once per process."""
    return Embedder(name)


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _cache_key(**genes) -> str:
    return hashlib.sha1(json.dumps(genes, sort_keys=True).encode()).hexdigest()[:12]


def _cached(path: Path, build):
    """Return the pickled value at `path`, building and saving it on first use."""
    if path.exists():
        return pickle.loads(path.read_bytes())
    value = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(value))
    return value


def minmax(scores: np.ndarray) -> np.ndarray:
    spread = scores.max() - scores.min()
    return (scores - scores.min()) / spread if spread > 0 else np.zeros_like(scores)


@dataclass
class Retriever:
    config: PipelineConfig
    chunks: list[Chunk]
    embedder: Embedder | None
    embeddings: np.ndarray | None
    bm25: BM25Okapi | None

    def scores(self, query: str) -> np.ndarray:
        """One relevance score per chunk."""
        mode = self.config.retrieval
        if mode == "bm25":
            return self.bm25.get_scores(tokenize(query))
        dense = self.embeddings @ self.embedder.queries([query])[0]
        if mode == "dense":
            return dense
        # Hybrid: put both on a 0-1 scale, then mix (alpha=1 pure dense, alpha=0 pure BM25).
        alpha = self.config.hybrid_alpha
        return alpha * minmax(dense) + (1 - alpha) * minmax(self.bm25.get_scores(tokenize(query)))

    def search(self, query: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        k = k or self.config.top_k
        scores = self.scores(query)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def build_retriever(config: PipelineConfig, docs: list[dict] | None = None, embedder_factory=get_embedder) -> Retriever:
    docs = docs if docs is not None else load_corpus()
    needs_dense = config.retrieval in ("dense", "hybrid")
    needs_embedder = needs_dense or config.chunker == "semantic"
    embedder = embedder_factory(config.embedder) if needs_embedder else None

    # Chunks depend on the embedder only for the semantic chunker (an inactive gene otherwise).
    chunk_genes = dict(
        chunker=config.chunker,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        chunk_header=config.chunk_header,
        embedder=config.embedder if config.chunker == "semantic" else None,
    )
    chunk_key = _cache_key(**chunk_genes)
    chunks = _cached(
        CACHE_DIR / f"chunks_{chunk_key}.pkl",
        lambda: chunk_corpus(docs, config, embed=embedder.passages if embedder else None),
    )

    embeddings = None
    if needs_dense:
        emb_key = _cache_key(chunks=chunk_key, embedder=config.embedder)
        embeddings = _cached(
            CACHE_DIR / f"emb_{emb_key}.pkl",
            lambda: embedder.passages([c.text for c in chunks]),
        )

    bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if config.retrieval in ("bm25", "hybrid") else None
    return Retriever(config, chunks, embedder, embeddings, bm25)

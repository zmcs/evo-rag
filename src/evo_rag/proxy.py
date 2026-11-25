"""Free proxy objectives for a config: no LLM calls, no API cost.

  recall    mean recall@k over the answerable dev questions (k = the config's top_k)
            -> stands in for correctness (rank correlation 0.93 in the landscape check)
  tokens    estimated input tokens per query sent to the generator (prompt + chunks)
            -> stands in for cost (cost is proportional to tokens)
  evidence_recall
            fraction of each question's verbatim evidence quotes that appear in the
            retrieved text. Stricter than recall: a retrieved fragment of the right
            article only counts if it contains the passage that answers the question.

Results are cached in data/results/proxy_cache.jsonl, so an interrupted search resumes
where it stopped and no config is ever scored twice.
"""

import json
import re
import time
from pathlib import Path

import numpy as np

from evo_rag.config import PipelineConfig
from evo_rag.generation import PROMPT_TEMPLATES, format_context
from evo_rag.planner import Retrieval, retrieve
from evo_rag.retrieval import build_retriever, load_corpus, minmax, tokenize
from evo_rag.retrieval_eval import load_questions, score_question
from evo_rag.search_space import config_key

CACHE_PATH = Path("data/results/proxy_cache.jsonl")
# Calibrated against Gemini's real count: ~820 input tokens per query for the default
# config (recursive 800 + header, k=5), whose prompts average ~4,100 characters.
CHARS_PER_TOKEN = 5
# Bump when the scoring changes: cached rows from an older version are recomputed.
PROXY_VERSION = 2


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def evidence_found(quotes: list[str], chunks: list, has_header: bool) -> float:
    """Fraction of quotes contained in the retrieved text.

    Retrieved chunks that are neighbours in the same document are re-joined first (with
    their headers removed), so a quote cut by a chunk boundary still counts when the
    generator received both halves.
    """
    ordered = sorted(chunks, key=lambda c: (c.doc_id, int(c.chunk_id.split("#")[1])))
    runs, prev = [], None
    for c in ordered:
        body = c.text.split("\n", 1)[1] if has_header and "\n" in c.text else c.text
        idx = int(c.chunk_id.split("#")[1])
        if prev and prev[0] == c.doc_id and prev[1] + 1 == idx:
            runs[-1] += " " + body
        else:
            runs.append(body)
        prev = (c.doc_id, idx)
    text = [_norm(r) for r in runs]
    return sum(any(_norm(q) in t for t in text) for q in quotes) / len(quotes)


class ProxyEvaluator:
    """llm is only needed for query_mode="decompose" (cached query plans; see planner.py)."""

    def __init__(self, split: str = "dev", cache_path: Path = CACHE_PATH, llm=None, **retriever_kwargs):
        self.llm = llm
        self.questions = load_questions(split)
        self.docs = load_corpus()
        self.cache_path = cache_path
        self.retriever_kwargs = retriever_kwargs
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row
        self._query_vectors: dict[str, np.ndarray] = {}  # embedder name -> question vectors

    def __call__(self, config: PipelineConfig) -> dict:
        key = config_key(config)
        if self.cache.get(key, {}).get("version") != PROXY_VERSION:
            row = {"key": key, "version": PROXY_VERSION, "config": config.model_dump(), **self._score(config)}
            self.cache[key] = row
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        return self.cache[key]

    def _score(self, config: PipelineConfig) -> dict:
        start = time.perf_counter()
        rows, n_chunks = self.per_question(config)
        return {
            "recall": float(np.mean([r["recall"] for r in rows])),
            "evidence_recall": float(np.mean([r["evidence_recall"] for r in rows])),
            "tokens": float(np.mean([r["tokens"] for r in rows])),
            "n_chunks": n_chunks,
            "seconds": time.perf_counter() - start,
        }

    def per_question(self, config: PipelineConfig) -> tuple[list[dict], int]:
        """Proxy scores for every question (not cached), plus the number of chunks.

        tokens = generator input tokens, plus the planning call (if any) converted to
        input-token equivalents, so the cost objective includes the planner's price.
        """
        retriever = build_retriever(config, self.docs, **self.retriever_kwargs)
        template = PROMPT_TEMPLATES[config.prompt_template]
        if config.query_mode == "single":
            scores = self._all_scores(retriever)  # (n_questions, n_chunks), batched
            found = [
                Retrieval([retriever.chunks[i] for i in np.argsort(-row)[: config.top_k]], [q["question"]])
                for q, row in zip(self.questions, scores)
            ]
        else:
            if self.llm is None:
                raise ValueError("query_mode='decompose' needs ProxyEvaluator(llm=LLM()) for query plans.")
            found = [retrieve(q["question"], retriever, config, self.llm) for q in self.questions]

        rows = []
        for q, f in zip(self.questions, found):
            prompt = template["system"] + template["user"].format(context=format_context(f.chunks), question=q["question"])
            rows.append({
                "qid": q["qid"],
                "type": q["type"],
                "recall": score_question([c.doc_id for c in f.chunks], set(q["doc_ids"]))["recall"],
                "evidence_recall": evidence_found([quote for _, quote in q["evidence"]], f.chunks, config.chunk_header),
                "tokens": len(prompt) / CHARS_PER_TOKEN + f.plan_cost_tokens,
                "queries": f.queries,
            })
        return rows, len(retriever.chunks)

    def _all_scores(self, retriever) -> np.ndarray:
        """Relevance scores for every question at once (batched, so dense search is fast)."""
        mode = retriever.config.retrieval
        texts = [q["question"] for q in self.questions]
        bm25 = None
        if mode in ("bm25", "hybrid"):
            bm25 = np.array([retriever.bm25.get_scores(tokenize(t)) for t in texts])
            if mode == "bm25":
                return bm25
        name = retriever.config.embedder
        if name not in self._query_vectors:
            self._query_vectors[name] = retriever.embedder.queries(texts)
        dense = self._query_vectors[name] @ retriever.embeddings.T
        if mode == "dense":
            return dense
        alpha = retriever.config.hybrid_alpha
        return np.array([alpha * minmax(d) + (1 - alpha) * minmax(b) for d, b in zip(dense, bm25)])

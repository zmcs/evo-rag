"""Split corpus documents into chunks, as described by a PipelineConfig.

Every chunk remembers the doc_id it came from; retrieval metrics depend on it.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from evo_rag.config import PipelineConfig

# A new paragraph starts at a line that is a paragraph number ("1.") or a point ("(a) ...").
PARAGRAPH_NUMBER_RE = re.compile(r"^\d+\.$")
POINT_RE = re.compile(r"^\((?:[a-z]{1,2}|[ivx]+|\d+)\)\s")
SENTENCE_END_RE = re.compile(r"(?<=[.;:])\s+")

# Separators for the recursive chunker, from coarsest to finest.
RECURSIVE_SEPARATORS = ["\n", ". ", "; ", ", ", " "]

EmbedFn = Callable[[list[str]], np.ndarray]  # texts -> L2-normalised vectors


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str  # what gets embedded/indexed (includes the header if enabled)


def to_paragraphs(text: str) -> list[str]:
    """Undo PDF line wrapping: rebuild numbered paragraphs and lettered points."""
    paragraphs: list[str] = []
    pending_number = None
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if PARAGRAPH_NUMBER_RE.match(line):
            pending_number = line  # "1." sits on its own line; glue it to what follows
            continue
        starts_new = pending_number is not None or POINT_RE.match(line) or not paragraphs
        if starts_new:
            paragraphs.append(f"{pending_number} {line}" if pending_number else line)
            pending_number = None
        else:
            paragraphs[-1] += " " + line
    return paragraphs


def doc_label(doc_id: str) -> str:
    kind, number = doc_id.split("_", 1)
    return f"Article {number}" if kind == "art" else f"Annex {number}"


def _tail(text: str, n_chars: int) -> str:
    """Last ~n_chars of text, starting at a word boundary."""
    if n_chars <= 0 or len(text) <= n_chars:
        return text if n_chars > 0 else ""
    tail = text[-n_chars:]
    return tail[tail.find(" ") + 1 :] if " " in tail else tail


def _add_overlap(pieces: list[str], overlap_chars: int) -> list[str]:
    """Prefix each chunk with the end of the previous one."""
    if overlap_chars <= 0 or len(pieces) < 2:
        return pieces
    overlapped = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        overlapped.append(f"{_tail(prev, overlap_chars)} {cur}")
    return overlapped


def split_fixed(text: str, size: int, overlap_chars: int) -> list[str]:
    """Sliding character window that ignores the text's structure."""
    flat = " ".join(text.split())
    step = max(1, size - overlap_chars)
    return [flat[i : i + size] for i in range(0, len(flat), step) if flat[i : i + size].strip()]


def _merge(pieces: list[str], separator: str, size: int) -> list[str]:
    """Greedily pack consecutive pieces into chunks of at most `size` characters."""
    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current}{separator}{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _split_recursive(text: str, size: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not separators:  # no separator left: hard cut
        return [text[i : i + size] for i in range(0, len(text), size)]
    separator, rest = separators[0], separators[1:]
    pieces = [p for p in text.split(separator) if p.strip()]
    if len(pieces) == 1:
        return _split_recursive(text, size, rest)
    # Split oversized pieces further, then pack small neighbours back together.
    small = [sub for p in pieces for sub in _split_recursive(p, size, rest)]
    return _merge(small, separator, size)


def split_recursive(text: str, size: int, overlap_chars: int) -> list[str]:
    """Split at paragraph, then sentence, then clause, then word boundaries."""
    structured = "\n".join(to_paragraphs(text))
    return _add_overlap(_split_recursive(structured, size, RECURSIVE_SEPARATORS), overlap_chars)


def split_semantic(
    text: str, size: int, overlap_chars: int, embed: EmbedFn, breakpoint_percentile: float = 25
) -> list[str]:
    """Start a new chunk where consecutive sentences are least similar in meaning."""
    sentences = [
        s for p in to_paragraphs(text) for s in SENTENCE_END_RE.split(p) if s.strip()
    ]
    # Sentences longer than a chunk are pre-split so no chunk exceeds `size`.
    sentences = [s for sent in sentences for s in _split_recursive(sent, size, [", ", " "])]
    if len(sentences) <= 1:
        return sentences

    vectors = embed(sentences)
    similarity = np.sum(vectors[:-1] * vectors[1:], axis=1)  # cosine (vectors are normalised)
    threshold = np.percentile(similarity, breakpoint_percentile)

    chunks, current = [], sentences[0]
    for sentence, sim in zip(sentences[1:], similarity):
        if sim < threshold or len(current) + 1 + len(sentence) > size:
            chunks.append(current)
            current = sentence
        else:
            current += " " + sentence
    chunks.append(current)
    return _add_overlap(chunks, overlap_chars)


def chunk_corpus(docs: list[dict], config: PipelineConfig, embed: EmbedFn | None = None) -> list[Chunk]:
    """Chunk every document according to the config's chunking genes."""
    overlap = int(config.chunk_size * config.chunk_overlap)
    chunks = []
    for doc in docs:
        if config.chunker == "fixed":
            pieces = split_fixed(doc["text"], config.chunk_size, overlap)
        elif config.chunker == "recursive":
            pieces = split_recursive(doc["text"], config.chunk_size, overlap)
        else:
            if embed is None:
                raise ValueError("The semantic chunker needs an embedding function.")
            pieces = split_semantic(doc["text"], config.chunk_size, overlap, embed)

        header = f"{doc_label(doc['doc_id'])}: {doc['title']}\n" if config.chunk_header else ""
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(f"{doc['doc_id']}#{i}", doc["doc_id"], header + piece))
    return chunks

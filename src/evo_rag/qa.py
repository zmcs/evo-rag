"""Validate the QA set against the corpus and print a summary.

Checks, for every question:
  - each evidence quote appears verbatim (whitespace-insensitive) in the cited document
  - every doc_id exists in the corpus
  - unanswerable questions cite no documents, all others cite at least one
  - multi-hop questions cite at least two different documents

Run:  uv run python -m evo_rag.qa
"""

import json
import re
from collections import Counter
from pathlib import Path

CORPUS_PATH = Path("data/processed/corpus.jsonl")
QA_PATH = Path("data/processed/qa.jsonl")


def normalise(text: str) -> str:
    """Collapse whitespace so PDF line breaks don't break quote matching."""
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(qa: list[dict], corpus: dict[str, str]) -> list[str]:
    errors = []
    for q in qa:
        qid, evidence, doc_ids = q["qid"], q["evidence"], q["doc_ids"]
        if q["type"] == "unanswerable":
            if doc_ids:
                errors.append(f"{qid}: unanswerable question should cite no documents")
            continue
        if not evidence:
            errors.append(f"{qid}: no evidence")
        if q["type"] == "multi_hop" and len(set(doc_ids)) < 2:
            errors.append(f"{qid}: multi-hop question cites fewer than 2 documents")
        for doc_id, quote in evidence:
            if doc_id not in corpus:
                errors.append(f"{qid}: unknown doc_id {doc_id}")
            elif normalise(quote) not in corpus[doc_id]:
                errors.append(f"{qid}: quote not found in {doc_id}: {quote[:60]!r}")
    return errors


def main() -> None:
    corpus = {d["doc_id"]: normalise(d["text"]) for d in load_jsonl(CORPUS_PATH)}
    qa = load_jsonl(QA_PATH)

    errors = validate(qa, corpus)
    for e in errors:
        print("ERROR", e)

    print(f"{len(qa)} questions, {len(errors)} errors")
    print("By type: ", dict(Counter(q["type"] for q in qa)))
    print("By split:", dict(Counter(q["split"] for q in qa)))
    print("Verified:", sum(q["verified"] for q in qa), "/", len(qa))


if __name__ == "__main__":
    main()

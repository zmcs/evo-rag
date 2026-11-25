"""Week 4, step 1: synthetic training pairs for adapting the embedder (~$0.06).

For a sample of passages from every article and annex, an LLM writes a realistic question
that the passage answers. The (question, passage) pairs train the adapters; the QA set is
never used for training.

Safeguards
  - questions too similar to any QA question (dev or test) are dropped, so evaluation
    questions cannot leak into training;
  - the split into train / val is by document, so validation passages are unseen.

  uv run python -m evo_rag.synth_data          # dry run: cost estimate
  uv run python -m evo_rag.synth_data --yes    # generate
"""

import argparse
import json
import random
import re
from pathlib import Path

from evo_rag.chunking import chunk_corpus, doc_label
from evo_rag.config import PipelineConfig
from evo_rag.llm import LLM, PRICES_USD_PER_M
from evo_rag.retrieval import load_corpus

OUT_PATH = Path("data/processed/synthetic_pairs.jsonl")
QA_PATH = Path("data/processed/qa.jsonl")
MODEL = "gemini-3.5-flash-lite"
# The passages adapters are trained on: the chunking used by the best pipelines' family.
REFERENCE_CHUNKING = PipelineConfig(chunker="recursive", chunk_size=600, chunk_overlap=0.0, chunk_header=True)
PASSAGES_PER_DOC = 4
VAL_FRACTION = 0.1
MAX_JACCARD = 0.5  # drop synthetic questions this close to any QA question

SYSTEM = "You write realistic questions that compliance officers and lawyers ask about the EU AI Act."
PROMPT = """Below are numbered passages from {label} ({title}) of the EU AI Act.

{passages}

For each passage, write one question that the passage answers on its own. Phrase it the way a
real user would ask it, in your own words: do not copy phrases from the passage. Do not mention
passage numbers.

Reply with JSON only: {{"questions": [{{"passage": 1, "question": "..."}}, ...]}}"""


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic (question, passage) pairs.")
    parser.add_argument("--yes", action="store_true", help="generate (otherwise only estimate the cost)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    docs = {d["doc_id"]: d for d in load_corpus()}
    chunks = [c for c in chunk_corpus(list(docs.values()), REFERENCE_CHUNKING) if len(c.text) >= 200]
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    rng = random.Random(args.seed)
    sampled = {d: rng.sample(cs, min(PASSAGES_PER_DOC, len(cs))) for d, cs in by_doc.items()}

    n_calls, n_passages = len(sampled), sum(len(v) for v in sampled.values())
    price_in, price_out = PRICES_USD_PER_M[MODEL]
    estimate = n_calls * (900 * price_in + 40 * PASSAGES_PER_DOC * price_out) / 1_000_000
    llm = LLM()
    print(f"{n_calls} documents, {n_passages} passages -> up to {n_passages} training pairs")
    print(f"Estimated cost: ~${estimate:.2f}   (spent so far ${llm.total_spend():.2f} of ${llm.budget_usd:.2f})")
    if estimate > llm.budget_usd - llm.total_spend():
        print("Refusing to run: the estimate exceeds what is left under the budget.")
        return
    if not args.yes:
        print("Dry run only. Re-run with --yes to generate.")
        return

    qa_words = [words(json.loads(line)["question"]) for line in QA_PATH.read_text(encoding="utf-8").splitlines()]
    doc_ids = sorted(sampled)
    rng.shuffle(doc_ids)
    val_docs = set(doc_ids[: max(1, round(VAL_FRACTION * len(doc_ids)))])

    rows, dropped, failed = [], 0, 0
    for doc_id in sorted(sampled):
        passages = sampled[doc_id]
        listing = "\n\n".join(f"[{i}] {c.text.split(chr(10), 1)[-1]}" for i, c in enumerate(passages, 1))
        response = llm.generate(
            model=MODEL, system=SYSTEM, json_mode=True, max_output_tokens=600,
            prompt=PROMPT.format(label=doc_label(doc_id), title=docs[doc_id]["title"], passages=listing),
        )
        try:
            items = json.loads(response.text)["questions"]
        except (json.JSONDecodeError, KeyError, TypeError):
            failed += 1
            continue
        for item in items:
            try:
                i, question = int(item["passage"]), str(item["question"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if not (1 <= i <= len(passages)) or len(question) < 15:
                continue
            if any(jaccard(words(question), qw) > MAX_JACCARD for qw in qa_words):
                dropped += 1
                continue
            c = passages[i - 1]
            rows.append({"query": question, "chunk_id": c.chunk_id, "doc_id": c.doc_id, "passage": c.text,
                         "split": "val" if doc_id in val_docs else "train"})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_val = sum(r["split"] == "val" for r in rows)
    print(f"\nSaved {len(rows)} pairs ({len(rows) - n_val} train, {n_val} val) to {OUT_PATH}")
    print(f"Dropped {dropped} too similar to QA questions; {failed} documents with unreadable replies")
    for r in rows[:3]:
        print(f"  Q: {r['query']}\n     -> {r['chunk_id']}")
    print(f"Total API spend so far: ${llm.total_spend():.4f}")


if __name__ == "__main__":
    main()

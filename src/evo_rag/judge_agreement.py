"""Measure how well the LLM judge agrees with your own labels.

1. Export a sample of judged answers from an evaluation:
     uv run python -m evo_rag.judge_agreement export            (latest evaluation)
     uv run python -m evo_rag.judge_agreement export <eval_id>
   This writes data/results/judge_labels.csv.

2. Open the CSV (Numbers, Excel or VS Code) and fill in, for every row:
     human_correctness  correct / partial / incorrect   (compare the answer with the gold answer)
     human_faithful     true / false                    (is every claim in the context column?)
   Fill these in WITHOUT looking at the judge columns, so they don't bias you.

3. Compute agreement:
     uv run python -m evo_rag.judge_agreement score
"""

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

from evo_rag.config import PipelineConfig
from evo_rag.retrieval import build_retriever

RESULTS_DIR = Path("data/results")
QA_PATH = Path("data/processed/qa.jsonl")
LABELS_PATH = RESULTS_DIR / "judge_labels.csv"
SCORE_TO_LABEL = {1.0: "correct", 0.5: "partial", 0.0: "incorrect"}
COLUMNS = [
    "qid", "type", "question", "gold_answer", "answer", "context",
    "human_correctness", "human_faithful", "judge_correctness", "judge_faithful", "judge_reason",
]


def load_eval(eval_id: str | None) -> tuple[dict, list[dict]]:
    with (RESULTS_DIR / "evals.jsonl").open(encoding="utf-8") as f:
        evals = [json.loads(line) for line in f]
    result = next(e for e in reversed(evals) if eval_id in (None, e["eval_id"]))
    with (RESULTS_DIR / "details" / f"{result['eval_id']}.jsonl").open(encoding="utf-8") as f:
        return result, [json.loads(line) for line in f]


def export(eval_id: str | None = None, n: int = 50, seed: int = 0) -> None:
    result, records = load_eval(eval_id)
    judged = [r for r in records if not r["abstained"]]  # abstentions are scored by rule
    sample = random.Random(seed).sample(judged, min(n, len(judged)))

    qa = {}
    with QA_PATH.open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            qa[q["qid"]] = q
    retriever = build_retriever(PipelineConfig(**result["config"]))
    chunk_text = {c.chunk_id: c.text for c in retriever.chunks}

    with LABELS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in sample:
            writer.writerow({
                "qid": r["qid"],
                "type": r["type"],
                "question": qa[r["qid"]]["question"],
                "gold_answer": qa[r["qid"]]["answer"],
                "answer": r["answer"],
                "context": "\n\n".join(chunk_text[c] for c in r["chunk_ids"]),
                "human_correctness": "",
                "human_faithful": "",
                "judge_correctness": SCORE_TO_LABEL[r["correctness"]],
                "judge_faithful": str(bool(r["faithful"])).lower(),
                "judge_reason": r["reason"],
            })
    print(f"Wrote {len(sample)} rows from {result['eval_id']} to {LABELS_PATH}")


def score() -> None:
    with LABELS_PATH.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["human_correctness"].strip() and r["human_faithful"].strip()]
    if not rows:
        raise SystemExit(f"No labelled rows yet in {LABELS_PATH}.")

    norm = lambda s: s.strip().lower()  # noqa: E731
    corr = [(norm(r["human_correctness"]), norm(r["judge_correctness"])) for r in rows]
    faith = [(norm(r["human_faithful"]), norm(r["judge_faithful"])) for r in rows]
    # Binary view: is the answer acceptable (correct) or not (partial/incorrect)?
    binary = [(h == "correct", j == "correct") for h, j in corr]

    agree = lambda pairs: sum(h == j for h, j in pairs) / len(pairs)  # noqa: E731
    print(f"{len(rows)} labelled answers\n")
    print(f"correctness, 3 levels:          {agree(corr):.0%} agreement")
    print(f"correctness, correct vs not:    {agree(binary):.0%} agreement")
    print(f"faithfulness:                   {agree(faith):.0%} agreement\n")
    print("Correctness disagreements (human -> judge):")
    for (h, j), count in Counter(p for p in corr if p[0] != p[1]).most_common():
        print(f"  {h:9} -> {j:9} {count}")
    lenient = sum(1 for h, j in binary if j and not h)
    strict = sum(1 for h, j in binary if h and not j)
    print(f"\nJudge more lenient than you: {lenient}   more strict: {strict}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "score"
    if command == "export":
        export(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        score()

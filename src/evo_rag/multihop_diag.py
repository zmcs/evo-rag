"""Why do multi-hop answers fail? Free diagnosis from saved evaluation results (no API calls).

Reads the evaluations run by `decompose_check --eval` (the multi-hop + numerical ones) and
puts every multi-hop question into one bucket:

  correct            judged correct
  retrieval miss     not correct, and some evidence quote was not in the context
  synthesis failure  not correct although ALL evidence was in the context:
                       - abstained  (the model said the context lacks the answer)
                       - partial    (judged partially correct)
                       - incorrect  (judged wrong)

Then it prints every synthesis failure in full, to judge by eye whether the model or the
judge is at fault.

  uv run python -m evo_rag.multihop_diag
  uv run python -m evo_rag.multihop_diag --all     # also print the retrieval misses
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from evo_rag.config import PipelineConfig
from evo_rag.landscape import short
from evo_rag.proxy import evidence_found
from evo_rag.retrieval import build_retriever

RESULTS_DIR = Path("data/results")
QA_PATH = Path("data/processed/qa.jsonl")


def bucket(record: dict, evidence_complete: bool) -> str:
    if record["correctness"] == 1.0:
        return "correct"
    if not evidence_complete:
        return "retrieval miss"
    if record["abstained"]:
        return "synthesis: abstained"
    return "synthesis: partial" if record["correctness"] == 0.5 else "synthesis: incorrect"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose multi-hop failures.")
    parser.add_argument("--all", action="store_true", help="also print the retrieval misses")
    parser.add_argument("--type", default="multi_hop", help="question type to diagnose")
    args = parser.parse_args()

    qa = {q["qid"]: q for q in map(json.loads, QA_PATH.read_text(encoding="utf-8").splitlines())}
    evals = [json.loads(line) for line in (RESULTS_DIR / "evals.jsonl").read_text().splitlines()]
    # The latest evaluation of each pipeline + mode restricted to question types (decompose_check --eval).
    latest = {}
    for e in evals:
        if e.get("question_types") and args.type in e["question_types"]:
            latest[json.dumps(e["config"], sort_keys=True)] = e
    if not latest:
        raise SystemExit("No type-restricted evaluations found. Run `decompose_check --eval` first.")

    for e in latest.values():
        config = PipelineConfig(**e["config"])
        chunks = {c.chunk_id: c for c in build_retriever(config).chunks}
        details = [json.loads(line) for line in (RESULTS_DIR / "details" / f"{e['eval_id']}.jsonl").read_text().splitlines()]
        rows = []
        for r in details:
            if r["type"] != args.type:
                continue
            q = qa[r["qid"]]
            context = [chunks[c] for c in r["chunk_ids"]]
            complete = evidence_found([quote for _, quote in q["evidence"]], context, config.chunk_header) == 1.0
            rows.append((bucket(r, complete), r, q))

        counts = Counter(b for b, _, _ in rows)
        print("=" * 100)
        print(f"{short(config)} | query_mode={config.query_mode}   ({len(rows)} {args.type} questions)")
        for name in ["correct", "retrieval miss", "synthesis: abstained", "synthesis: partial", "synthesis: incorrect"]:
            print(f"  {name:22} {counts.get(name, 0):3d}")

        shown = [x for x in rows if x[0].startswith("synthesis") or (args.all and x[0] == "retrieval miss")]
        for b, r, q in shown:
            print(f"\n  --- {r['qid']} [{b}]")
            print(f"  Q:      {q['question']}")
            print(f"  GOLD:   {q['answer']}")
            print(f"  ANSWER: {r['answer']}")
            print(f"  JUDGE:  {r['reason']}")
            if r.get("queries"):
                print(f"  PLAN:   {' | '.join(r['queries'][1:]) or '(question only)'}")
        print()


if __name__ == "__main__":
    main()

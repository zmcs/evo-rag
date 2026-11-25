"""The fitness function: evaluate(config, n_questions) -> metrics.

  - Questions come in a fixed, type-stratified order, so a smaller n is always a prefix
    of a larger one (n=50 is a subset of n=178). That is what multi-fidelity needs.
  - `repeat` re-asks the API instead of using the cache, to measure noise.
  - Every evaluation is appended to data/results/evals.jsonl; per-question details go to
    data/results/details/<eval_id>.jsonl.

Run:  uv run python -m evo_rag.evaluate --n 50
"""

import argparse
import hashlib
import json
import random
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from evo_rag.config import PipelineConfig
from evo_rag.generation import generate_answer
from evo_rag.judge import judge_answer
from evo_rag.llm import LLM
from evo_rag.planner import retrieve
from evo_rag.retrieval import build_retriever
from evo_rag.retrieval_eval import score_question

QA_PATH = Path("data/processed/qa.jsonl")
RESULTS_DIR = Path("data/results")


def qa_version() -> str:
    """Short hash of the QA file, so every result records which QA set produced it."""
    return hashlib.sha1(QA_PATH.read_bytes()).hexdigest()[:8]


def ordered_questions(split: str = "dev", seed: int = 0) -> list[dict]:
    """All questions of a split in a fixed order that interleaves question types proportionally."""
    with QA_PATH.open(encoding="utf-8") as f:
        qa = [json.loads(line) for line in f if line.strip()]
    by_type = defaultdict(list)
    for q in qa:
        if q["split"] == split:
            by_type[q["type"]].append(q)
    rng = random.Random(seed)
    keyed = []
    for questions in by_type.values():
        rng.shuffle(questions)
        keyed += [((i + 0.5) / len(questions), q["qid"], q) for i, q in enumerate(questions)]
    return [q for *_, q in sorted(keyed)]


def evaluate(
    config: PipelineConfig,
    n_questions: int | None = None,
    split: str = "dev",
    seed: int = 0,
    repeat: int = 0,
    llm: LLM | None = None,
    log: bool = True,
    question_types: list[str] | None = None,
    **retriever_kwargs,
) -> dict:
    """question_types restricts the evaluation to some types, e.g. ["multi_hop"]."""
    llm = llm or LLM()
    questions = [q for q in ordered_questions(split, seed) if not question_types or q["type"] in question_types]
    questions = questions[:n_questions]
    retriever = build_retriever(config, **retriever_kwargs)

    records = []
    for q in questions:
        start = time.perf_counter()
        found = retrieve(q["question"], retriever, config, llm, cache_salt=repeat)
        chunks = found.chunks
        retrieval_s = time.perf_counter() - start + found.plan_latency_s

        response = generate_answer(q["question"], chunks, config, llm, cache_salt=repeat)
        verdict = judge_answer(q, response.text, chunks, llm, cache_salt=repeat)
        retrieved_docs = [c.doc_id for c in chunks]
        record = {
            "qid": q["qid"],
            "type": q["type"],
            "answer": response.text,
            "retrieved_docs": retrieved_docs,
            "chunk_ids": [c.chunk_id for c in chunks],
            **asdict(verdict),
            "gen_cost_usd": response.cost_usd + found.plan_cost_usd,  # pipeline cost incl. planning
            "queries": found.queries,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens + response.thinking_tokens,
            "latency_s": retrieval_s + response.latency_s,
        }
        if q["type"] != "unanswerable":
            record |= score_question(retrieved_docs, set(q["doc_ids"]))
        records.append(record)

    metrics = summarise(records)
    result = {
        "eval_id": f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}",  # unique per evaluation
        "config": config.model_dump(),
        "split": split,
        "n_questions": len(questions),
        "seed": seed,
        "repeat": repeat,
        "qa_version": qa_version(),
        "question_types": question_types,
        "metrics": metrics,
    }
    if log:
        save(result, records)
    return result


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def summarise(records: list[dict]) -> dict:
    answerable = [r for r in records if r["type"] != "unanswerable"]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]
    metrics = {
        # quality
        "correctness": mean(r["correctness"] for r in records),
        "faithfulness": mean(r["faithful"] for r in records),
        "abstain_when_unanswerable": mean(r["abstained"] for r in unanswerable),
        "abstain_when_answerable": mean(r["abstained"] for r in answerable),
        # retrieval (answerable questions only)
        "recall": mean(r["recall"] for r in answerable),
        "all_gold_found": mean(r["all"] for r in answerable),
        "mrr": mean(r["mrr"] for r in answerable),
        # cost of the pipeline itself (generation only: the judge is measurement, not pipeline)
        "cost_usd_per_query": mean(r["gen_cost_usd"] for r in records),
        "latency_s_per_query": mean(r["latency_s"] for r in records),
        "input_tokens_per_query": mean(r["input_tokens"] for r in records),
        # bookkeeping
        "judge_cost_usd": sum(r["cost_usd"] for r in records),
        "judge_errors": sum(r["judge_error"] for r in records),
    }
    for qtype in sorted({r["type"] for r in records}):
        metrics[f"correctness_{qtype}"] = mean(r["correctness"] for r in records if r["type"] == qtype)
    return metrics


def save(result: dict, records: list[dict]) -> None:
    (RESULTS_DIR / "details").mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "evals.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    with (RESULTS_DIR / "details" / f"{result['eval_id']}.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one pipeline config.")
    parser.add_argument("--n", type=int, default=None, help="number of questions (default: all)")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--repeats", type=int, default=1, help=">1 measures noise (fresh API calls)")
    parser.add_argument("--retrieval", default="hybrid", choices=["dense", "bm25", "hybrid"])
    args = parser.parse_args()
    if args.split == "test":
        print("WARNING: evaluating on the held-out test split.")

    config = PipelineConfig(retrieval=args.retrieval)
    llm = LLM()
    for repeat in range(args.repeats):
        result = evaluate(config, args.n, args.split, repeat=repeat, llm=llm)
        print(f"\n=== {result['eval_id']}  ({result['n_questions']} {args.split} questions, repeat {repeat}) ===")
        for name, value in result["metrics"].items():
            print(f"  {name:28} {value:.4f}" if isinstance(value, float) else f"  {name:28} {value}")
    print(f"\nTotal API spend so far: ${llm.total_spend():.4f}")


if __name__ == "__main__":
    main()

"""The finish line: a quick dev check, then ONE evaluation on the held-out test set.

  uv run python -m evo_rag.final_eval --check          # dry run of the dev check
  uv run python -m evo_rag.final_eval --check --yes    # cheap pipeline with / without LoRA, on dev (~$0.05)
  uv run python -m evo_rag.final_eval --test           # dry run of the test evaluation
  uv run python -m evo_rag.final_eval --test --yes     # frozen pipelines on the test set (~$0.30)

The pipelines are frozen here, in code, before the test set is touched:
  baseline   the Week 1 default pipeline (for the before/after comparison)
  cheap      the cheap end of the front, with the Week 4 LoRA embedder
  best       the top of the front: hybrid, k=13, query planning, combine prompt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from evo_rag.config import PipelineConfig
from evo_rag.evaluate import evaluate, ordered_questions
from evo_rag.landscape import short
from evo_rag.llm import LLM
from evo_rag.validate_front import estimate_usd

RESULTS_DIR = Path("data/results")

BASELINE = PipelineConfig()  # Week 1 defaults: dense bge-small, recursive 800 + header, k=5, default prompt
CHEAP = PipelineConfig(retrieval="dense", embedder="bge-small-lora", chunker="recursive", chunk_size=600,
                       chunk_overlap=0.0, chunk_header=True, top_k=5, prompt_template="combine")
BEST = PipelineConfig(retrieval="hybrid", hybrid_alpha=0.37, embedder="bge-small", chunker="recursive",
                      chunk_size=1600, chunk_overlap=0.0, chunk_header=True, top_k=13,
                      query_mode="decompose", prompt_template="combine")
# Approximate input tokens per query (measured on dev), only for the cost estimate.
TOKENS = {"baseline": 820, "cheap": 900, "best": 4100}


def confidence(eval_id: str) -> float:
    """95% half-width of mean correctness across questions."""
    rows = [json.loads(line) for line in (RESULTS_DIR / "details" / f"{eval_id}.jsonl").read_text().splitlines()]
    scores = np.array([r["correctness"] for r in rows])
    return float(1.96 * scores.std(ddof=1) / np.sqrt(len(scores)))


def run(pipelines: dict, split: str, n: int | None, llm: LLM) -> list[dict]:
    out = []
    for name, config in pipelines.items():
        print(f"\n[{name}] {short(config)}", flush=True)
        result = evaluate(config, n, split=split, llm=llm)
        m = result["metrics"]
        ci = confidence(result["eval_id"])
        print(f"  correctness {m['correctness']:.3f} ± {ci:.3f} | single {m.get('correctness_single_hop', float('nan')):.3f} "
              f"| multi {m.get('correctness_multi_hop', float('nan')):.3f} | numerical "
              f"{m.get('correctness_numerical', float('nan')):.3f} | declines unanswerable "
              f"{m['abstain_when_unanswerable']:.3f} | faithfulness {m['faithfulness']:.3f} | "
              f"${m['cost_usd_per_query'] * 1000:.3f}/1k q", flush=True)
        out.append({"name": name, "config": config.model_dump(), "split": split, "n": result["n_questions"],
                    "ci95": ci, "metrics": m})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Dev check and final test evaluation.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="dev check: cheap pipeline with / without LoRA")
    group.add_argument("--test", action="store_true", help="the one-time evaluation on the test set")
    parser.add_argument("--yes", action="store_true", help="run (otherwise only estimate the cost)")
    args = parser.parse_args()

    llm = LLM()
    if args.check:
        pipelines = {"cheap, base bge-small": CHEAP.model_copy(update={"embedder": "bge-small"}),
                     "cheap, LoRA": CHEAP}
        split, n = "dev", 50
        estimate = sum(estimate_usd(TOKENS["cheap"], n, CHEAP.generator) for _ in pipelines)
    else:
        pipelines = {"baseline": BASELINE, "cheap": CHEAP, "best": BEST}
        split, n = "test", None
        n_test = len(ordered_questions("test"))
        estimate = sum(estimate_usd(TOKENS[k], n_test, c.generator) for k, c in pipelines.items())
        estimate += n_test * 0.00025  # query plans for the test questions

    remaining = llm.budget_usd - llm.total_spend()
    print(f"{'Dev check' if args.check else 'FINAL TEST EVALUATION'}: {len(pipelines)} pipelines on {split}")
    for name, config in pipelines.items():
        print(f"  {name:22} {short(config)}")
    print(f"Estimated cost: ~${estimate:.2f}   (spent so far ${llm.total_spend():.2f}; ${remaining:.2f} left)")
    if estimate > remaining:
        print("Refusing to run: the estimate exceeds what is left under the budget.")
        return
    if args.test:
        print("The test set should be evaluated once. Results are cached, so re-running reprints the same numbers.")
    if not args.yes:
        print("Dry run only. Re-run with --yes to evaluate.")
        return

    results = run(pipelines, split, n, llm)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = "check" if args.check else "test"
    (RESULTS_DIR / f"final_{tag}_{stamp}.json").write_text(json.dumps(results, indent=1))
    print(f"\nSaved to data/results/final_{tag}_{stamp}.json")
    print(f"Total API spend so far: ${llm.total_spend():.4f}")


if __name__ == "__main__":
    main()

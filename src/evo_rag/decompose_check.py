"""Week 3, step 1: does query decomposition help, and where?

Uses the pipelines validated in Stage 2 (latest validate_*.json) and compares
query_mode="single" with query_mode="decompose" at the same top_k.

  uv run python -m evo_rag.decompose_check            # dry run: estimates only
  uv run python -m evo_rag.decompose_check --plan     # plan all dev questions (~$0.04),
                                                      # then compare evidence recall for free
  uv run python -m evo_rag.decompose_check --eval     # paid: real correctness on the
                                                      # multi-hop + numerical dev questions
"""

import argparse
import glob
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from evo_rag.config import PipelineConfig
from evo_rag.evaluate import evaluate
from evo_rag.landscape import short
from evo_rag.llm import LLM, PRICES_USD_PER_M
from evo_rag.planner import PLANNER_MODEL
from evo_rag.proxy import ProxyEvaluator
from evo_rag.retrieval_eval import load_questions
from evo_rag.validate_front import estimate_usd

RESULTS_DIR = Path("data/results")
TYPES = ["single_hop", "multi_hop", "numerical"]
EVAL_TYPES = ["multi_hop", "numerical"]


def latest_validated() -> list[dict]:
    files = glob.glob(str(RESULTS_DIR / "validate_*.json"))
    if not files:
        raise SystemExit("No validate_*.json found. Run validate_front first.")
    return json.loads(Path(max(files, key=os.path.getmtime)).read_text())


def by_type(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r["type"]].append(r[key])
    return {t: float(np.mean(v)) for t, v in groups.items()} | {"all": float(np.mean([r[key] for r in rows]))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single vs decomposed queries.")
    parser.add_argument("--plan", action="store_true", help="plan queries (~$0.04), then free comparison")
    parser.add_argument("--eval", action="store_true", help="paid real-LLM check on multi-hop + numerical")
    parser.add_argument("--eval-configs", type=int, default=2, help="pipelines in the paid check")
    args = parser.parse_args()

    validated = sorted(latest_validated(), key=lambda r: -r.get("correctness_full", r["correctness"]))
    configs = [PipelineConfig(**r["config"]) for r in validated]
    questions = load_questions("dev")
    eval_n = sum(q["type"] in EVAL_TYPES for q in questions)
    llm = LLM()

    price_in, price_out = PRICES_USD_PER_M[PLANNER_MODEL]
    plan_cost = len(questions) * (330 * price_in + 50 * price_out) / 1_000_000
    eval_cost = sum(
        estimate_usd(r["proxy_tokens"] + 1000, eval_n, c.generator)  # +planning, pessimistic
        + estimate_usd(r["proxy_tokens"], eval_n, c.generator)
        for r, c in zip(validated[: args.eval_configs], configs)
    )
    print(f"{len(configs)} validated pipelines, {len(questions)} answerable dev questions "
          f"({eval_n} multi-hop + numerical)\n")
    print(f"  --plan  plan every question once:                 ~${plan_cost:.2f}")
    print(f"  --eval  {args.eval_configs} pipelines x 2 modes x {eval_n} questions:        "
          f"~${eval_cost:.2f} (pessimistic; cached answers are free)")
    print(f"  spent so far ${llm.total_spend():.2f} of the ${llm.budget_usd:.2f} budget")
    if not (args.plan or args.eval):
        print("\nDry run only. Use --plan first, then --eval.")
        return

    proxy = ProxyEvaluator(llm=llm)
    results = []
    print("\nFree comparison: evidence recall, single -> decompose (tokens include the planning call)")
    print(f"{'tokens':>13}  {'all':>13}  {'single-hop':>13}  {'multi-hop':>13}  {'numerical':>13}  pipeline")
    for config in configs:
        single, _ = proxy.per_question(config)
        decomposed, _ = proxy.per_question(config.model_copy(update={"query_mode": "decompose"}))
        ev_s, ev_d = by_type(single, "evidence_recall"), by_type(decomposed, "evidence_recall")
        tok_s, tok_d = by_type(single, "tokens")["all"], by_type(decomposed, "tokens")["all"]
        cells = "  ".join(f"{ev_s[t]:.2f} -> {ev_d[t]:.2f}" for t in ["all"] + TYPES)
        print(f"{tok_s:5.0f} -> {tok_d:5.0f}  {cells}  {short(config)}")
        results.append({"config": config.model_dump(), "evidence_single": ev_s, "evidence_decompose": ev_d,
                        "recall_single": by_type(single, "recall"), "recall_decompose": by_type(decomposed, "recall"),
                        "tokens_single": tok_s, "tokens_decompose": tok_d})
    n_queries = [len(r["queries"]) for r in decomposed]
    print(f"\nPlans: {np.mean(n_queries):.1f} queries per question on average (including the question itself)")
    example = next((r for r in decomposed if r["type"] == "multi_hop" and len(r["queries"]) > 1), None)
    if example:
        print("Example plan for a multi-hop question:")
        for q in example["queries"]:
            print(f"   - {q}")

    if args.eval:
        print(f"\nPaid check on {eval_n} multi-hop + numerical dev questions")
        for config in configs[: args.eval_configs]:
            print(f"\n{short(config)}", flush=True)
            for mode in ("single", "decompose"):
                m = evaluate(config.model_copy(update={"query_mode": mode}), llm=llm,
                             question_types=EVAL_TYPES)["metrics"]
                print(f"  {mode:9}  correctness {m['correctness']:.3f}  "
                      f"(multi-hop {m['correctness_multi_hop']:.3f}, numerical {m['correctness_numerical']:.3f})  "
                      f"faithfulness {m['faithfulness']:.3f}  ${m['cost_usd_per_query'] * 1000:.3f}/1k q",
                      flush=True)
                results.append({"config": config.model_dump(), "mode": mode, "metrics": m})

    stamp = time.strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"decompose_{stamp}.json").write_text(json.dumps(results, indent=1))
    print(f"\nTotal API spend so far: ${llm.total_spend():.4f}")


if __name__ == "__main__":
    main()

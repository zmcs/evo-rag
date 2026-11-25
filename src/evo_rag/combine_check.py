"""Week 3, step 2: does the `combine` prompt turn refusals into correct multi-hop answers,
without answering questions that should be refused?

Compares prompt_template "default" and "combine" for the best validated pipelines, with
query planning on, on the dev multi-hop, numerical AND unanswerable questions (the last
group checks that the looser prompt still declines when it should).

  uv run python -m evo_rag.combine_check          # dry run: cost estimate
  uv run python -m evo_rag.combine_check --yes    # run it
"""

import argparse
import json
import time
from pathlib import Path

from evo_rag.config import PipelineConfig
from evo_rag.decompose_check import latest_validated
from evo_rag.evaluate import evaluate, ordered_questions
from evo_rag.landscape import short
from evo_rag.llm import LLM
from evo_rag.validate_front import estimate_usd

RESULTS_DIR = Path("data/results")
TYPES = ["multi_hop", "numerical", "unanswerable"]
PROMPTS = ["default", "combine"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the default and combine prompts.")
    parser.add_argument("--configs", type=int, default=2, help="best validated pipelines to test")
    parser.add_argument("--yes", action="store_true", help="run (otherwise only estimate the cost)")
    args = parser.parse_args()

    validated = sorted(latest_validated(), key=lambda r: -r.get("correctness_full", r["correctness"]))[: args.configs]
    questions = [q for q in ordered_questions("dev") if q["type"] in TYPES]
    llm = LLM()
    # Pessimistic: every answer paid again (+1000 tokens for planning); cached ones are free.
    estimate = sum(estimate_usd(r["proxy_tokens"] + 1000, len(questions), "gemini-3.5-flash-lite")
                   for r in validated) * len(PROMPTS)
    print(f"{len(validated)} pipelines x {len(PROMPTS)} prompts x {len(questions)} dev questions "
          f"(multi-hop, numerical, unanswerable), query planning on")
    print(f"Estimated cost: up to ~${estimate:.2f} (pessimistic; cached answers are free)   "
          f"spent so far ${llm.total_spend():.2f} of the ${llm.budget_usd:.2f} budget")
    if not args.yes:
        print("Dry run only. Re-run with --yes to evaluate.")
        return

    results = []
    for r in validated:
        base = PipelineConfig(**r["config"]).model_copy(update={"query_mode": "decompose"})
        print(f"\n{short(base)} | planning on", flush=True)
        for prompt in PROMPTS:
            m = evaluate(base.model_copy(update={"prompt_template": prompt}), llm=llm, question_types=TYPES)["metrics"]
            print(f"  {prompt:8} multi-hop {m['correctness_multi_hop']:.3f} | numerical {m['correctness_numerical']:.3f} | "
                  f"declines unanswerable {m['abstain_when_unanswerable']:.3f} | refuses answerable "
                  f"{m['abstain_when_answerable']:.3f} | faithfulness {m['faithfulness']:.3f} | "
                  f"${m['cost_usd_per_query'] * 1000:.3f}/1k q", flush=True)
            results.append({"config": base.model_dump(), "prompt": prompt, "metrics": m})

    stamp = time.strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"combine_{stamp}.json").write_text(json.dumps(results, indent=1))
    print("\nWhat to look for: higher multi-hop with `combine`, while 'declines unanswerable' stays near 1.0 "
          "and faithfulness stays >= 0.95.")
    print(f"Total API spend so far: ${llm.total_spend():.4f}")


if __name__ == "__main__":
    main()

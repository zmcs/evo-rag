"""Stage 2: validate the NSGA-II proxy front with the real LLM (the only step that costs money).

  1. Pick up to --max-configs pipelines spread along the proxy front (cheap to expensive),
     skipping those with recall below --min-recall (too weak to be worth paying for).
  2. Evaluate each on the first --n dev questions with the real generator and judge.
  3. Re-evaluate the --full best ones on the whole dev set.
  4. Compare the real correctness-vs-cost front with the random-search front from the
     landscape run (already paid for, same 50 questions), and check how well proxy recall
     predicted real correctness.

  uv run python -m evo_rag.validate_front            # dry run: plan + cost estimate
  uv run python -m evo_rag.validate_front --yes      # spend it
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from evo_rag.config import PipelineConfig
from evo_rag.evaluate import evaluate, ordered_questions
from evo_rag.landscape import pareto_front, short, spearman
from evo_rag.llm import LLM, PRICES_USD_PER_M

RESULTS_DIR = Path("data/results")


def latest(pattern: str) -> Path:
    files = glob.glob(str(RESULTS_DIR / pattern))
    if not files:
        raise SystemExit(f"No {pattern} in {RESULTS_DIR}. Run the earlier stage first.")
    return Path(max(files, key=os.path.getmtime))


def pick_spread(front: list[dict], max_configs: int, min_quality: float, metric: str) -> list[dict]:
    """Up to max_configs pipelines spread evenly (in log tokens) along the front."""
    usable = sorted((r for r in front if r[metric] >= min_quality), key=lambda r: r["tokens"])
    if len(usable) <= max_configs:
        return usable
    targets = np.linspace(np.log10(usable[0]["tokens"]), np.log10(usable[-1]["tokens"]), max_configs)
    chosen = []
    for t in targets:
        best = min((r for r in usable if r not in chosen), key=lambda r: abs(np.log10(r["tokens"]) - t))
        chosen.append(best)
    return sorted(chosen, key=lambda r: r["tokens"])


def estimate_usd(tokens: float, n_questions: int, model: str) -> float:
    """Cost of generating and judging n questions; the judge also reads the context."""
    price_in, price_out = PRICES_USD_PER_M[model]
    generation = tokens * price_in + 80 * price_out
    judging = 0.85 * ((tokens + 350) * price_in + 50 * price_out)  # ~15% abstain, no judge call
    return n_questions * (generation + judging) / 1_000_000


def plot(nsga_rows: list[dict], random_rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, blue, orange = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=surface)
    ax.set_facecolor(surface)
    for rows, color, name in ((random_rows, orange, "Random search (landscape)"),
                              (nsga_rows, blue, "NSGA-II on free proxy")):
        pts = [(r["correctness"], r["cost"] * 1000) for r in rows]
        front = sorted((pts[i] for i in pareto_front([(c, k / 1000) for c, k in pts])), key=lambda p: p[1])
        ax.scatter([p[1] for p in pts], [p[0] for p in pts], s=28, color=color, alpha=0.35, linewidths=0, zorder=2)
        ax.step([p[1] for p in front], [p[0] for p in front], where="post", color=color, linewidth=2,
                zorder=3, label=name)
        ax.scatter([p[1] for p in front], [p[0] for p in front], s=56, color=color, edgecolors=surface,
                   linewidths=2, zorder=4)
    ax.set_xscale("log")
    plain = plt.FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(plain)
    ax.set_xlabel("Generation cost (USD per 1,000 queries, log scale)", color=muted)
    ax.set_ylabel(f"Correctness (first {nsga_rows[0]['n']} dev questions)", color=muted)
    ax.set_title("Real correctness vs cost: NSGA-II front vs random search", color=ink, loc="left")
    ax.grid(color="#e8e7e3", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d4d3cd")
    ax.tick_params(which="both", colors=muted)
    ax.legend(frameon=False, labelcolor=muted, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    print(f"Plot saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the proxy front with the real LLM.")
    parser.add_argument("--max-configs", type=int, default=8)
    parser.add_argument("--min-quality", type=float, default=None,
                        help="skip front members below this proxy score (default 0.75 recall / 0.65 evidence)")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--full", type=int, default=3, help="best configs re-run on the whole dev set")
    parser.add_argument("--yes", action="store_true", help="run (otherwise only estimate the cost)")
    args = parser.parse_args()

    evolve_file, landscape_file = latest("evolve_*.json"), latest("landscape_*.json")
    run = json.loads(evolve_file.read_text())
    metric = run["settings"].get("objective", "recall")
    min_quality = args.min_quality if args.min_quality is not None else (0.75 if metric == "recall" else 0.65)
    picks = pick_spread(run["nsga2_front"], args.max_configs, min_quality, metric)
    random_rows = json.loads(landscape_file.read_text())
    n_full = len(ordered_questions("dev"))
    llm = LLM()

    configs = [PipelineConfig(**p["config"]) for p in picks]
    low = [estimate_usd(p["tokens"], args.n, c.generator) for p, c in zip(picks, configs)]
    extra = sorted((estimate_usd(p["tokens"], n_full - args.n, c.generator) for p, c in zip(picks, configs)),
                   reverse=True)[: args.full]  # pessimistic: assume the most expensive ones win
    print(f"From {evolve_file.name}: {len(picks)} pipelines on the proxy front ({metric} >= {min_quality})\n")
    for p, c, cost in zip(picks, configs, low):
        print(f"  ~${cost:.3f}  {metric} {p[metric]:.3f}  ~{p['tokens']:5.0f} tok  {short(c)}")
    total = sum(low) + sum(extra)
    print(f"\nEstimated cost: ~${total:.2f}   (spent so far ${llm.total_spend():.2f} "
          f"of the ${llm.budget_usd:.2f} budget)")
    print("Pipelines already evaluated in the landscape run are served from the cache for free.")
    if not args.yes:
        print("Dry run only. Re-run with --yes to evaluate.")
        return

    rows = []
    for p, config in zip(picks, configs):
        print(f"\n{short(config)}", flush=True)
        m = evaluate(config, args.n, llm=llm)["metrics"]
        rows.append({"config": config.model_dump(), "short": short(config), "n": args.n,
                     "proxy_recall": p[metric], "proxy_tokens": p["tokens"],
                     "correctness": m["correctness"], "faithfulness": m["faithfulness"],
                     "cost": m["cost_usd_per_query"], "real_tokens": m["input_tokens_per_query"]})
        print(f"  correctness {m['correctness']:.3f}  faithfulness {m['faithfulness']:.3f}  "
              f"${m['cost_usd_per_query'] * 1000:.3f}/1k q  {m['input_tokens_per_query']:.0f} tok")

    for r in sorted(rows, key=lambda r: -r["correctness"])[: args.full]:
        print(f"\nFull dev set: {r['short']}", flush=True)
        r["correctness_full"] = evaluate(PipelineConfig(**r["config"]), None, llm=llm)["metrics"]["correctness"]

    print("\n" + "=" * 110)
    print(f"{'proxy':>7} {'corr':>6} {'full':>6} {'faith':>6} {'$/1k q':>7} {'tok est':>8} {'tok real':>8}  pipeline")
    for r in rows:
        full = f"{r['correctness_full']:.3f}" if "correctness_full" in r else "     -"
        print(f"{r['proxy_recall']:7.3f} {r['correctness']:6.3f} {full:>6} {r['faithfulness']:6.3f} "
              f"{r['cost'] * 1000:7.3f} {r['proxy_tokens']:8.0f} {r['real_tokens']:8.0f}  {r['short']}")

    if len(rows) >= 3:
        print(f"\nProxy check: rank correlation {metric} vs real correctness = "
              f"{spearman([r['proxy_recall'] for r in rows], [r['correctness'] for r in rows]):.2f}")
    violations = [r["short"] for r in rows if r["faithfulness"] < 0.95]
    print(f"Faithfulness constraint (>= 0.95): {'all satisfied' if not violations else violations}")

    nsga_front = {i for i in pareto_front([(r["correctness"], r["cost"]) for r in rows])}
    rand_front = pareto_front([(r["correctness"], r["cost"]) for r in random_rows])
    print(f"Real front sizes: NSGA-II {len(nsga_front)}, random search {len(rand_front)}")
    print("Best correctness: NSGA-II "
          f"{max(r['correctness'] for r in rows):.3f}  vs random {max(r['correctness'] for r in random_rows):.3f}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"validate_{stamp}.json").write_text(json.dumps(rows, indent=1))
    plot(rows, [r | {"n": args.n} for r in random_rows], RESULTS_DIR / f"validate_{stamp}.png")
    print(f"Total API spend so far: ${llm.total_spend():.4f}")


if __name__ == "__main__":
    main()

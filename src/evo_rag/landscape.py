"""Landscape check: is the search space worth optimizing?

Evaluates random configs (plus the default) at low fidelity, then re-evaluates a spread of
them on the full dev set, and answers three questions:
  1. Do configs differ meaningfully in correctness?
  2. Do correctness and cost conflict (is there a real Pareto front)?
  3. Does the low-fidelity ranking predict the full ranking?

First see the estimated cost, then run:
  uv run python -m evo_rag.landscape                 # dry run: estimate only
  uv run python -m evo_rag.landscape --yes           # actually run

Random configs also double as the random-search baseline for Week 2.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from evo_rag.chunking import chunk_corpus
from evo_rag.config import PipelineConfig
from evo_rag.evaluate import evaluate, ordered_questions
from evo_rag.generation import PROMPT_TEMPLATES
from evo_rag.llm import LLM, PRICES_USD_PER_M
from evo_rag.retrieval import load_corpus

RESULTS_DIR = Path("data/results")


def random_config(rng: random.Random) -> PipelineConfig:
    """Uniform sample from the search space (the reranker is not implemented yet, so it stays off)."""
    return PipelineConfig(
        chunker=rng.choice(["fixed", "recursive", "semantic"]),
        chunk_size=rng.randrange(200, 2001, 50),
        chunk_overlap=round(rng.uniform(0.0, 0.3), 2),
        chunk_header=rng.random() < 0.5,
        embedder=rng.choice(["bge-small", "bge-base", "e5-base"]),
        retrieval=rng.choice(["dense", "bm25", "hybrid"]),
        hybrid_alpha=round(rng.uniform(0.0, 1.0), 2),
        top_k=rng.randint(1, 20),
        prompt_template=rng.choice(sorted(PROMPT_TEMPLATES)),
    )


def estimate_cost_usd(config: PipelineConfig, docs: list[dict], n_questions: int) -> float:
    """Rough cost of generation + judging, from the average chunk length (no API calls).

    The semantic chunker is estimated with recursive chunks of the same size.
    """
    proxy = config.model_copy(update={"chunker": "recursive"}) if config.chunker == "semantic" else config
    chunks = chunk_corpus(docs, proxy)
    context_tokens = config.top_k * np.mean([len(c.text) for c in chunks]) / 4  # ~4 characters per token
    price_in, price_out = PRICES_USD_PER_M[config.generator]
    generation = (context_tokens + 200) * price_in + 80 * price_out
    judging = (context_tokens + 450) * price_in + 50 * price_out
    return n_questions * (generation + judging) / 1_000_000


def short(config: PipelineConfig) -> str:
    alpha = f" a={config.hybrid_alpha}" if config.retrieval == "hybrid" else ""
    embedder = "" if config.retrieval == "bm25" and config.chunker != "semantic" else f" {config.embedder}"
    return (
        f"{config.retrieval}{alpha}{embedder} | {config.chunker} {config.chunk_size}"
        f"{' +hdr' if config.chunk_header else ''} | k={config.top_k}"
        f"{' | planning' if config.query_mode == 'decompose' else ''} | {config.prompt_template}"
    )


def spearman(a: list[float], b: list[float]) -> float:
    ranks = lambda x: np.argsort(np.argsort(x))  # noqa: E731
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def pareto_front(points: list[tuple[float, float]]) -> list[int]:
    """Indices of points not dominated when maximizing correctness and minimizing cost."""
    front = []
    for i, (corr_i, cost_i) in enumerate(points):
        dominated = any(
            corr_j >= corr_i and cost_j <= cost_i and (corr_j > corr_i or cost_j < cost_i)
            for j, (corr_j, cost_j) in enumerate(points) if j != i
        )
        if not dominated:
            front.append(i)
    return front


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink_muted, accent, dominated = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#b5b4ad"
    front = set(pareto_front([(r["correctness"], r["cost"]) for r in rows]))

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=surface)
    ax.set_facecolor(surface)
    for i, r in enumerate(rows):
        on_front = i in front
        ax.scatter(
            r["cost"] * 1000, r["correctness"], s=64, zorder=3,
            color=accent if on_front else dominated, edgecolors=surface, linewidths=2,
        )
        if on_front:
            ax.annotate(
                r["label"], (r["cost"] * 1000, r["correctness"]), xytext=(8, 4),
                textcoords="offset points", fontsize=8, color=ink_muted,
            )
    xs = sorted((rows[i]["cost"] * 1000, rows[i]["correctness"]) for i in front)
    ax.step([x for x, _ in xs], [y for _, y in xs], where="post", color=accent, linewidth=2, zorder=2)

    ax.set_xlabel("Generation cost (USD per 1,000 queries)", color=ink_muted)
    ax.set_ylabel("Correctness (dev, low fidelity)", color=ink_muted)
    ax.set_title("Random configs: correctness vs cost (blue = Pareto front)", color=ink, loc="left")
    ax.grid(color="#e8e7e3", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d4d3cd")
    ax.tick_params(colors=ink_muted)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    print(f"Plot saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Landscape check over random configs.")
    parser.add_argument("--configs", type=int, default=12, help="number of random configs")
    parser.add_argument("--n", type=int, default=50, help="low-fidelity question count")
    parser.add_argument("--full", type=int, default=4, help="configs re-evaluated on the full dev set")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yes", action="store_true", help="run (otherwise only estimate the cost)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    configs = [PipelineConfig()] + [random_config(rng) for _ in range(args.configs)]
    labels = ["default"] + [f"r{i:02d}" for i in range(1, args.configs + 1)]

    docs = load_corpus()
    n_full = len(ordered_questions("dev"))
    low = [estimate_cost_usd(c, docs, args.n) for c in configs]
    # Full re-evaluation only pays for the questions beyond the cached low-fidelity prefix.
    extra_per_config = sorted((e * (n_full - args.n) / args.n for e in low), reverse=True)
    estimate = sum(low) + sum(extra_per_config[: args.full])  # pessimistic: the most expensive ones

    print(f"{len(configs)} configs, n={args.n}, then {args.full} on all {n_full} dev questions\n")
    for label, config, cost in zip(labels, configs, low):
        print(f"  {label:8} ~${cost:.3f}  {short(config)}")
    llm = LLM()
    print(f"\nEstimated cost: up to ~${estimate:.2f}   "
          f"(spent so far ${llm.total_spend():.2f} of ${llm.budget_usd:.2f} budget)")
    if not args.yes:
        print("Dry run only. Re-run with --yes to evaluate.")
        return

    rows = []
    for label, config in zip(labels, configs):
        print(f"\n[{label}] {short(config)}", flush=True)
        m = evaluate(config, args.n, llm=llm)["metrics"]
        rows.append({"label": label, "config": config.model_dump(), "short": short(config),
                     "correctness": m["correctness"], "faithfulness": m["faithfulness"],
                     "recall": m["recall"], "cost": m["cost_usd_per_query"],
                     "latency": m["latency_s_per_query"]})
        print(f"  correctness {m['correctness']:.3f}  faithfulness {m['faithfulness']:.3f}  "
              f"cost ${m['cost_usd_per_query']:.5f}/q  latency {m['latency_s_per_query']:.2f}s")

    # Re-evaluate a spread of configs (best to worst) on the full dev set.
    by_corr = sorted(range(len(rows)), key=lambda i: -rows[i]["correctness"])
    picks = sorted({by_corr[round(x)] for x in np.linspace(0, len(rows) - 1, args.full)})
    for i in picks:
        print(f"\n[{rows[i]['label']}] full dev set", flush=True)
        rows[i]["correctness_full"] = evaluate(PipelineConfig(**rows[i]["config"]), None, llm=llm)["metrics"]["correctness"]

    print("\n" + "=" * 100)
    print(f"{'config':8} {'corr':>6} {'full':>6} {'faith':>6} {'recall':>7} {'$/1k q':>7} {'lat s':>6}  description")
    for r in sorted(rows, key=lambda r: -r["correctness"]):
        full = f"{r['correctness_full']:.3f}" if "correctness_full" in r else "     -"
        print(f"{r['label']:8} {r['correctness']:6.3f} {full:>6} {r['faithfulness']:6.3f} {r['recall']:7.3f} "
              f"{r['cost'] * 1000:7.3f} {r['latency']:6.2f}  {r['short']}")

    corr = [r["correctness"] for r in rows]
    print(f"\n1. Spread of correctness: {min(corr):.3f} to {max(corr):.3f} (std {np.std(corr):.3f})")
    front = pareto_front([(r["correctness"], r["cost"]) for r in rows])
    print(f"2. Pareto front (max correctness, min cost): {[rows[i]['label'] for i in front]}")
    paired = [(r["correctness"], r["correctness_full"]) for r in rows if "correctness_full" in r]
    if len(paired) >= 3:
        print(f"3. Rank correlation, n={args.n} vs full ({len(paired)} configs): "
              f"{spearman(*zip(*paired)):.2f}  (few points: indicative only)")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"landscape_{stamp}.json").write_text(json.dumps(rows, indent=1))
    plot(rows, RESULTS_DIR / f"landscape_{stamp}.png")
    print(f"Total API spend so far: ${llm.total_spend():.4f}")


if __name__ == "__main__":
    main()

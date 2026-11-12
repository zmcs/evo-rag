"""Stage 1: NSGA-II on the free proxy objectives, against random search at equal budget.

Objectives (both minimised internally):
  -recall@k           (maximise retrieval recall: proxy for correctness)
  log10(tokens)       (minimise input tokens per query: proxy for cost)

Cost enters on a log scale: cost differences are multiplicative ("3x cheaper"), and the
front spans ~50x in tokens. A monotone transform leaves dominance (the Pareto front)
unchanged, but it makes crowding distance and hypervolume weight cheap and expensive
regions evenly instead of letting the expensive end dominate.

Budget is counted in *distinct pipelines evaluated*: that is what costs time (new indexes)
and, in Stage 2, money. Random search gets exactly the same number of distinct pipelines.

Run:  uv run python -m evo_rag.evolve --pop 20 --gens 15
No API calls. Time is spent building indexes (embedding the corpus) for new chunking settings.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import RankAndCrowding, binary_tournament
from pymoo.core.callback import Callback
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mixed import MixedVariableGA, MixedVariableMating
from pymoo.core.problem import ElementwiseProblem
from pymoo.indicators.hv import HV
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize

from evo_rag.proxy import ProxyEvaluator
from evo_rag.search_space import VARIABLES, config_key, random_genome, to_config

RESULTS_DIR = Path("data/results")
REF_POINT = np.array([0.0, 4.0])  # recall 0, 10^4 = 10k tokens/query: worse than any sensible config


OBJECTIVE = "recall"  # or "evidence_recall"; set from --objective


class ProxyProblem(ElementwiseProblem):
    def __init__(self, evaluator: ProxyEvaluator):
        super().__init__(vars=VARIABLES, n_obj=2)
        self.evaluator = evaluator
        self.trace: list[str] = []  # distinct pipeline keys, in the order first evaluated
        self._seen: set[str] = set()

    def _evaluate(self, x, out, *args, **kwargs):
        config = to_config(x)
        key = config_key(config)
        if key not in self._seen:
            self._seen.add(key)
            self.trace.append(key)
        result = self.evaluator(config)
        out["F"] = [-result[OBJECTIVE], np.log10(result["tokens"])]


class SamePipeline(ElementwiseDuplicateElimination):
    """Two genomes are duplicates when they build the same pipeline (inactive genes ignored)."""

    def is_equal(self, a, b):
        return config_key(to_config(a.X)) == config_key(to_config(b.X))


class Progress(Callback):
    def __init__(self, problem: ProxyProblem):
        super().__init__()
        self.problem, self.start = problem, time.time()

    def notify(self, algorithm):
        F = algorithm.opt.get("F")
        best = -F[:, 0].min()
        print(
            f"  gen {algorithm.n_gen:3d} | distinct pipelines {len(self.problem.trace):4d} | "
            f"front size {len(F):3d} | best {OBJECTIVE} {best:.3f} | {time.time() - self.start:6.0f}s",
            flush=True,
        )


def objectives(evaluator: ProxyEvaluator, key: str) -> np.ndarray:
    row = evaluator.cache[key]
    return np.array([-row[OBJECTIVE], np.log10(row["tokens"])])


def hv_curve(evaluator: ProxyEvaluator, trace: list[str]) -> list[float]:
    """Hypervolume of everything evaluated so far, after each new distinct pipeline."""
    hv, points, curve = HV(ref_point=REF_POINT), [], []
    for key in trace:
        points.append(objectives(evaluator, key))
        curve.append(float(hv(np.array(points))))
    return curve


def front_keys(evaluator: ProxyEvaluator, keys: list[str]) -> list[str]:
    """Non-dominated pipelines among `keys`, sorted by tokens."""
    F = {k: objectives(evaluator, k) for k in keys}
    front = [
        k for k in keys
        if not any(np.all(F[j] <= F[k]) and np.any(F[j] < F[k]) for j in keys if j != k)
    ]
    return sorted(set(front), key=lambda k: F[k][1])


def run_nsga2(evaluator: ProxyEvaluator, pop: int, gens: int, seed: int, selection: str) -> list[str]:
    """selection: 'tournament' (canonical NSGA-II: binary tournament on rank, then crowding)
    or 'random' (pymoo's mixed-variable default)."""
    problem = ProxyProblem(evaluator)
    parent_selection = (
        TournamentSelection(func_comp=binary_tournament) if selection == "tournament" else RandomSelection()
    )
    algorithm = MixedVariableGA(
        pop_size=pop,
        survival=RankAndCrowding(),
        mating=MixedVariableMating(selection=parent_selection, eliminate_duplicates=SamePipeline()),
        eliminate_duplicates=SamePipeline(),
        # As in pymoo's NSGA2 class: rank and crowd the initial population too, so the
        # first tournament already has rank and crowding distance to compare.
        advance_after_initial_infill=True,
    )
    # pymoo's NSGA-II tournament reads this from the algorithm: compare parents by
    # dominance first, then by crowding distance (as in its own NSGA2 class).
    algorithm.tournament_type = "comp_by_dom_and_crowding"
    minimize(problem, algorithm, ("n_gen", gens), seed=seed, callback=Progress(problem), verbose=False)
    return problem.trace


def run_random(evaluator: ProxyEvaluator, budget: int, seed: int) -> list[str]:
    rng, trace, seen = random.Random(seed + 1), [], set()
    start = time.time()
    while len(trace) < budget:
        config = to_config(random_genome(rng))
        key = config_key(config)
        if key in seen:
            continue
        seen.add(key)
        trace.append(key)
        evaluator(config)
        if len(trace) % 20 == 0:
            print(f"  random {len(trace):4d}/{budget} | {time.time() - start:6.0f}s", flush=True)
    return trace


def plot(evaluator, nsga_trace, random_trace, nsga_hv, random_hv, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
    blue, orange = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=surface)

    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        ax.grid(color=grid, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d4d3cd")
        ax.tick_params(which="both", colors=muted)

    # Left: all evaluated pipelines and each method's front.
    for trace, color, name in ((random_trace, orange, "Random search"), (nsga_trace, blue, "NSGA-II")):
        F = np.array([objectives(evaluator, k) for k in trace])
        ax1.scatter(10 ** F[:, 1], -F[:, 0], s=12, color=color, alpha=0.25, linewidths=0, zorder=2)
        front = np.array([objectives(evaluator, k) for k in front_keys(evaluator, trace)])
        ax1.step(10 ** front[:, 1], -front[:, 0], where="post", color=color, linewidth=2, zorder=3, label=name)
        ax1.scatter(10 ** front[:, 1], -front[:, 0], s=48, color=color, edgecolors=surface, linewidths=2, zorder=4)
    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax1.set_xlabel("Estimated input tokens per query (log scale)", color=muted)
    ax1.set_ylabel(f"{OBJECTIVE.replace('_', ' ').capitalize()} (dev)", color=muted)
    ax1.set_title("Proxy Pareto fronts (faint dots: every pipeline evaluated)", color=ink, loc="left", fontsize=11)
    ax1.legend(frameon=False, labelcolor=muted)

    # Right: hypervolume against number of distinct pipelines evaluated.
    x = np.arange(1, len(nsga_hv) + 1)
    ax2.plot(x, random_hv, color=orange, linewidth=2, label="Random search")
    ax2.plot(x, nsga_hv, color=blue, linewidth=2, label="NSGA-II")
    ax2.annotate("NSGA-II", (x[-1], nsga_hv[-1]), xytext=(4, 0), textcoords="offset points",
                 color=muted, fontsize=9, va="center")
    ax2.annotate("Random", (x[-1], random_hv[-1]), xytext=(4, 0), textcoords="offset points",
                 color=muted, fontsize=9, va="center")
    ax2.set_xlabel("Distinct pipelines evaluated", color=muted)
    ax2.set_ylabel("Hypervolume (higher is better)", color=muted)
    ax2.set_title("Search efficiency at equal budget", color=ink, loc="left", fontsize=11)
    ax2.legend(frameon=False, labelcolor=muted)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    print(f"Plot saved to {path}")


def main(**evaluator_kwargs) -> None:
    parser = argparse.ArgumentParser(description="NSGA-II vs random search on free proxy objectives.")
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objective", default="recall", choices=["recall", "evidence_recall"],
                        help="quality proxy to maximise (see check_proxy for which predicts correctness)")
    parser.add_argument("--selection", default="tournament", choices=["tournament", "random"],
                        help="parent selection: canonical NSGA-II tournament, or random")
    args = parser.parse_args()
    global OBJECTIVE
    OBJECTIVE = args.objective

    evaluator = ProxyEvaluator(**evaluator_kwargs)
    print(f"Proxy cache holds {len(evaluator.cache)} pipelines already scored.\n")
    print(f"NSGA-II: population {args.pop}, {args.gens} generations, objective {OBJECTIVE}, "
          f"{args.selection} parent selection")
    nsga_trace = run_nsga2(evaluator, args.pop, args.gens, args.seed, args.selection)
    print(f"\nRandom search: {len(nsga_trace)} distinct pipelines (same budget)")
    random_trace = run_random(evaluator, len(nsga_trace), args.seed)

    nsga_hv, random_hv = hv_curve(evaluator, nsga_trace), hv_curve(evaluator, random_trace)
    half = len(nsga_hv) // 2
    print(f"\nHypervolume after {half} pipelines:  NSGA-II {nsga_hv[half - 1]:.4f}   random {random_hv[half - 1]:.4f}")
    print(f"Hypervolume after {len(nsga_hv)} pipelines: NSGA-II {nsga_hv[-1]:.4f}   random {random_hv[-1]:.4f}")

    front = front_keys(evaluator, nsga_trace)
    print(f"\nNSGA-II proxy front ({len(front)} pipelines):")
    for k in front:
        row = evaluator.cache[k]
        c = row["config"]
        print(f"  recall {row['recall']:.3f}  evidence {row['evidence_recall']:.3f}  tokens {row['tokens']:6.0f}  "
              f"{c['retrieval']} {c['embedder']} | {c['chunker']} {c['chunk_size']} "
              f"ov={c['chunk_overlap']} {'+hdr' if c['chunk_header'] else ''} | k={c['top_k']}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "settings": vars(args),
        "nsga2_front": [evaluator.cache[k] for k in front],
        "random_front": [evaluator.cache[k] for k in front_keys(evaluator, random_trace)],
        "nsga2_hv": nsga_hv,
        "random_hv": random_hv,
        "nsga2_trace": nsga_trace,
        "random_trace": random_trace,
    }
    path = RESULTS_DIR / f"evolve_{stamp}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nResults saved to {path}")
    plot(evaluator, nsga_trace, random_trace, nsga_hv, random_hv, RESULTS_DIR / f"evolve_{stamp}.png")


if __name__ == "__main__":
    main()

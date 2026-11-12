"""Free checks on the Stage 1 result before paying for Stage 2 (no API calls).

  1. Which proxy predicts real correctness better? Compares article recall and evidence
     recall against the real correctness of the landscape configs (already paid for).
  2. Does the NSGA-II front hold up under evidence recall, or did it exploit article recall?
  3. Hypervolume with a reference point at a minimum useful quality, so pipelines too weak
     to use no longer count, for both article recall and evidence recall.

  uv run python -m evo_rag.check_proxy --ref-recall 0.75 --ref-evidence 0.6
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import HV

from evo_rag.config import PipelineConfig
from evo_rag.landscape import spearman
from evo_rag.proxy import ProxyEvaluator
from evo_rag.search_space import config_key

RESULTS_DIR = Path("data/results")
MAX_LOG_TOKENS = 4.0  # 10k tokens/query, as in evolve.py


def latest(pattern: str) -> dict:
    files = glob.glob(str(RESULTS_DIR / pattern))
    if not files:
        raise SystemExit(f"No {pattern} in {RESULTS_DIR}.")
    return json.loads(Path(max(files, key=os.path.getmtime)).read_text())


def hypervolume(rows: list[dict], metric: str, ref: float) -> float:
    """HV of (metric, log10 tokens), counting only pipelines better than the reference."""
    points = np.array([[-r[metric], np.log10(r["tokens"])] for r in rows])
    return float(HV(ref_point=np.array([-ref, MAX_LOG_TOKENS]))(points)) if len(points) else 0.0


def curve(rows: list[dict], metric: str, ref: float) -> list[float]:
    return [hypervolume(rows[: i + 1], metric, ref) for i in range(len(rows))]


def main(**evaluator_kwargs) -> None:
    parser = argparse.ArgumentParser(description="Free checks on the proxy and the Stage 1 front.")
    parser.add_argument("--ref-recall", type=float, default=0.75, help="minimum useful article recall")
    parser.add_argument("--ref-evidence", type=float, default=0.6, help="minimum useful evidence recall")
    args = parser.parse_args()
    proxy = ProxyEvaluator(**evaluator_kwargs)

    # 1. Which proxy predicts real correctness?
    landscape = latest("landscape_*.json")
    rows = []
    for r in landscape:
        config = PipelineConfig(**r["config"])
        p = proxy(config)
        rows.append((r["label"], p["recall"], p["evidence_recall"], r["correctness"]))
    print(f"1. Proxy vs real correctness ({len(rows)} landscape configs, first 50 dev questions)\n")
    print(f"   {'config':8} {'recall':>7} {'evidence':>9} {'real corr':>10}")
    for label, rec, ev, corr in sorted(rows, key=lambda x: -x[3]):
        print(f"   {label:8} {rec:7.3f} {ev:9.3f} {corr:10.3f}")
    rec, ev, corr = (list(x) for x in zip(*[r[1:] for r in rows]))
    print(f"\n   rank correlation with real correctness:  recall {spearman(rec, corr):.2f}   "
          f"evidence recall {spearman(ev, corr):.2f}")
    print(f"   linear correlation:                      recall {np.corrcoef(rec, corr)[0, 1]:.2f}   "
          f"evidence recall {np.corrcoef(ev, corr)[0, 1]:.2f}")

    # 2. The NSGA-II front under evidence recall.
    run = latest("evolve_*.json")
    print(f"\n2. NSGA-II proxy front, re-scored with evidence recall\n")
    print(f"   {'recall':>7} {'evidence':>9} {'gap':>6} {'tokens':>7}  pipeline")
    for r in run["nsga2_front"]:
        p = proxy(PipelineConfig(**r["config"]))
        c = r["config"]
        print(f"   {p['recall']:7.3f} {p['evidence_recall']:9.3f} {p['recall'] - p['evidence_recall']:6.3f} "
              f"{p['tokens']:7.0f}  {c['retrieval']} {c['embedder']} | {c['chunker']} {c['chunk_size']} "
              f"{'+hdr' if c['chunk_header'] else ''} | k={c['top_k']}")

    # Gap by chunk size over every pipeline evaluated (both methods).
    keys = list(dict.fromkeys(run["nsga2_trace"] + run["random_trace"]))
    scored = {k: proxy(PipelineConfig(**json.loads(k))) for k in keys}
    gaps = defaultdict(list)
    for p in scored.values():
        gaps[p["config"]["chunk_size"]].append(p["recall"] - p["evidence_recall"])
    print(f"\n   Mean gap (recall - evidence recall) by chunk size, over all {len(keys)} pipelines:")
    for size in sorted(gaps):
        print(f"     {size:5d} chars: {np.mean(gaps[size]):.3f}  ({len(gaps[size])} pipelines)")

    # 3. Hypervolume with meaningful reference points.
    nsga = [scored[k] for k in run["nsga2_trace"]]
    rand = [scored[k] for k in run["random_trace"]]
    half = len(nsga) // 2
    print("\n3. Hypervolume, counting only useful pipelines")
    for metric, ref in (("recall", args.ref_recall), ("evidence_recall", args.ref_evidence)):
        n_curve, r_curve = curve(nsga, metric, ref), curve(rand, metric, ref)
        print(f"   {metric:16} (ref {ref}):  after {half}: NSGA-II {n_curve[half - 1]:.4f}  random {r_curve[half - 1]:.4f}"
              f"   |  after {len(nsga)}: NSGA-II {n_curve[-1]:.4f}  random {r_curve[-1]:.4f}")
    print("\n   Note: NSGA-II optimised article recall, not evidence recall. Its result in the "
          "evidence row shows how much of its gain was real.")


if __name__ == "__main__":
    main()

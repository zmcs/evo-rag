"""Week 4, step 3: do the adapted embedders retrieve better on the real dev questions? (free)

Evidence recall on the dev QA set, for every embedder trained by adapt.py, in two settings:
  dense only     dense retrieval, recursive 600 + header, k=5: isolates the embedder
  best pipeline  hybrid a=0.37, recursive 1600 + header, k=13, query planning:
                 does the gain survive in the full system?

  uv run python -m evo_rag.adapt_check
"""

import json
import time
from pathlib import Path

import numpy as np

from evo_rag.config import PipelineConfig
from evo_rag.llm import LLM
from evo_rag.proxy import ProxyEvaluator
from evo_rag.retrieval import ADAPTED_EMBEDDERS, MODELS_DIR

RESULTS_DIR = Path("data/results")
TYPES = ["single_hop", "multi_hop", "numerical"]
SETTINGS = {
    "dense only": PipelineConfig(retrieval="dense", chunker="recursive", chunk_size=600, chunk_overlap=0.0,
                                 chunk_header=True, top_k=5),
    "best pipeline": PipelineConfig(retrieval="hybrid", hybrid_alpha=0.37, chunker="recursive", chunk_size=1600,
                                    chunk_overlap=0.0, chunk_header=True, top_k=13, query_mode="decompose"),
}


def main(**evaluator_kwargs) -> None:
    embedders = ["bge-small"] + [n for n in ADAPTED_EMBEDDERS if (MODELS_DIR / n).exists()]
    if len(embedders) == 1:
        raise SystemExit("No adapted embedders found: run `python -m evo_rag.adapt` first.")
    proxy = ProxyEvaluator(llm=LLM(), **evaluator_kwargs)  # query plans come from the cache
    results = []
    for setting, base in SETTINGS.items():
        print(f"\n{setting}: {base.retrieval}, {base.chunker} {base.chunk_size}, k={base.top_k}"
              f"{', planning' if base.query_mode == 'decompose' else ''}  (evidence recall on dev)")
        print(f"  {'embedder':22} {'all':>6} {'single':>7} {'multi':>6} {'num':>6}")
        for name in embedders:
            rows, _ = proxy.per_question(base.model_copy(update={"embedder": name}))
            by_type = {t: float(np.mean([r["evidence_recall"] for r in rows if r["type"] == t])) for t in TYPES}
            overall = float(np.mean([r["evidence_recall"] for r in rows]))
            print(f"  {name:22} {overall:6.3f} {by_type['single_hop']:7.3f} {by_type['multi_hop']:6.3f} "
                  f"{by_type['numerical']:6.3f}", flush=True)
            results.append({"setting": setting, "embedder": name, "evidence_recall": overall, **by_type})

    stamp = time.strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"adapt_check_{stamp}.json").write_text(json.dumps(results, indent=1))
    print("\nOn 158 dev questions, differences under ~0.02 are a few questions: treat them as noise.")


if __name__ == "__main__":
    main()

"""The search space: genes, their ranges, and the mapping to a PipelineConfig.

Shared by NSGA-II and random search, so both explore exactly the same space.

Design notes
  - chunk_size and chunk_overlap sit on coarse grids. Every new chunking setting means
    re-embedding the whole corpus, so a continuous overlap would make almost every
    individual require its own index.
  - Inactive genes are canonicalised: hybrid_alpha only matters for hybrid retrieval, and
    the embedder is unused by plain BM25. Two genomes that build the same pipeline map to
    the same canonical config, so they count as duplicates and share one evaluation.
  - prompt_template is fixed during the proxy search (retrieval metrics cannot see it).
  - query_mode chooses whether an LLM plans sub-queries before retrieval. Plans are cached,
    so the proxy search stays free; the planning cost is part of the token objective.
"""

import json
import random

from pymoo.core.variable import Binary, Choice, Integer, Real

from evo_rag.config import PipelineConfig

CHUNK_SIZES = list(range(200, 2001, 200))  # 200, 400, ..., 2000
OVERLAPS = [0.0, 0.1, 0.2, 0.3]
TOP_K_MAX = 20

VARIABLES = {
    "chunker": Choice(options=["fixed", "recursive", "semantic"]),
    "chunk_size_idx": Integer(bounds=(0, len(CHUNK_SIZES) - 1)),
    "overlap_idx": Integer(bounds=(0, len(OVERLAPS) - 1)),
    "chunk_header": Binary(),
    "embedder": Choice(options=["bge-small", "bge-base", "e5-base"]),
    "retrieval": Choice(options=["dense", "bm25", "hybrid"]),
    "hybrid_alpha": Real(bounds=(0.0, 1.0)),
    "top_k": Integer(bounds=(1, TOP_K_MAX)),
    "query_mode": Choice(options=["single", "decompose"]),
}


def to_config(x: dict, prompt_template: str = "default") -> PipelineConfig:
    """Genome (pymoo's dict of variable values) -> canonical PipelineConfig."""
    retrieval = x["retrieval"]
    chunker = x["chunker"]
    uses_embedder = retrieval != "bm25" or chunker == "semantic"
    return PipelineConfig(
        chunker=chunker,
        chunk_size=CHUNK_SIZES[int(x["chunk_size_idx"])],
        chunk_overlap=OVERLAPS[int(x["overlap_idx"])],
        chunk_header=bool(x["chunk_header"]),
        embedder=x["embedder"] if uses_embedder else "bge-small",
        retrieval=retrieval,
        hybrid_alpha=round(float(x["hybrid_alpha"]), 2) if retrieval == "hybrid" else 0.5,
        top_k=int(x["top_k"]),
        query_mode=x.get("query_mode", "single"),
        prompt_template=prompt_template,
    )


def config_key(config: PipelineConfig) -> str:
    """Identity of a pipeline: equal keys mean identical pipelines.

    query_mode is left out when it has its original value ("single"), so pipelines scored
    before that gene existed keep their keys and their cached results stay valid.
    """
    data = config.model_dump()
    if data.get("query_mode") == "single":
        data.pop("query_mode")
    return json.dumps(data, separators=(",", ":"))


def random_genome(rng: random.Random) -> dict:
    """Uniform sample from VARIABLES (the random-search baseline uses this)."""
    return {
        "chunker": rng.choice(VARIABLES["chunker"].options),
        "chunk_size_idx": rng.randrange(len(CHUNK_SIZES)),
        "overlap_idx": rng.randrange(len(OVERLAPS)),
        "chunk_header": rng.random() < 0.5,
        "embedder": rng.choice(VARIABLES["embedder"].options),
        "retrieval": rng.choice(VARIABLES["retrieval"].options),
        "hybrid_alpha": rng.random(),
        "top_k": rng.randint(1, TOP_K_MAX),
        "query_mode": rng.choice(VARIABLES["query_mode"].options),
    }

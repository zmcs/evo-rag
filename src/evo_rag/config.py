from pydantic import BaseModel
from typing import Literal

class PipelineConfig(BaseModel):
    chunk_size: int = 512            # e.g. 128–2048
    chunk_overlap: float = 0.1       # fraction of chunk_size
    chunker: Literal["fixed", "recursive", "semantic"] = "recursive"
    embedder: Literal["bge-small", "bge-base", "e5-base"] = "bge-small"
    retrieval: Literal["dense", "bm25", "hybrid"] = "dense"
    hybrid_alpha: float = 0.5        # only used if retrieval == "hybrid"
    top_k: int = 5
    reranker: bool = False
    generator: str = "small-cheap-model"
    prompt_template: str = "default" # later: evolved by the LLM mutator
from typing import Literal

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """One complete RAG pipeline, described as a set of genes."""

    # --- Chunking: how documents are split ---
    chunker: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = Field(800, ge=200, le=2000)  # characters (~50-500 tokens)
    chunk_overlap: float = Field(0.1, ge=0.0, le=0.5)  # fraction of chunk_size
    chunk_header: bool = True  # prefix each chunk with "Article 5: <title>"

    # --- Retrieval: how relevant chunks are found ---
    embedder: Literal["bge-small", "bge-base", "e5-base"] = "bge-small"
    retrieval: Literal["dense", "bm25", "hybrid"] = "dense"
    hybrid_alpha: float = Field(0.5, ge=0.0, le=1.0)  # only used if retrieval == "hybrid"
    top_k: int = Field(5, ge=1, le=20)
    reranker: bool = False
    # "decompose": an LLM splits the question into search queries (see planner.py)
    query_mode: Literal["single", "decompose"] = "single"

    # --- Generation: how the answer is written ---
    generator: Literal["gemini-3.5-flash-lite"] = "gemini-3.5-flash-lite"
    prompt_template: str = "default"  # a key of PROMPT_TEMPLATES; later evolved by the LLM mutator

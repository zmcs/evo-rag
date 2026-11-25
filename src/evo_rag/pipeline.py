"""A complete RAG pipeline: retrieve chunks, then generate an answer.

Try it:  uv run python -m evo_rag.pipeline "What does Article 5 prohibit?"
"""

import sys
from dataclasses import dataclass

from evo_rag.config import PipelineConfig
from evo_rag.generation import generate_answer
from evo_rag.llm import LLM, LLMResponse
from evo_rag.planner import retrieve
from evo_rag.retrieval import Retriever, build_retriever


@dataclass
class PipelineResult:
    question: str
    answer: str
    doc_ids: list[str]  # source article of each retrieved chunk, in rank order
    chunks: list[str]
    response: LLMResponse


class Pipeline:
    def __init__(self, config: PipelineConfig, retriever: Retriever, llm: LLM):
        self.config, self.retriever, self.llm = config, retriever, llm

    def ask(self, question: str) -> PipelineResult:
        chunks = retrieve(question, self.retriever, self.config, self.llm).chunks
        response = generate_answer(question, chunks, self.config, self.llm)
        return PipelineResult(
            question=question,
            answer=response.text,
            doc_ids=[c.doc_id for c in chunks],
            chunks=[c.text for c in chunks],
            response=response,
        )


def build_pipeline(config: PipelineConfig, llm: LLM | None = None, **retriever_kwargs) -> Pipeline:
    return Pipeline(config, build_retriever(config, **retriever_kwargs), llm or LLM())


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What is the maximum fine for using a prohibited AI practice?"
    pipeline = build_pipeline(PipelineConfig(retrieval="hybrid"))
    result = pipeline.ask(question)
    r = result.response
    print(f"Q: {question}\n\nA: {result.answer}\n")
    print("Retrieved from:", result.doc_ids)
    print(
        f"Tokens: {r.input_tokens} in, {r.output_tokens} out, {r.thinking_tokens} thinking | "
        f"cost ${r.cost_usd:.5f} | {r.latency_s:.2f}s{' (cached)' if r.cached else ''}"
    )
    print(f"Total spend so far: ${pipeline.llm.total_spend():.4f}")


if __name__ == "__main__":
    main()

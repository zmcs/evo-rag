"""Turn a question plus retrieved chunks into an answer.

Prompt templates are looked up by name (the config's `prompt_template` gene).
In Week 3 the LLM mutator will write new templates and add them here.
"""

from evo_rag.chunking import Chunk
from evo_rag.config import PipelineConfig
from evo_rag.llm import LLM, LLMResponse

# Fixed phrase for abstaining, so unanswerable questions can be scored exactly.
ABSTAIN = "The provided context does not contain the answer."

PROMPT_TEMPLATES = {
    "default": {
        "system": (
            "You answer questions about the EU AI Act using only the context provided. "
            f'If the context does not contain the answer, reply exactly: "{ABSTAIN}"'
        ),
        "user": "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    },
    "concise_cited": {
        "system": (
            "You are a precise legal assistant for the EU AI Act. Answer only from the context. "
            "Be concise: one to three sentences. Cite the article or annex you used in brackets, "
            "e.g. [Article 5]. Do not use outside knowledge. "
            f'If the context does not contain the answer, reply exactly: "{ABSTAIN}"'
        ),
        "user": "Context:\n{context}\n\nQuestion: {question}",
    },
}


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] {chunk.text}" for i, chunk in enumerate(chunks, 1))


def generate_answer(question: str, chunks: list[Chunk], config: PipelineConfig, llm: LLM) -> LLMResponse:
    template = PROMPT_TEMPLATES[config.prompt_template]
    prompt = template["user"].format(context=format_context(chunks), question=question)
    return llm.generate(model=config.generator, prompt=prompt, system=template["system"])

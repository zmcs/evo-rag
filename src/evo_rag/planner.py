"""Query planning: split a question into search queries, retrieve for each, merge.

query_mode = "single"     retrieve with the question itself (the original pipeline)
query_mode = "decompose"  one cheap LLM call writes 1-3 search queries; each query gets
                          its own ranked list, and the lists are merged in turns (the
                          original question first) until top_k chunks are collected.

The context budget (top_k) is the same in both modes, so a gain from "decompose" comes
from choosing better chunks, not from sending more text. Plans are cached by the LLM
wrapper, so each question is planned once.
"""

import json
from dataclasses import dataclass

from evo_rag.chunking import Chunk
from evo_rag.config import PipelineConfig
from evo_rag.llm import LLM, PRICES_USD_PER_M

PLANNER_MODEL = "gemini-3.5-flash-lite"
MAX_QUERIES = 3

PLANNER_SYSTEM = "You plan document searches for questions about the EU AI Act (Regulation (EU) 2024/1689)."

PLANNER_PROMPT = """Question:
{question}

Write between 1 and {max_queries} short search queries that together would retrieve every
provision needed to answer the question. Legal questions often need more than one provision,
for example an obligation and the penalty for breaching it, a definition and the rule that
uses it, or a rule and the date from which it applies. Use a single query if one provision
is enough. Do not answer the question.

Reply with JSON only: {{"queries": ["...", "..."]}}"""


@dataclass
class Retrieval:
    chunks: list[Chunk]
    queries: list[str]
    plan_cost_usd: float = 0.0
    plan_latency_s: float = 0.0
    plan_cost_tokens: float = 0.0  # planning cost expressed in generator input-token equivalents


def plan_queries(question: str, llm: LLM, cache_salt: int = 0) -> tuple[list[str], object]:
    """Return [question] + up to MAX_QUERIES planned queries, and the LLM response."""
    response = llm.generate(
        model=PLANNER_MODEL,
        prompt=PLANNER_PROMPT.format(question=question, max_queries=MAX_QUERIES),
        system=PLANNER_SYSTEM,
        json_mode=True,
        max_output_tokens=200,
        cache_salt=cache_salt,
    )
    try:
        planned = [q.strip() for q in json.loads(response.text)["queries"] if isinstance(q, str) and q.strip()]
    except (json.JSONDecodeError, KeyError, TypeError):
        planned = []  # unparseable plan: fall back to the question alone
    queries = list(dict.fromkeys([question] + planned[:MAX_QUERIES]))
    return queries, response


def merge_round_robin(ranked_lists: list[list[int]], k: int) -> list[int]:
    """Take the best not-yet-chosen item from each list in turn until k items are chosen."""
    chosen, seen, rank = [], set(), 0
    longest = max((len(r) for r in ranked_lists), default=0)
    while len(chosen) < k and rank < longest:
        for ranked in ranked_lists:
            if rank < len(ranked) and ranked[rank] not in seen and len(chosen) < k:
                seen.add(ranked[rank])
                chosen.append(ranked[rank])
        rank += 1
    return chosen


def retrieve(question: str, retriever, config: PipelineConfig, llm: LLM | None = None, cache_salt: int = 0) -> Retrieval:
    if config.query_mode == "single":
        return Retrieval([c for c, _ in retriever.search(question)], [question])

    queries, response = plan_queries(question, llm, cache_salt)
    ranked = [list(retriever.ranked_indices(q, config.top_k)) for q in queries]
    chunks = [retriever.chunks[i] for i in merge_round_robin(ranked, config.top_k)]
    price_in, price_out = PRICES_USD_PER_M[config.generator]
    cost_tokens = response.input_tokens + (response.output_tokens + response.thinking_tokens) * price_out / price_in
    return Retrieval(chunks, queries, response.cost_usd, response.latency_s, cost_tokens)

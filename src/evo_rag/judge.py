"""Score a generated answer.

Two scores per answer:
  correctness  1 / 0.5 / 0 (correct / partial / incorrect), compared with the gold answer
  faithful     1 if every claim in the answer is supported by the retrieved context

Abstentions are scored by rule, without calling the judge:
  - abstaining on an unanswerable question: correct and faithful
  - abstaining on an answerable question: incorrect, but faithful (it invented nothing)

An answerable question whose answer gives real content AND the abstention sentence (it
answered one part and said another part is missing) is not a pure abstention: it goes to
the judge, which can give partial credit. On an unanswerable question, any answer that
contains the abstention sentence counts as declining.
"""

import json
import re
from dataclasses import dataclass

from evo_rag.generation import ABSTAIN, format_context
from evo_rag.llm import LLM

JUDGE_MODEL = "gemini-3.5-flash-lite"  # fixed: part of the measurement, not a gene

CORRECTNESS_SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

JUDGE_SYSTEM = "You are a strict, careful evaluator of answers to questions about the EU AI Act."

JUDGE_PROMPT = """Question:
{question}

Reference answer (written and verified by an expert):
{gold}

Context the assistant was given:
{context}

Assistant's answer:
{answer}

Evaluate the assistant's answer on two criteria.

1. correctness, compared with the reference answer:
   - "correct": contains the key facts of the reference answer and contradicts nothing in it.
     Extra details are fine as long as they do not contradict the reference.
   - "partial": gets some key facts right but misses or misstates others.
   - "incorrect": wrong, contradicts the reference, or does not answer the question.

2. faithful: true if every factual claim in the assistant's answer is supported by the
   context above; false if any claim is missing from the context, even if it is true.

Reply with JSON only:
{{"reason": "<one sentence>", "correctness": "correct" | "partial" | "incorrect", "faithful": true | false}}"""


@dataclass
class Judgement:
    correctness: float
    faithful: float
    abstained: bool
    reason: str
    cost_usd: float = 0.0
    judge_error: bool = False


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def is_abstention(answer: str) -> bool:
    """The answer contains the abstention sentence (possibly with other text)."""
    return normalise(ABSTAIN) in normalise(answer)


def is_pure_abstention(answer: str, max_extra_words: int = 6) -> bool:
    """The answer is the abstention sentence and little else."""
    if not is_abstention(answer):
        return False
    remaining = normalise(answer).replace(normalise(ABSTAIN), " ").split()
    return len(remaining) <= max_extra_words


def judge_answer(
    question: dict, answer: str, chunks: list, llm: LLM, cache_salt: int = 0
) -> Judgement:
    """`question` is a QA record; `chunks` are the Chunk objects the generator saw."""
    unanswerable = question["type"] == "unanswerable"

    if unanswerable and is_abstention(answer):
        return Judgement(correctness=1.0, faithful=1.0, abstained=True, reason="Correct abstention.")
    if not unanswerable and is_pure_abstention(answer):
        return Judgement(correctness=0.0, faithful=1.0, abstained=True, reason="Abstained on an answerable question.")

    prompt = JUDGE_PROMPT.format(
        question=question["question"],
        gold=question["answer"],
        context=format_context(chunks),
        answer=answer,
    )
    response = llm.generate(
        model=JUDGE_MODEL, prompt=prompt, system=JUDGE_SYSTEM, json_mode=True, cache_salt=cache_salt
    )
    try:
        verdict = json.loads(response.text)
        correctness = CORRECTNESS_SCORES[verdict["correctness"]]
        faithful = float(bool(verdict["faithful"]))
        reason, error = str(verdict.get("reason", "")), False
    except (json.JSONDecodeError, KeyError, TypeError):
        correctness, faithful, reason, error = 0.0, 0.0, f"Unparseable judge output: {response.text[:200]}", True

    if unanswerable:  # it should have abstained, whatever the judge thinks of the content
        correctness, reason = 0.0, "Answered an unanswerable question. " + reason

    return Judgement(correctness, faithful, False, reason, response.cost_usd, error)

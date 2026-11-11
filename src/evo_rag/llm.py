"""Thin wrapper around the Gemini API: caching, token/cost accounting and a hard budget.

Every call is logged to a local SQLite file (data/cache/llm.sqlite):
  - identical requests are answered from the cache, for free
  - total spend is summed over all calls, and the wrapper refuses to call the API
    once it reaches EVO_RAG_BUDGET_USD (set in .env)
"""

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = Path("data/cache/llm.sqlite")
DEFAULT_BUDGET_USD = 4.0  # leaves a margin under a 5 EUR top-up

# USD per 1M tokens (input, output). Output includes thinking tokens.
# Check https://ai.google.dev/gemini-api/docs/pricing and update when prices change.
PRICES_USD_PER_M = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
}

RETRYABLE_CODES = {429, 500, 503}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int  # visible answer tokens
    thinking_tokens: int  # hidden reasoning tokens (billed as output)
    cost_usd: float
    latency_s: float  # of the original API call, even when served from cache
    cached: bool = False


def cost_usd(model: str, input_tokens: int, output_tokens: int, thinking_tokens: int) -> float:
    price_in, price_out = PRICES_USD_PER_M[model]
    return (input_tokens * price_in + (output_tokens + thinking_tokens) * price_out) / 1_000_000


class LLM:
    def __init__(self, cache_path: Path = CACHE_PATH, budget_usd: float | None = None, client=None):
        self.budget_usd = budget_usd if budget_usd is not None else float(
            os.getenv("EVO_RAG_BUDGET_USD", DEFAULT_BUDGET_USD)
        )
        self._client = client
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS calls (ts REAL, model TEXT, input_tokens INT,"
            " output_tokens INT, thinking_tokens INT, cost_usd REAL)"
        )

    @property
    def client(self):
        if self._client is None:
            from google import genai  # reads GEMINI_API_KEY from the environment

            self._client = genai.Client()
        return self._client

    def total_spend(self) -> float:
        return self.db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM calls").fetchone()[0]

    def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        thinking_level: str = "minimal",
        json_mode: bool = False,
        cache_salt: int = 0,
    ) -> LLMResponse:
        """Call the model, or return the cached response for an identical request.

        json_mode   ask the model to reply with JSON only
        cache_salt  a different value forces a fresh API call (used to measure noise)
        """
        settings = dict(
            model=model, prompt=prompt, system=system, temperature=temperature,
            max_output_tokens=max_output_tokens, thinking_level=thinking_level, json_mode=json_mode,
        )
        key_material = settings | ({"salt": cache_salt} if cache_salt else {})
        key = hashlib.sha256(json.dumps(key_material, sort_keys=True).encode()).hexdigest()
        row = self.db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            return LLMResponse(**(json.loads(row[0]) | {"cached": True}))

        spent = self.total_spend()
        if spent >= self.budget_usd:
            raise BudgetExceeded(
                f"Spent ${spent:.4f} of the ${self.budget_usd:.2f} budget. "
                "Raise EVO_RAG_BUDGET_USD in .env to continue."
            )

        response = self._call_with_retries(**settings)
        self.db.execute("INSERT INTO cache VALUES (?, ?)", (key, json.dumps(asdict(response))))
        self.db.execute(
            "INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), model, response.input_tokens, response.output_tokens,
             response.thinking_tokens, response.cost_usd),
        )
        self.db.commit()
        return response

    def _call_with_retries(
        self, model, prompt, system, temperature, max_output_tokens, thinking_level, json_mode, attempts=5
    ):
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
            response_mime_type="application/json" if json_mode else None,
        )
        for attempt in range(attempts):
            try:
                start = time.perf_counter()
                result = self.client.models.generate_content(model=model, contents=prompt, config=config)
                latency = time.perf_counter() - start
                break
            except errors.APIError as e:
                if e.code not in RETRYABLE_CODES or attempt == attempts - 1:
                    raise
                time.sleep(2**attempt)  # 1, 2, 4, 8 s

        usage = result.usage_metadata
        n_in = usage.prompt_token_count or 0
        n_out = usage.candidates_token_count or 0
        n_think = usage.thoughts_token_count or 0
        return LLMResponse(
            text=(result.text or "").strip(),
            input_tokens=n_in,
            output_tokens=n_out,
            thinking_tokens=n_think,
            cost_usd=cost_usd(model, n_in, n_out, n_think),
            latency_s=latency,
        )

"""Client wrapping a backend with policy, caching, retries, concurrency limits and a cost ledger."""

from __future__ import annotations

import asyncio
import json
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scorecard.io import sha256_text, stable_json
from scorecard.llm.types import (
    Backend,
    LLMRequest,
    LLMResponse,
    Outcome,
    RetryableError,
    enforce_policy,
)


@dataclass(frozen=True)
class Price:
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


class ResponseCache:
    """Content-addressed store, so re-running a pipeline never pays twice."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, body TEXT)")

    def get(self, key: str) -> LLMResponse | None:
        row = self._db.execute("SELECT body FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        body = json.loads(row[0])
        return LLMResponse(
            body["text"],
            Outcome(body["outcome"]),
            body["input_tokens"],
            body["output_tokens"],
            cached=True,
            detail=body.get("detail", ""),
        )

    def put(self, key: str, response: LLMResponse) -> None:
        body = {
            "text": response.text,
            "outcome": response.outcome.value,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "detail": response.detail,
        }
        self._db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, json.dumps(body)))
        self._db.commit()

    def close(self) -> None:
        self._db.close()


class CostLedger:
    """Append-only record of every call: tokens, price, prompt hash and outcome."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def total_usd(self) -> float:
        if not self.path.exists():
            return 0.0
        with self.path.open(encoding="utf-8") as handle:
            return float(sum(json.loads(line)["cost_usd"] for line in handle if line.strip()))


def cache_key(request: LLMRequest) -> str:
    return sha256_text(
        stable_json(
            {
                "model": request.model,
                "messages": request.messages,
                "schema": request.json_schema,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "salt": request.salt,
            }
        )
    )


class LLMClient:
    def __init__(
        self,
        backend: Backend,
        cache: ResponseCache | None = None,
        ledger: CostLedger | None = None,
        price: Price | None = None,
        concurrency: int = 8,
        max_attempts: int = 5,
        base_delay: float = 1.0,
    ) -> None:
        self._backend = backend
        self._cache = cache
        self._ledger = ledger
        self._price = price or Price()
        self._concurrency = concurrency
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._gate: asyncio.Semaphore | None = None
        self._gate_loop: asyncio.AbstractEventLoop | None = None

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._gate is None or self._gate_loop is not loop:
            self._gate, self._gate_loop = asyncio.Semaphore(self._concurrency), loop
        return self._gate

    async def complete(self, request: LLMRequest) -> LLMResponse:
        enforce_policy(self._backend.provider_class, request.purpose)
        key = cache_key(request)
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                self._record(request, key, hit, 0.0)
                return hit
        async with self._semaphore():
            start = time.perf_counter()
            response = await self._call_with_retries(request)
            elapsed = time.perf_counter() - start
        # Errors stay out of the cache so a re-run can recover. Refusals stay in: they cost money.
        if self._cache is not None and response.outcome is not Outcome.ERROR:
            self._cache.put(key, response)
        self._record(request, key, response, elapsed)
        return response

    async def _call_with_retries(self, request: LLMRequest) -> LLMResponse:
        failure = ""
        for attempt in range(self._max_attempts):
            try:
                return await self._backend.complete(request)
            except RetryableError as exc:
                failure = str(exc)
                if attempt + 1 < self._max_attempts:
                    delay = self._base_delay * 2**attempt
                    await asyncio.sleep(delay * (0.5 + random.random() / 2))
        return LLMResponse("", Outcome.ERROR, detail=f"retries exhausted: {failure}")

    def _record(self, request: LLMRequest, key: str, res: LLMResponse, elapsed: float) -> None:
        if self._ledger is None:
            return
        cost = 0.0 if res.cached else self._price.cost(res.input_tokens, res.output_tokens)
        self._ledger.record(
            {
                "ts": time.time(),
                "model": request.model,
                "purpose": request.purpose.value,
                "prompt_hash": key,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "cost_usd": cost,
                "cached": res.cached,
                "outcome": res.outcome.value,
                "elapsed_s": round(elapsed, 3),
            }
        )

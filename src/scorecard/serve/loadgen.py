"""Open-loop load generator. Arrivals don't wait for responses, avoiding coordinated omission."""

from __future__ import annotations

import asyncio
import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import httpx
import numpy as np


@dataclass(frozen=True)
class LoadResult:
    target_rps: float
    duration_s: float
    sent: int
    completed_ok: int
    statuses: dict[str, int]
    achieved_rps: float
    calls_per_hour: float
    p50_s: float
    p95_s: float
    p99_s: float
    error_rate: float

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def summarize(
    target_rps: float, wall: float, sent: int, statuses: Counter[str], latencies: Sequence[float]
) -> LoadResult:
    ok = statuses.get("200", 0)
    percentiles = (
        np.percentile(latencies, [50, 95, 99]).tolist() if latencies else [float("nan")] * 3
    )
    return LoadResult(
        target_rps=target_rps,
        duration_s=wall,
        sent=sent,
        completed_ok=ok,
        statuses=dict(statuses),
        achieved_rps=ok / wall if wall else 0.0,
        calls_per_hour=ok / wall * 3600 if wall else 0.0,
        p50_s=percentiles[0],
        p95_s=percentiles[1],
        p99_s=percentiles[2],
        error_rate=(sent - ok) / sent if sent else 0.0,
    )


async def run_load(
    client: httpx.AsyncClient,
    bodies: Sequence[dict[str, Any]],
    target_rps: float,
    duration_s: float,
    api_key: str,
    seed: int = 0,
) -> LoadResult:
    """Poisson arrivals. Latency counts from the scheduled arrival, including client-side delay."""
    rng = random.Random(seed)
    statuses: Counter[str] = Counter()
    latencies: list[float] = []
    tasks: list[asyncio.Task[None]] = []
    start = time.perf_counter()

    async def one(body: dict[str, Any], scheduled: float) -> None:
        try:
            reply = await client.post("/v1/scorecards", json=body, headers={"x-api-key": api_key})
        except httpx.HTTPError as exc:
            statuses[type(exc).__name__] += 1
            return
        statuses[str(reply.status_code)] += 1
        if reply.status_code == 200:
            latencies.append(time.perf_counter() - scheduled)

    scheduled = start
    index = 0
    while True:
        scheduled += rng.expovariate(target_rps)
        if scheduled - start >= duration_s:
            break
        delay = scheduled - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(bodies[index % len(bodies)], scheduled)))
        index += 1
    await asyncio.gather(*tasks)
    return summarize(target_rps, time.perf_counter() - start, len(tasks), statuses, latencies)


async def sweep_load(
    client: httpx.AsyncClient,
    bodies: Sequence[dict[str, Any]],
    rates: Sequence[float],
    duration_s: float,
    api_key: str,
) -> list[LoadResult]:
    return [await run_load(client, bodies, rate, duration_s, api_key) for rate in rates]

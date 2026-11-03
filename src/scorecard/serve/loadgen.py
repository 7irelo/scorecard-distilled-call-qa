"""Open-loop load generator. Arrivals don't wait for responses, avoiding coordinated omission."""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
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


def saturation_point(
    results: Sequence[LoadResult], max_error_rate: float = 0.01, min_delivery: float = 0.9
) -> LoadResult | None:
    """The highest tested rate the server kept up with, which sets peak throughput for costs."""
    healthy = [
        result
        for result in results
        if result.error_rate <= max_error_rate
        and result.achieved_rps >= min_delivery * result.target_rps
    ]
    return max(healthy, key=lambda result: result.achieved_rps, default=None)


def body_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Build an API request from a dataset row, so load follows real transcript lengths."""
    transcript = row["transcript"]
    return {
        "call_id": row["scenario_id"],
        "turns": [
            {"id": turn["id"], "speaker": turn["speaker"], "text": turn["text"]}
            for turn in transcript["turns"]
        ],
    }


class GpuSampler:
    """Samples GPU use during a load test. Reports nothing on machines without nvidia-smi."""

    _QUERY = "--query-gpu=utilization.gpu,memory.used"

    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self.samples: list[tuple[float, float]] = []
        self._task: asyncio.Task[None] | None = None

    def _read(self) -> tuple[float, float] | None:
        if shutil.which("nvidia-smi") is None:
            return None
        try:
            output = subprocess.run(
                ["nvidia-smi", self._QUERY, "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout
            used, memory = output.splitlines()[0].split(",")
            return float(used), float(memory)
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    async def _run(self) -> None:
        while True:
            sample = await asyncio.to_thread(self._read)
            if sample is not None:
                self.samples.append(sample)
            await asyncio.sleep(self.interval_s)

    async def __aenter__(self) -> GpuSampler:
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def summary(self) -> dict[str, float] | None:
        if not self.samples:
            return None
        usage = [sample[0] for sample in self.samples]
        memory = [sample[1] for sample in self.samples]
        return {
            "gpu_util_mean": float(np.mean(usage)),
            "gpu_util_max": float(np.max(usage)),
            "gpu_memory_mb_max": float(np.max(memory)),
        }


async def probe_ttft(
    client: httpx.AsyncClient,
    model: str,
    messages: Sequence[dict[str, str]],
    max_tokens: int = 1024,
) -> dict[str, float]:
    """Time to first token and total time, by streaming directly from vLLM."""
    body = {
        "model": model,
        "messages": list(messages),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    start = time.perf_counter()
    first: float | None = None
    async with client.stream("POST", "chat/completions", json=body) as reply:
        reply.raise_for_status()
        async for line in reply.aiter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            choices = json.loads(line[len("data: ") :]).get("choices") or [{}]
            if first is None and choices[0].get("delta", {}).get("content"):
                first = time.perf_counter() - start
    total = time.perf_counter() - start
    return {"ttft_s": first if first is not None else float("nan"), "total_s": total}


def ledger_stats(path: Path, model: str | None = None) -> dict[str, dict[str, float]]:
    """Measured tokens, latency and cost per model from the ledger. Cached calls don't count."""
    rows: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["cached"] or (model is not None and row["model"] != model):
                continue
            rows.setdefault(row["model"], []).append(row)
    stats: dict[str, dict[str, float]] = {}
    for name, entries in sorted(rows.items()):
        elapsed = [entry["elapsed_s"] for entry in entries]
        stats[name] = {
            "calls": float(len(entries)),
            "mean_input_tokens": float(np.mean([entry["input_tokens"] for entry in entries])),
            "mean_output_tokens": float(np.mean([entry["output_tokens"] for entry in entries])),
            "p50_s": float(np.percentile(elapsed, 50)),
            "p95_s": float(np.percentile(elapsed, 95)),
            "usd_per_1k": float(np.mean([entry["cost_usd"] for entry in entries])) * 1000,
            "failure_rate": float(np.mean([entry["outcome"] != "ok" for entry in entries])),
        }
    return stats


def load_markdown(results: Sequence[LoadResult], gpu: dict[str, float] | None = None) -> str:
    lines = [
        "| Target rps | Achieved rps | Calls/hour | p50 s | p95 s | p99 s | Errors |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.target_rps:g} | {r.achieved_rps:.2f} | {r.calls_per_hour:,.0f} | "
            f"{r.p50_s:.2f} | {r.p95_s:.2f} | {r.p99_s:.2f} | {r.error_rate:.1%} |"
        )
    peak = saturation_point(results)
    lines.append("")
    lines.append(
        f"Peak sustained: {peak.calls_per_hour:,.0f} calls/hour at {peak.target_rps:g} rps."
        if peak
        else "No tested rate was sustained, so test lower rates."
    )
    if gpu:
        lines.append(
            f"GPU utilisation {gpu['gpu_util_mean']:.0f}% mean, {gpu['gpu_util_max']:.0f}% max; "
            f"memory {gpu['gpu_memory_mb_max']:,.0f} MB max."
        )
    return "\n".join(lines) + "\n"

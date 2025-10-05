"""FastAPI service. Transcript text never reaches logs or metrics, only IDs, sizes and timings."""

import asyncio
import hmac
import logging
import os
import time
import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from scorecard.baselines.llm_system import parse_scorecard
from scorecard.data.models import Speaker, Transcript, Turn
from scorecard.schema import Scorecard, semantic_issues
from scorecard.serve.inference import BackendUnavailable, FakeBackend, InferenceBackend, VLLMBackend

log = logging.getLogger("scorecard.api")


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    api_key: str
    model_revision: str = "unset"
    adapter_hash: str = "unset"
    max_transcript_chars: int = 60_000
    max_concurrency: int = 16
    max_queue: int = 64
    request_timeout_s: float = 120.0


class TurnIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    speaker: Speaker
    text: str


class ScorecardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    call_id: str = Field(min_length=1, max_length=128)
    turns: list[TurnIn] = Field(min_length=1)


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_revision: str
    adapter_hash: str
    schema_version: str


class ScorecardResponse(BaseModel):
    call_id: str
    request_id: str
    scorecard: Scorecard
    model_info: ModelInfo
    semantic_warnings: list[str]
    attempts: int


class _Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "scorecard_requests_total", "Requests by outcome", ["outcome"], registry=self.registry
        )
        self.latency = Histogram(
            "scorecard_request_seconds",
            "End-to-end request latency",
            buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128),
            registry=self.registry,
        )
        self.in_flight = Gauge(
            "scorecard_in_flight", "Requests being processed or queued", registry=self.registry
        )
        self.tokens = Counter(
            "scorecard_tokens_total", "Tokens by direction", ["direction"], registry=self.registry
        )
        self.invalid = Counter(
            "scorecard_invalid_outputs_total", "Invalid model outputs", registry=self.registry
        )


def create_app(settings: Settings, backend: InferenceBackend) -> FastAPI:
    app = FastAPI(title="Scorecard", version="1.0")
    metrics = _Metrics()
    gate = asyncio.Semaphore(settings.max_concurrency)
    admitted = {"count": 0}
    limit = settings.max_concurrency + settings.max_queue
    info = ModelInfo(
        model_revision=settings.model_revision,
        adapter_hash=settings.adapter_hash,
        schema_version="1.0",
    )

    def authorize(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_key):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.middleware("http")
    async def reject_oversized_bodies(request: Request, call_next):  # type: ignore[no-untyped-def]
        length = request.headers.get("content-length")
        # Allows for JSON overhead here. The exact text limit is enforced after parsing.
        if length and length.isdigit() and int(length) > 2 * settings.max_transcript_chars + 65_536:
            metrics.requests.labels("too_large").inc()
            return JSONResponse({"detail": "transcript too large"}, status_code=413)
        return await call_next(request)

    async def generate_validated(
        transcript: Transcript, request_id: str
    ) -> tuple[Scorecard, list[str], int]:
        """One retry on invalid output, then an explicit error rather than a guessed card."""
        for attempt in (1, 2):
            generation = await backend.generate(transcript)
            metrics.tokens.labels("input").inc(generation.input_tokens)
            metrics.tokens.labels("output").inc(generation.output_tokens)
            card, _ = parse_scorecard(generation.text)
            if card is not None:
                return card, semantic_issues(card, transcript.turn_ids()), attempt
            metrics.invalid.inc()
            log.warning("invalid model output request_id=%s attempt=%d", request_id, attempt)
        raise HTTPException(status_code=502, detail="model returned invalid output twice")

    @app.post("/v1/scorecards", response_model=ScorecardResponse)
    async def create_scorecard(
        body: ScorecardRequest, _: Annotated[None, Depends(authorize)]
    ) -> ScorecardResponse:
        request_id = uuid.uuid4().hex
        chars = sum(len(turn.text) for turn in body.turns)
        if chars > settings.max_transcript_chars:
            # Rejected, not truncated: truncating would silently corrupt compliance results.
            metrics.requests.labels("too_large").inc()
            raise HTTPException(status_code=413, detail="transcript too large")
        if len({turn.id for turn in body.turns}) != len(body.turns):
            metrics.requests.labels("bad_request").inc()
            raise HTTPException(status_code=422, detail="turn ids must be unique")
        if admitted["count"] >= limit:
            metrics.requests.labels("overloaded").inc()
            raise HTTPException(status_code=429, detail="server busy", headers={"Retry-After": "5"})

        transcript = Transcript(
            scenario_id=body.call_id,
            turns=tuple(Turn(id=t.id, speaker=t.speaker, text=t.text) for t in body.turns),
        )
        start = time.perf_counter()
        admitted["count"] += 1
        metrics.in_flight.inc()
        try:
            async with asyncio.timeout(settings.request_timeout_s), gate:
                card, warnings, attempts = await generate_validated(transcript, request_id)
        except TimeoutError as exc:
            metrics.requests.labels("timeout").inc()
            raise HTTPException(status_code=504, detail="inference timed out") from exc
        except BackendUnavailable as exc:
            metrics.requests.labels("backend_error").inc()
            log.error("backend error request_id=%s type=%s", request_id, exc)
            raise HTTPException(status_code=503, detail="inference unavailable") from exc
        except HTTPException:
            metrics.requests.labels("invalid_output").inc()
            raise
        finally:
            admitted["count"] -= 1
            metrics.in_flight.dec()
        elapsed = time.perf_counter() - start
        metrics.latency.observe(elapsed)
        metrics.requests.labels("ok").inc()
        log.info(
            "scored request_id=%s turns=%d chars=%d seconds=%.3f attempts=%d",
            request_id,
            len(body.turns),
            chars,
            elapsed,
            attempts,
        )
        return ScorecardResponse(
            call_id=body.call_id,
            request_id=request_id,
            scorecard=card,
            model_info=info,
            semantic_warnings=warnings,
            attempts=attempts,
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> Response:
        ok = await backend.ready()
        return JSONResponse({"ready": ok}, status_code=200 if ok else 503)

    @app.get("/metrics")
    async def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def app_from_env() -> FastAPI:
    """Entry point for uvicorn: `uvicorn --factory scorecard.serve.app:app_from_env`."""
    logging.basicConfig(level=logging.INFO)
    settings = Settings(
        api_key=os.environ["SCORECARD_API_KEY"],
        model_revision=os.environ.get("SCORECARD_MODEL_REVISION", "unset"),
        adapter_hash=os.environ.get("SCORECARD_ADAPTER_HASH", "unset"),
        max_concurrency=int(os.environ.get("SCORECARD_MAX_CONCURRENCY", "16")),
    )
    backend: InferenceBackend
    if os.environ.get("SCORECARD_BACKEND", "vllm") == "fake":
        backend = FakeBackend()
    else:
        backend = VLLMBackend(
            os.environ["SCORECARD_VLLM_URL"],
            os.environ["SCORECARD_VLLM_MODEL"],
            os.environ.get("SCORECARD_VLLM_KEY", ""),
        )
    return create_app(settings, backend)

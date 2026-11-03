"""Inference backends for the API: vLLM in production and a fake for tests and CPU machines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from scorecard.baselines.rules import predict_rules
from scorecard.data.models import Transcript
from scorecard.prompts import small_model_messages
from scorecard.schema import scorecard_json_schema


class BackendUnavailable(RuntimeError):
    pass


class TranscriptTooLong(ValueError):
    """The call exceeds the model's context. It must be rejected, not truncated."""


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class InferenceBackend(Protocol):
    async def generate(self, transcript: Transcript) -> Generation: ...

    async def ready(self) -> bool: ...


def _is_context_error(text: str) -> bool:
    lowered = text.lower()
    return "context length" in lowered or "maximum context" in lowered


class VLLMBackend:
    """Calls vLLM's OpenAI endpoint with schema-constrained decoding."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 300.0,
        max_tokens: int = 1024,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=timeout,
            transport=transport,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._schema = scorecard_json_schema()

    async def generate(self, transcript: Transcript) -> Generation:
        body = {
            "model": self._model,
            "messages": [{"role": r, "content": c} for r, c in small_model_messages(transcript)],
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "scorecard", "schema": self._schema},
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            reply = await self._client.post("chat/completions", json=body)
            if reply.status_code == 400 and _is_context_error(reply.text):
                raise TranscriptTooLong("exceeds model context")
            reply.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendUnavailable(type(exc).__name__) from exc
        payload = reply.json()
        usage = payload.get("usage") or {}
        return Generation(
            payload["choices"][0]["message"].get("content") or "",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

    async def ready(self) -> bool:
        try:
            return (await self._client.get("models")).status_code == 200
        except httpx.HTTPError:
            return False


class FakeBackend:
    """Rule-based stand-in. Lets the API run in CI and on Windows, without a GPU."""

    def __init__(self, fail_first: int = 0, invalid_first: int = 0) -> None:
        self.fail_first = fail_first
        self.invalid_first = invalid_first
        self.calls = 0

    async def generate(self, transcript: Transcript) -> Generation:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise BackendUnavailable("simulated")
        if self.calls <= self.fail_first + self.invalid_first:
            return Generation("{not json")
        card = predict_rules(transcript)
        words = sum(len(t.text.split()) for t in transcript.turns)
        return Generation(card.canonical_json(), words, len(card.canonical_json().split()))

    async def ready(self) -> bool:
        return True

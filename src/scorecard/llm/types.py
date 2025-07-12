"""Provider-neutral request and response types and the training-data-path policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class Purpose(StrEnum):
    GENERATE = "generate"
    VERIFY = "verify"
    TEACHER = "teacher"
    JUDGE = "judge"
    BASELINE = "baseline"


class ProviderClass(StrEnum):
    OPEN_WEIGHT = "open_weight"
    CLOSED = "closed"


class Outcome(StrEnum):
    OK = "ok"
    REFUSED = "refused"
    TRUNCATED = "truncated"
    ERROR = "error"


class PolicyViolation(RuntimeError):
    """A closed provider was asked to do work outside baselines (docs/adr/0001-licensing.md)."""


class RetryableError(RuntimeError):
    """A rate limit, server fault or network problem worth retrying."""


def enforce_policy(provider_class: ProviderClass, purpose: Purpose) -> None:
    if provider_class is ProviderClass.CLOSED and purpose is not Purpose.BASELINE:
        raise PolicyViolation(
            f"closed providers may only run baselines on evaluation data, not {purpose.value!r}"
        )


@dataclass(frozen=True)
class LLMRequest:
    model: str
    messages: tuple[tuple[str, str], ...]
    purpose: Purpose
    json_schema: dict[str, Any] | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    salt: str = ""
    """Changes the cache key, for deliberate re-sampling such as teacher stability checks."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    outcome: Outcome
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    detail: str = ""


class Backend(Protocol):
    provider_class: ProviderClass

    async def complete(self, request: LLMRequest) -> LLMResponse: ...

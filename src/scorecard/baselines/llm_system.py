"""Zero-shot, few-shot and API baselines. All share one prompt, one parser and one record format."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from scorecard.data.models import Transcript
from scorecard.llm.client import LLMClient
from scorecard.llm.types import LLMRequest, Outcome, Purpose
from scorecard.prompts import small_model_messages
from scorecard.schema import Scorecard, scorecard_json_schema

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass(frozen=True)
class Prediction:
    scenario_id: str
    card: Scorecard | None
    parse_valid: bool
    outcome: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: str = ""


def parse_scorecard(text: str) -> tuple[Scorecard | None, bool]:
    """Strict parsing after removing wrappers. The bool reports validity on the raw model output."""
    cleaned = _FENCE.sub("", _THINK.sub("", text).strip())
    try:
        json.loads(cleaned)
        return Scorecard.model_validate_json(cleaned), True
    except (json.JSONDecodeError, ValidationError):
        return None, False


async def predict_llm(
    client: LLMClient,
    model: str,
    transcript: Transcript,
    examples: tuple[tuple[Transcript, str], ...] = (),
    constrained: bool = True,
    max_tokens: int = 1024,
) -> Prediction:
    response = await client.complete(
        LLMRequest(
            model=model,
            messages=small_model_messages(transcript, examples),
            purpose=Purpose.BASELINE,
            json_schema=scorecard_json_schema() if constrained else None,
            max_tokens=max_tokens,
        )
    )
    if response.outcome is not Outcome.OK:
        return Prediction(
            transcript.scenario_id,
            None,
            False,
            response.outcome.value,
            response.input_tokens,
            response.output_tokens,
            response.text,
        )
    card, valid = parse_scorecard(response.text)
    return Prediction(
        transcript.scenario_id,
        card,
        valid,
        "ok" if valid else "invalid",
        response.input_tokens,
        response.output_tokens,
        response.text,
    )


async def predict_all(
    client: LLMClient,
    model: str,
    transcripts: Sequence[Transcript],
    examples: tuple[tuple[Transcript, str], ...] = (),
    constrained: bool = True,
) -> list[Prediction]:
    return list(
        await asyncio.gather(
            *(predict_llm(client, model, t, examples, constrained) for t in transcripts)
        )
    )

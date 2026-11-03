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
from scorecard.train.format import (
    PER_FIELD_NAMES,
    assemble_fields,
    per_field_messages,
    per_field_schema,
)

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


async def predict_per_field(
    client: LLMClient,
    model: str,
    transcript: Transcript,
    constrained: bool = True,
    max_tokens: int = 512,
) -> Prediction:
    """One call per field, for the ablation arm that trades inference cost for focus."""
    responses = await asyncio.gather(
        *(
            client.complete(
                LLMRequest(
                    model=model,
                    messages=tuple(
                        (m["role"], m["content"]) for m in per_field_messages(transcript, field)
                    ),
                    purpose=Purpose.BASELINE,
                    json_schema=per_field_schema(field) if constrained else None,
                    max_tokens=max_tokens,
                )
            )
            for field in PER_FIELD_NAMES
        )
    )
    input_tokens = sum(r.input_tokens for r in responses)
    output_tokens = sum(r.output_tokens for r in responses)
    bad = next((r for r in responses if r.outcome is not Outcome.OK), None)
    if bad is not None:
        return Prediction(
            transcript.scenario_id, None, False, bad.outcome.value, input_tokens, output_tokens
        )
    outputs = {
        field: _FENCE.sub("", _THINK.sub("", r.text).strip())
        for field, r in zip(PER_FIELD_NAMES, responses, strict=True)
    }
    try:
        card = assemble_fields(outputs)
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError):
        return Prediction(
            transcript.scenario_id, None, False, "invalid", input_tokens, output_tokens
        )
    return Prediction(transcript.scenario_id, card, True, "ok", input_tokens, output_tokens)


async def predict_all(
    client: LLMClient,
    model: str,
    transcripts: Sequence[Transcript],
    examples: tuple[tuple[Transcript, str], ...] = (),
    constrained: bool = True,
    per_field: bool = False,
) -> list[Prediction]:
    if per_field:
        return list(
            await asyncio.gather(
                *(predict_per_field(client, model, t, constrained) for t in transcripts)
            )
        )
    return list(
        await asyncio.gather(
            *(predict_llm(client, model, t, examples, constrained) for t in transcripts)
        )
    )

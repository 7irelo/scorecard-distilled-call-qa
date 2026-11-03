"""Turn labelled calls into training examples. Needs no ML libraries, so it is fully unit-tested."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from scorecard.data.models import LanguageMix, Transcript
from scorecard.data.reality import assert_no_reality_rows
from scorecard.prompts import COMPACT_TASK, user_message
from scorecard.schema import (
    COMPLIANCE_CHECKS,
    Compliance,
    ComplianceCheck,
    Escalation,
    Scorecard,
    Sentiment,
)
from scorecard.train.config import LabelFilter, TrainConfig

IGNORE_INDEX = -100


class ChatTokenizer(Protocol):
    eos_token: str

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


class CallTypeField(BaseModel):
    call_type: str


class SummaryField(BaseModel):
    summary: str


def per_field_targets(card: Scorecard) -> dict[str, str]:
    """Each field as its own target, for the per-field-prompt ablation."""
    targets = {
        "summary": json.dumps({"summary": card.summary}, ensure_ascii=False),
        "call_type": json.dumps({"call_type": card.call_type.value}),
        "sentiment": card.sentiment.model_dump_json(),
        "escalation": card.escalation.model_dump_json(),
    }
    for check in COMPLIANCE_CHECKS:
        targets[f"compliance.{check}"] = getattr(card.compliance, check).model_dump_json()
    return targets


PER_FIELD_NAMES: tuple[str, ...] = (
    "summary",
    "call_type",
    "sentiment",
    "escalation",
    *(f"compliance.{check}" for check in COMPLIANCE_CHECKS),
)


def per_field_schema(field: str) -> dict[str, Any]:
    models: dict[str, type[BaseModel]] = {
        "summary": SummaryField,
        "call_type": CallTypeField,
        "sentiment": Sentiment,
        "escalation": Escalation,
    }
    model = ComplianceCheck if field.startswith("compliance.") else models[field]
    return model.model_json_schema()


def assemble_fields(outputs: dict[str, str]) -> Scorecard:
    """Rebuild a scorecard from per-field outputs. Raises if any field is missing or invalid."""
    compliance = {check: json.loads(outputs[f"compliance.{check}"]) for check in COMPLIANCE_CHECKS}
    return Scorecard.model_validate(
        {
            "summary": json.loads(outputs["summary"])["summary"],
            "call_type": json.loads(outputs["call_type"])["call_type"],
            "sentiment": json.loads(outputs["sentiment"]),
            "compliance": Compliance.model_validate(compliance),
            "escalation": json.loads(outputs["escalation"]),
        }
    )


def per_field_messages(transcript: Transcript, field: str) -> list[dict[str, str]]:
    system = f"{COMPACT_TASK}\n\nReturn only this one part of the scorecard, as JSON: {field}."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message(transcript)},
    ]


def full_messages(transcript: Transcript) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": COMPACT_TASK},
        {"role": "user", "content": user_message(transcript)},
    ]


def render_prompt(
    tokenizer: ChatTokenizer, messages: list[dict[str, str]], disable_thinking: bool = True
) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if disable_thinking:
        # Templates without this variable ignore it.
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def encode_example(
    tokenizer: ChatTokenizer,
    messages: list[dict[str, str]],
    target: str,
    max_seq_length: int,
    disable_thinking: bool = True,
) -> dict[str, list[int]] | None:
    """Tokenize with loss on assistant tokens only. Over-long examples return None, never truncated.

    Calls have disclosures at the start and resolutions at the end, so cutting either silently
    teaches the model the wrong label.
    """
    prompt = render_prompt(tokenizer, messages, disable_thinking)
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    target_ids = list(
        tokenizer(target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    )
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_seq_length:
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + target_ids,
    }


def expand_row(
    row: dict[str, Any], config: TrainConfig
) -> Iterable[tuple[list[dict[str, str]], str]]:
    transcript = Transcript.model_validate(row["transcript"])
    card = Scorecard.model_validate(row["card"])
    if config.output_mode.value == "per_field":
        for field, target in per_field_targets(card).items():
            yield per_field_messages(transcript, field), target
    else:
        yield full_messages(transcript), card.canonical_json()


@dataclass(frozen=True)
class EncodingReport:
    encoded: int
    too_long: int
    max_length: int


def encode_rows(
    tokenizer: ChatTokenizer, rows: Iterable[dict[str, Any]], config: TrainConfig
) -> tuple[list[dict[str, list[int]]], EncodingReport]:
    encoded: list[dict[str, list[int]]] = []
    too_long = longest = 0
    for row in rows:
        for messages, target in expand_row(row, config):
            example = encode_example(
                tokenizer, messages, target, config.max_seq_length, config.disable_thinking
            )
            if example is None:
                too_long += 1
                continue
            longest = max(longest, len(example["input_ids"]))
            encoded.append(example)
    return encoded, EncodingReport(len(encoded), too_long, longest)


def _stable_order(scenario_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{scenario_id}".encode()).hexdigest()


def select_training_rows(rows: list[dict[str, Any]], config: TrainConfig) -> list[dict[str, Any]]:
    """Apply ablation filters. Sizes are nested: the 500-call set contains the 250-call set.

    Ordering uses a fixed key rather than the run seed, so the size ablation changes one thing.
    """
    assert_no_reality_rows(rows)
    selected = [r for r in rows if r.get("card") is not None]
    if config.label_filter is LabelFilter.CLEAN:
        selected = [r for r in selected if r.get("qc_clean", False)]
    selected.sort(key=lambda r: _stable_order(r["scenario_id"], 0))
    if config.code_switch_share < 1.0:
        english = [r for r in selected if r["language_mix"] == LanguageMix.EN.value]
        switched = [r for r in selected if r["language_mix"] != LanguageMix.EN.value]
        keep = int(len(switched) * config.code_switch_share)
        selected = english + switched[:keep]
        selected.sort(key=lambda r: _stable_order(r["scenario_id"], 0))
    if config.train_size is not None:
        if config.train_size > len(selected):
            raise ValueError(f"asked for {config.train_size} rows, only {len(selected)} available")
        selected = selected[: config.train_size]
    return selected


def describe_selection(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(r["language_mix"] for r in rows))

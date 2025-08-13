"""Teacher labelling and label quality control."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field

from pydantic import ValidationError

from scorecard.data.models import ScenarioSpec, Transcript
from scorecard.io import sha256_text
from scorecard.llm.client import LLMClient
from scorecard.llm.types import LLMRequest, Outcome, Purpose
from scorecard.prompts import teacher_messages
from scorecard.schema import Scorecard, flatten_labels, scorecard_json_schema, semantic_issues


@dataclass(frozen=True)
class TeacherLabel:
    scenario_id: str
    card: Scorecard | None
    outcome: str
    model: str
    prompt_hash: str
    failure: str = ""

    def to_row(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "card": self.card.model_dump(mode="json") if self.card else None,
            "outcome": self.outcome,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "failure": self.failure,
        }


async def label_transcript(
    client: LLMClient,
    model: str,
    rubric: str,
    transcript: Transcript,
    examples: tuple[tuple[Transcript, str], ...] = (),
    salt: str = "",
) -> TeacherLabel:
    messages = teacher_messages(rubric, transcript, examples)
    # Hash covers only the stable part of the prompt, so it identifies the prompt version.
    prompt_hash = sha256_text(messages[0][1])
    response = await client.complete(
        LLMRequest(
            model=model,
            messages=messages,
            purpose=Purpose.TEACHER,
            json_schema=scorecard_json_schema(),
            max_tokens=2048,
            salt=salt,
        )
    )
    if response.outcome is not Outcome.OK:
        return TeacherLabel(
            transcript.scenario_id,
            None,
            response.outcome.value,
            model,
            prompt_hash,
            response.detail,
        )
    try:
        card = Scorecard.model_validate_json(response.text)
    except ValidationError as exc:
        return TeacherLabel(
            transcript.scenario_id, None, "invalid", model, prompt_hash, str(exc)[:500]
        )
    return TeacherLabel(transcript.scenario_id, card, "ok", model, prompt_hash)


async def label_all(
    client: LLMClient,
    model: str,
    rubric: str,
    transcripts: list[Transcript],
    examples: tuple[tuple[Transcript, str], ...] = (),
    salt: str = "",
) -> list[TeacherLabel]:
    return list(
        await asyncio.gather(
            *(label_transcript(client, model, rubric, t, examples, salt) for t in transcripts)
        )
    )


def planted_labels(spec: ScenarioSpec) -> dict[str, str]:
    """Planted truth, expressed in the same flat form as flatten_labels."""
    labels = {f"compliance.{check}": status.value for check, status in spec.compliance.items()}
    labels["escalation.flag"] = str(spec.escalation.flag).lower()
    labels["escalation.reason"] = spec.escalation.reason.value
    labels["call_type"] = spec.call_type.value
    labels["sentiment.opening"] = spec.sentiment.opening.value
    labels["sentiment.middle"] = spec.sentiment.middle.value
    labels["sentiment.closing"] = spec.sentiment.closing.value
    return labels


@dataclass(frozen=True)
class QCResult:
    scenario_id: str
    semantic: tuple[str, ...] = ()
    planted_disagreements: tuple[str, ...] = ()
    clean: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "clean", not self.semantic and not self.planted_disagreements)


def qc_label(spec: ScenarioSpec, transcript: Transcript, card: Scorecard) -> QCResult:
    flat = flatten_labels(card)
    disagreements = tuple(
        f"{name}:planted_{planted}_teacher_{flat[name]}"
        for name, planted in planted_labels(spec).items()
        if flat[name] != planted
    )
    return QCResult(
        spec.scenario_id,
        tuple(semantic_issues(card, transcript.turn_ids())),
        disagreements,
    )


def stability(first: list[TeacherLabel], second: list[TeacherLabel]) -> dict[str, float]:
    """Per-field agreement between two labelling passes. Limits achievable student agreement."""
    by_id = {label.scenario_id: label.card for label in second}
    agree: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for label in first:
        other = by_id.get(label.scenario_id)
        if label.card is None or other is None:
            continue
        a, b = flatten_labels(label.card), flatten_labels(other)
        for name in a:
            total[name] += 1
            agree[name] += a[name] == b[name]
    return {name: agree[name] / total[name] for name in total}


def qc_report(results: list[QCResult], labels: list[TeacherLabel]) -> dict[str, object]:
    outcomes = Counter(label.outcome for label in labels)
    semantic = Counter(issue for r in results for issue in r.semantic)
    disagreements = Counter(d.split(":")[0] for r in results for d in r.planted_disagreements)
    return {
        "labels": len(labels),
        "outcomes": dict(outcomes),
        "clean_share": sum(r.clean for r in results) / len(results) if results else 0.0,
        "semantic_issues": dict(semantic),
        "planted_disagreements_by_field": dict(disagreements),
    }

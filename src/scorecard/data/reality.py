"""The reality-check set: role-played calls, recorded with consent and kept out of training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scorecard.data.models import Speaker, Transcript, Turn

REALITY_PREFIX = "reality-"


class ConsentMissing(ValueError):
    """A reality-check call has no consent record, so it can't be used."""


class RealityLeak(ValueError):
    """A reality-check call reached a training set."""


def utterances_to_turns(
    utterances: Sequence[Mapping[str, Any]], agent_speaker: int = 0
) -> tuple[Turn, ...]:
    """Merge consecutive diarized utterances from one speaker into a single turn."""
    merged: list[tuple[int, str]] = []
    for utterance in utterances:
        text = str(utterance["transcript"]).strip()
        if not text:
            continue
        speaker = int(utterance["speaker"])
        if merged and merged[-1][0] == speaker:
            merged[-1] = (speaker, f"{merged[-1][1]} {text}")
        else:
            merged.append((speaker, text))
    return tuple(
        Turn(
            id=index,
            speaker=Speaker.AGENT if speaker == agent_speaker else Speaker.CUSTOMER,
            text=text,
        )
        for index, (speaker, text) in enumerate(merged, start=1)
    )


def reality_row(call_id: str, consent_id: str, turns: Sequence[Turn]) -> dict[str, Any]:
    if not consent_id.strip():
        raise ConsentMissing(f"call {call_id} has no consent record")
    transcript = Transcript(
        scenario_id=f"{REALITY_PREFIX}{call_id}", turns=tuple(turns), noise="real_stt"
    )
    return {"consent_id": consent_id, **transcript.model_dump(mode="json")}


def load_reality_rows(rows: Iterable[Mapping[str, Any]]) -> list[Transcript]:
    """Reads a reality file, refusing any call with no consent record or the wrong ID prefix."""
    transcripts: list[Transcript] = []
    for row in rows:
        scenario_id = str(row.get("scenario_id", ""))
        if not scenario_id.startswith(REALITY_PREFIX):
            raise ValueError(f"{scenario_id!r} is not a reality-check ID")
        if not str(row.get("consent_id", "")).strip():
            raise ConsentMissing(f"call {scenario_id} has no consent record")
        body = {key: value for key, value in row.items() if key != "consent_id"}
        transcripts.append(Transcript.model_validate(body))
    return transcripts


def assert_no_reality_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        if str(row["scenario_id"]).startswith(REALITY_PREFIX):
            raise RealityLeak(f"{row['scenario_id']} must stay out of training")


@dataclass(frozen=True)
class SimToRealGap:
    system: str
    synthetic: float
    reality: float
    n_reality: int

    @property
    def gap(self) -> float:
        return self.synthetic - self.reality


def gap_markdown(gaps: Sequence[SimToRealGap]) -> str:
    lines = [
        "# Sim-to-real gap",
        "",
        "Structured score (Q without the summary term) on the synthetic test set and on recorded "
        "role-played calls. The recorded set is small, so treat this as a warning sign rather "
        "than a precise measure. Role-play is also not the same as real customer calls.",
        "",
        "| System | Synthetic | Reality check | Gap | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in gaps:
        lines.append(
            f"| {item.system} | {item.synthetic:.3f} | {item.reality:.3f} | "
            f"{item.gap:+.3f} | {item.n_reality} |"
        )
    return "\n".join(lines) + "\n"

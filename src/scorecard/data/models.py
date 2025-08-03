"""Scenario specs (planted truth) and transcripts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from scorecard.schema import CallType, ComplianceStatus, EscalationReason, SentimentLabel


class Speaker(StrEnum):
    AGENT = "AGENT"
    CUSTOMER = "CUSTOMER"


class LanguageMix(StrEnum):
    EN = "en"
    EN_ZU = "en_zu"
    EN_AF = "en_af"
    EN_ZU_AF = "en_zu_af"


class SwitchStyle(StrEnum):
    NONE = "none"
    SINGLE_INSERTIONS = "single_insertions"
    INTRA_SENTENCE = "intra_sentence"
    WHOLE_TURN = "whole_turn"
    WHEN_UPSET = "when_upset"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlantedSentiment(_Frozen):
    opening: SentimentLabel
    middle: SentimentLabel
    closing: SentimentLabel


class PlantedEscalation(_Frozen):
    flag: bool
    reason: EscalationReason


class ScenarioSpec(_Frozen):
    scenario_id: str
    group_id: str
    seed: int
    call_type: CallType
    issue_id: str
    issue: str
    company: str
    agent_name: str
    customer_name: str
    temperament: str
    language_mix: LanguageMix
    switch_style: SwitchStyle
    target_turns: int
    outcome: str
    hung_up_early: bool
    compliance: dict[str, ComplianceStatus]
    sentiment: PlantedSentiment
    escalation: PlantedEscalation
    weight: float
    """Inverse sampling correction, used to recover population estimates after oversampling."""


class Turn(_Frozen):
    id: int
    speaker: Speaker
    text: str


class Transcript(_Frozen):
    scenario_id: str
    turns: tuple[Turn, ...]
    noise: str = "clean"

    def turn_ids(self) -> set[int]:
        return {turn.id for turn in self.turns}

    def render(self) -> str:
        return "\n".join(f"[{t.id}] {t.speaker.value}: {t.text}" for t in self.turns)

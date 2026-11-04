"""The scorecard schema: single source of truth for labels, targets, constraints and responses."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"
MAX_SUMMARY_WORDS = 80
# Hard caps stop constrained decoding from running to max_tokens on a stuck model. The 80-word
# rule stays in semantic_issues, because a JSON schema can only limit characters.
MAX_SUMMARY_CHARS = 1200
MAX_EVIDENCE_TURNS = 10


class CallType(StrEnum):
    BILLING_DISPUTE = "billing_dispute"
    OUTAGE = "outage"
    CANCELLATION = "cancellation"
    OTHER = "other"


class SentimentLabel(StrEnum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


SENTIMENT_ORDER = {label: rank for rank, label in enumerate(SentimentLabel)}


class Trajectory(StrEnum):
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"


class ComplianceStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class EscalationReason(StrEnum):
    CUSTOMER_THREATENED_REGULATOR = "customer_threatened_regulator"
    UNRESOLVED_REPEAT_CONTACT = "unresolved_repeat_contact"
    AGENT_ERROR = "agent_error"
    SAFETY = "safety"
    NONE = "none"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Sentiment(_Strict):
    opening: SentimentLabel
    middle: SentimentLabel
    closing: SentimentLabel
    trajectory: Trajectory


class ComplianceCheck(_Strict):
    status: ComplianceStatus
    evidence_turns: list[int] = Field(default_factory=list, max_length=MAX_EVIDENCE_TURNS)


class Compliance(_Strict):
    agent_introduced: ComplianceCheck
    recording_disclosed: ComplianceCheck
    identity_verified: ComplianceCheck
    terms_explained: ComplianceCheck
    next_steps_confirmed: ComplianceCheck


COMPLIANCE_CHECKS: tuple[str, ...] = tuple(Compliance.model_fields)


class Escalation(_Strict):
    flag: bool
    reason: EscalationReason


class Scorecard(_Strict):
    schema_version: Literal["1.0"] = "1.0"
    summary: str = Field(max_length=MAX_SUMMARY_CHARS)
    call_type: CallType
    sentiment: Sentiment
    compliance: Compliance
    escalation: Escalation

    def canonical_json(self) -> str:
        """Stable field order and no ASCII escaping, so training targets stay stable."""
        return self.model_dump_json()


def scorecard_json_schema() -> dict[str, Any]:
    return Scorecard.model_json_schema()


def semantic_issues(card: Scorecard, turn_ids: set[int] | None = None) -> list[str]:
    """Rules a JSON schema cannot express. An empty list means the card is coherent."""
    issues: list[str] = []
    if not card.summary.strip():
        issues.append("summary_empty")
    if len(card.summary.split()) > MAX_SUMMARY_WORDS:
        issues.append("summary_too_long")

    opening = SENTIMENT_ORDER[card.sentiment.opening]
    closing = SENTIMENT_ORDER[card.sentiment.closing]
    expected = (
        Trajectory.IMPROVING
        if closing > opening
        else Trajectory.WORSENING
        if closing < opening
        else Trajectory.STABLE
    )
    if card.sentiment.trajectory is not expected:
        issues.append("trajectory_contradicts_segments")

    if card.escalation.flag != (card.escalation.reason is not EscalationReason.NONE):
        issues.append("escalation_flag_reason_mismatch")

    for name in COMPLIANCE_CHECKS:
        check: ComplianceCheck = getattr(card.compliance, name)
        if check.status is ComplianceStatus.PASS and not check.evidence_turns:
            issues.append(f"{name}_pass_without_evidence")
        if check.status is ComplianceStatus.NOT_APPLICABLE and check.evidence_turns:
            issues.append(f"{name}_not_applicable_with_evidence")
        if turn_ids is not None and not set(check.evidence_turns) <= turn_ids:
            issues.append(f"{name}_evidence_turn_missing")
    return issues


def flatten_labels(card: Scorecard) -> dict[str, str]:
    """Categorical fields as a flat mapping, the shared unit of scoring and agreement."""
    flat = {
        "call_type": card.call_type.value,
        "sentiment.opening": card.sentiment.opening.value,
        "sentiment.middle": card.sentiment.middle.value,
        "sentiment.closing": card.sentiment.closing.value,
        "sentiment.trajectory": card.sentiment.trajectory.value,
        "escalation.flag": str(card.escalation.flag).lower(),
        "escalation.reason": card.escalation.reason.value,
    }
    for name in COMPLIANCE_CHECKS:
        flat[f"compliance.{name}"] = getattr(card.compliance, name).status.value
    return flat

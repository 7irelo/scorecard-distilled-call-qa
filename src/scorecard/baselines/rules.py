"""Regex baseline. Answers the question: does this task need an LLM at all?"""

from __future__ import annotations

import re

from scorecard.data.models import Speaker, Transcript, Turn
from scorecard.schema import (
    COMPLIANCE_CHECKS,
    SENTIMENT_ORDER,
    CallType,
    Compliance,
    ComplianceCheck,
    ComplianceStatus,
    Escalation,
    EscalationReason,
    Scorecard,
    Sentiment,
    SentimentLabel,
    Trajectory,
)

_INTRO = re.compile(
    r"\b(my name is|you'?re speaking (to|with)|this is \w+ (from|at)|ngingu)\b", re.I
)
_RECORDING = re.compile(r"\b(record(ed|ing)?|opgeneem|opname)\b", re.I)
_IDENTITY = re.compile(
    r"\b(id (number|digits)|date of birth|account number|address|passport|last \w+ digits)\b", re.I
)
_TERMS = re.compile(
    r"\b(notice period|exit fee|cancellation fee|contract term|terms|penalty)\b", re.I
)
_NEXT = re.compile(
    r"\b(next step|within \w+ (hours|days)|will (call|sms|email|send)|reference number|"
    r"technician will|you will (get|receive))\b",
    re.I,
)
_REGULATOR = re.compile(r"\b(ombud|ombudsman|icasa|regulator|consumer commission|hopewell)\b", re.I)
_REPEAT = re.compile(
    r"\b(third time|fourth time|called (again|several)|keep calling|many times)\b", re.I
)
_SAFETY = re.compile(r"\b(unsafe|danger|emergency|medical|alarm)\b", re.I)
_AGENT_ERROR = re.compile(r"\b(my mistake|we made an error|that was wrong|i was wrong)\b", re.I)
_NEGATIVE = re.compile(
    r"\b(angry|unacceptable|ridiculous|useless|fed up|terrible|eish|hayibo|frustrat\w+)\b", re.I
)
_POSITIVE = re.compile(r"\b(thank you|thanks|ngiyabonga|dankie|great|appreciate|perfect)\b", re.I)
_CANCEL = re.compile(r"\b(cancel|kanselleer|terminate)\w*\b", re.I)
_OUTAGE = re.compile(r"\b(down|outage|no signal|offline|load.?shedding|router|ugesi)\b", re.I)
_BILLING = re.compile(r"\b(bill|debit|charged?|refund|statement|invoice|imali|rekening)\b", re.I)


def _matches(turns: list[Turn], pattern: re.Pattern[str]) -> list[int]:
    return [turn.id for turn in turns if pattern.search(turn.text)]


def _segment_sentiment(turns: list[Turn]) -> SentimentLabel:
    text = " ".join(turn.text for turn in turns)
    score = len(_POSITIVE.findall(text)) - len(_NEGATIVE.findall(text))
    if score <= -2:
        return SentimentLabel.VERY_NEGATIVE
    if score == -1:
        return SentimentLabel.NEGATIVE
    if score == 0:
        return SentimentLabel.NEUTRAL
    return SentimentLabel.POSITIVE if score == 1 else SentimentLabel.VERY_POSITIVE


def _call_type(text: str) -> CallType:
    counts = {
        CallType.CANCELLATION: len(_CANCEL.findall(text)),
        CallType.OUTAGE: len(_OUTAGE.findall(text)),
        CallType.BILLING_DISPUTE: len(_BILLING.findall(text)),
    }
    best = max(counts, key=lambda key: counts[key])
    return best if counts[best] else CallType.OTHER


def predict_rules(transcript: Transcript) -> Scorecard:
    turns = list(transcript.turns)
    agent = [turn for turn in turns if turn.speaker is Speaker.AGENT]
    customer = [turn for turn in turns if turn.speaker is Speaker.CUSTOMER]
    text = " ".join(turn.text for turn in turns)
    call_type = _call_type(text)

    def check(evidence: list[int]) -> ComplianceCheck:
        status = ComplianceStatus.PASS if evidence else ComplianceStatus.FAIL
        return ComplianceCheck(status=status, evidence_turns=evidence[:3])

    identity = _matches(agent, _IDENTITY)
    terms = _matches(agent, _TERMS)
    results = {
        "agent_introduced": check(_matches(agent[:5], _INTRO)),
        "recording_disclosed": check(_matches(agent[:6], _RECORDING)),
        "identity_verified": check(identity if len(identity) >= 2 else []),
        "terms_explained": check(terms)
        if call_type in {CallType.CANCELLATION, CallType.BILLING_DISPUTE}
        else ComplianceCheck(status=ComplianceStatus.NOT_APPLICABLE),
        "next_steps_confirmed": check(_matches(agent[-6:], _NEXT)),
    }
    assert set(results) == set(COMPLIANCE_CHECKS)

    third = max(1, len(customer) // 3)
    opening = _segment_sentiment(customer[:third])
    middle = _segment_sentiment(customer[third:-third] or customer)
    closing = _segment_sentiment(customer[-third:])
    trajectory = (
        Trajectory.IMPROVING
        if SENTIMENT_ORDER[closing] > SENTIMENT_ORDER[opening]
        else Trajectory.WORSENING
        if SENTIMENT_ORDER[closing] < SENTIMENT_ORDER[opening]
        else Trajectory.STABLE
    )

    reason = EscalationReason.NONE
    for pattern, candidate in (
        (_SAFETY, EscalationReason.SAFETY),
        (_REGULATOR, EscalationReason.CUSTOMER_THREATENED_REGULATOR),
        (_AGENT_ERROR, EscalationReason.AGENT_ERROR),
        (_REPEAT, EscalationReason.UNRESOLVED_REPEAT_CONTACT),
    ):
        if pattern.search(text):
            reason = candidate
            break

    summary = " ".join(" ".join(t.text for t in customer[:2]).split()[:60]) or "No customer speech."
    return Scorecard(
        summary=summary,
        call_type=call_type,
        sentiment=Sentiment(opening=opening, middle=middle, closing=closing, trajectory=trajectory),
        compliance=Compliance(**results),
        escalation=Escalation(flag=reason is not EscalationReason.NONE, reason=reason),
    )

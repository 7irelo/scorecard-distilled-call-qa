"""Blind labelling, adjudication and inter-rater agreement.

The blind pass takes only transcripts, so teacher output cannot anchor the human rater.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from scorecard.data.models import Transcript
from scorecard.schema import (
    COMPLIANCE_CHECKS,
    CallType,
    ComplianceStatus,
    EscalationReason,
    Scorecard,
    SentimentLabel,
    Trajectory,
    flatten_labels,
    semantic_issues,
)

Ask = Callable[[str], str]
Say = Callable[[str], None]


def _choose(ask: Ask, say: Say, prompt: str, options: list[str]) -> str:
    menu = "  ".join(f"{i + 1}={option}" for i, option in enumerate(options))
    while True:
        reply = ask(f"{prompt} [{menu}]: ").strip()
        if reply.isdigit() and 1 <= int(reply) <= len(options):
            return options[int(reply) - 1]
        say("Enter one of the listed numbers.")


def _turns(ask: Ask, say: Say, prompt: str, valid: set[int]) -> list[int]:
    while True:
        reply = ask(f"{prompt} evidence turn IDs, comma separated: ").strip()
        try:
            turns = [int(part) for part in reply.split(",") if part.strip()]
        except ValueError:
            say("Use numbers separated by commas.")
            continue
        if set(turns) <= valid:
            return turns
        say("Those turns are not in this transcript.")


def label_call(transcript: Transcript, ask: Ask, say: Say) -> Scorecard:
    """One blind pass over one call. Repeats until the card is coherent."""
    say(transcript.render())
    valid = transcript.turn_ids()
    while True:
        sentiment_options = [s.value for s in SentimentLabel]
        compliance: dict[str, dict[str, object]] = {}
        for check in COMPLIANCE_CHECKS:
            status = _choose(ask, say, check, [s.value for s in ComplianceStatus])
            turns = (
                []
                if status == ComplianceStatus.NOT_APPLICABLE.value
                else _turns(ask, say, check, valid)
            )
            compliance[check] = {"status": status, "evidence_turns": turns}
        reason = _choose(ask, say, "escalation reason", [r.value for r in EscalationReason])
        card = Scorecard.model_validate(
            {
                "summary": ask("Summary, 80 words at most: ").strip(),
                "call_type": _choose(ask, say, "call type", [c.value for c in CallType]),
                "sentiment": {
                    "opening": _choose(ask, say, "sentiment opening", sentiment_options),
                    "middle": _choose(ask, say, "sentiment middle", sentiment_options),
                    "closing": _choose(ask, say, "sentiment closing", sentiment_options),
                    "trajectory": _choose(ask, say, "trajectory", [t.value for t in Trajectory]),
                },
                "compliance": compliance,
                "escalation": {"flag": reason != EscalationReason.NONE.value, "reason": reason},
            }
        )
        issues = semantic_issues(card, valid)
        if not issues:
            return card
        say(f"Please fix: {', '.join(issues)}")


def run_blind_session(
    transcripts: Iterable[Transcript], output: Path, rater: str, ask: Ask, say: Say
) -> int:
    """Label calls in order, saving after each one. Finished calls are skipped on resume."""
    done: set[str] = set()
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            done = {json.loads(line)["scenario_id"] for line in handle if line.strip()}
    output.parent.mkdir(parents=True, exist_ok=True)
    labelled = 0
    for transcript in transcripts:
        if transcript.scenario_id in done:
            continue
        card = label_call(transcript, ask, say)
        row = {
            "scenario_id": transcript.scenario_id,
            "rater": rater,
            "card": card.model_dump(mode="json"),
        }
        with output.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        labelled += 1
    return labelled


@dataclass(frozen=True)
class Disagreement:
    scenario_id: str
    field: str
    human: str
    teacher: str | None
    planted: str | None


def find_disagreements(
    scenario_id: str,
    human: Scorecard,
    teacher: Scorecard | None,
    planted: dict[str, str],
) -> list[Disagreement]:
    """Fields where the human differs from teacher or planted truth. These go to adjudication."""
    human_flat = flatten_labels(human)
    teacher_flat = flatten_labels(teacher) if teacher else {}
    return [
        Disagreement(scenario_id, name, value, teacher_flat.get(name), planted.get(name))
        for name, value in human_flat.items()
        if value != teacher_flat.get(name) or (name in planted and value != planted[name])
    ]


def teacher_error_taxonomy(
    decisions: Iterable[dict[str, str]],
) -> dict[str, dict[str, int]]:
    """Count adjudication verdicts per field. `verdict` names who was right."""
    table: dict[str, Counter[str]] = {}
    for decision in decisions:
        table.setdefault(decision["field"], Counter())[decision["verdict"]] += 1
    return {name: dict(counts) for name, counts in table.items()}


def inter_rater_agreement(
    first: dict[str, Scorecard], second: dict[str, Scorecard]
) -> dict[str, float]:
    """Cohen's kappa per field across the calls both raters labelled."""
    shared = sorted(set(first) & set(second))
    if not shared:
        return {}
    a = [flatten_labels(first[s]) for s in shared]
    b = [flatten_labels(second[s]) for s in shared]
    result: dict[str, float] = {}
    for name in a[0]:
        left, right = [row[name] for row in a], [row[name] for row in b]
        if left == right:
            result[name] = 1.0
        else:
            result[name] = float(cohen_kappa_score(left, right))
    return result

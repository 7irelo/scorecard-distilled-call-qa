"""Adjudication after the blind pass, and the teacher error-rate report.

The teacher's error rate is the project's core data-quality claim. It is reported per field, with
intervals, and with human mistakes counted alongside the teacher's.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scorecard.data.models import Transcript
from scorecard.eval.stats import Interval, wilson
from scorecard.gold.labelling import Ask, Disagreement, Say, find_disagreements
from scorecard.schema import Scorecard, flatten_labels, semantic_issues

VERDICTS = ("human", "teacher", "planted", "other")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def adjudication_queue(
    human: Mapping[str, Scorecard],
    teacher: Mapping[str, Scorecard | None],
    planted: Mapping[str, Mapping[str, str]],
) -> list[Disagreement]:
    queue: list[Disagreement] = []
    for scenario_id in sorted(human):
        queue += find_disagreements(
            scenario_id,
            human[scenario_id],
            teacher.get(scenario_id),
            dict(planted.get(scenario_id, {})),
        )
    return queue


def run_adjudication(
    queue: Iterable[Disagreement],
    transcripts: Mapping[str, Transcript],
    output: Path,
    ask: Ask,
    say: Say,
) -> int:
    """Resolve each disagreement, saving after each one. Finished items are skipped on resume."""
    done: set[tuple[str, str]] = set()
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        done = {(row["scenario_id"], row["field"]) for row in rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in queue:
        if (item.scenario_id, item.field) in done:
            continue
        say(transcripts[item.scenario_id].render())
        say(f"{item.field}: you={item.human} teacher={item.teacher} planted={item.planted}")
        verdict = ""
        while verdict not in VERDICTS:
            verdict = ask(f"Who is right? [{'/'.join(VERDICTS)}]: ").strip().lower()
        value = {"human": item.human, "teacher": item.teacher, "planted": item.planted}.get(verdict)
        if verdict == "other" or value is None:
            value = ask("Correct value: ").strip()
        evidence: list[int] | None = None
        if item.field.startswith("compliance.") and value == "pass" and value != item.human:
            reply = ask("Evidence turn IDs, comma separated: ")
            evidence = [int(part) for part in reply.split(",") if part.strip()]
        row: dict[str, Any] = {
            **asdict(item),
            "verdict": verdict,
            "value": value,
            "evidence_turns": evidence,
            "note": ask("Why (short note): ").strip(),
        }
        with output.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        count += 1
    return count


def _set_field(body: dict[str, Any], decision: Mapping[str, Any]) -> None:
    name, value = decision["field"], decision["value"]
    if name.startswith("compliance."):
        check = name.split(".", 1)[1]
        turns = decision.get("evidence_turns")
        current = body["compliance"][check]["evidence_turns"]
        body["compliance"][check] = {
            "status": value,
            "evidence_turns": [] if value != "pass" else (turns if turns is not None else current),
        }
    elif name.startswith("sentiment."):
        body["sentiment"][name.split(".", 1)[1]] = value
    elif name == "escalation.flag":
        body["escalation"]["flag"] = value == "true"
    elif name == "escalation.reason":
        body["escalation"]["reason"] = value
    else:
        body[name] = value


def apply_decisions(
    human: Mapping[str, Scorecard], decisions: Iterable[Mapping[str, Any]]
) -> tuple[dict[str, Scorecard], dict[str, list[str]]]:
    """Build final gold. Cards left incoherent are returned for fixing, never silently repaired."""
    bodies = {scenario_id: card.model_dump(mode="json") for scenario_id, card in human.items()}
    for decision in decisions:
        if decision["scenario_id"] in bodies:
            _set_field(bodies[decision["scenario_id"]], decision)
    cards = {key: Scorecard.model_validate(body) for key, body in bodies.items()}
    issues = {key: found for key, card in cards.items() if (found := semantic_issues(card))}
    return cards, issues


@dataclass(frozen=True)
class FieldAudit:
    field: str
    errors: int
    n: int
    error_rate: Interval
    weighted_error_rate: Interval
    english_error: float
    code_switched_error: float


def _kish_interval(wrong: Sequence[bool], weights: Sequence[float]) -> Interval:
    """Population estimate, with an interval sized by the effective sample after weighting."""
    total = sum(weights)
    if not total:
        return Interval(float("nan"), float("nan"), float("nan"))
    effective = total**2 / sum(w**2 for w in weights)
    rate = sum(w for bad, w in zip(wrong, weights, strict=True) if bad) / total
    return wilson(rate * effective, effective)


def _rate(flags: Sequence[bool]) -> float:
    return sum(flags) / len(flags) if flags else float("nan")


def audit_teacher(
    gold: Mapping[str, Scorecard],
    teacher: Mapping[str, Scorecard | None],
    weights: Mapping[str, float],
    language_mix: Mapping[str, str],
) -> list[FieldAudit]:
    """Per-field teacher error against final gold. A missing teacher label counts as an error."""
    ids = sorted(gold)
    if not ids:
        raise ValueError("no gold labels to audit")
    results: list[FieldAudit] = []
    for name in sorted(flatten_labels(gold[ids[0]])):
        wrong: list[bool] = []
        for scenario_id in ids:
            card = teacher.get(scenario_id)
            value = flatten_labels(card)[name] if card is not None else None
            wrong.append(value != flatten_labels(gold[scenario_id])[name])
        english = [w for w, i in zip(wrong, ids, strict=True) if language_mix[i] == "en"]
        switched = [w for w, i in zip(wrong, ids, strict=True) if language_mix[i] != "en"]
        results.append(
            FieldAudit(
                field=name,
                errors=sum(wrong),
                n=len(ids),
                error_rate=wilson(sum(wrong), len(ids)),
                weighted_error_rate=_kish_interval(wrong, [weights[i] for i in ids]),
                english_error=_rate(english),
                code_switched_error=_rate(switched),
            )
        )
    return results


def verdict_counts(decisions: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = {}
    for decision in decisions:
        table.setdefault(decision["field"], Counter())[decision["verdict"]] += 1
    return {name: dict(counts) for name, counts in sorted(table.items())}


def audit_markdown(
    audits: Sequence[FieldAudit],
    decisions: Sequence[Mapping[str, Any]] = (),
    summary_claims: Sequence[Mapping[str, Any]] = (),
    dry_run: bool = False,
) -> str:
    lines: list[str] = []
    if dry_run:
        lines.append("> **DRY RUN. Simulated labels, so these numbers mean nothing.**\n")
    calls = audits[0].n if audits else 0
    lines += [
        "# Teacher audit",
        "",
        f"Teacher labels compared with blind, adjudicated human gold on {calls} calls.",
        "",
        "| Field | Errors | Error rate [95% CI] | Population estimate [95% CI] | English | "
        "Code-switched |",
        "|---|---:|---|---|---:|---:|",
    ]
    for item in audits:
        lines.append(
            f"| {item.field} | {item.errors}/{item.n} | {item.error_rate.format()} | "
            f"{item.weighted_error_rate.format()} | {item.english_error:.3f} | "
            f"{item.code_switched_error:.3f} |"
        )
    if summary_claims:
        interval = wilson(
            sum(bool(row["supported"]) for row in summary_claims), len(summary_claims)
        )
        lines += ["", f"Teacher summary claims supported: {interval.format()}."]
    if decisions:
        counts = Counter(str(decision["verdict"]) for decision in decisions)
        lines += [
            "",
            "## Adjudication",
            "",
            f"{len(decisions)} disagreements. The first human label stood in "
            f"{counts['human']}, the teacher was right in {counts['teacher']}, planted truth in "
            f"{counts['planted']}, and {counts['other']} needed another value.",
            "",
            "| Field | Human | Teacher | Planted | Other |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, row in verdict_counts(decisions).items():
            lines.append(
                f"| {name} | {row.get('human', 0)} | {row.get('teacher', 0)} | "
                f"{row.get('planted', 0)} | {row.get('other', 0)} |"
            )
    return "\n".join(lines) + "\n"


def split_claims_simple(summary: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split(summary) if part.strip()]


def label_claims(
    rows: Iterable[Mapping[str, Any]],
    transcripts: Mapping[str, Transcript],
    output: Path,
    ask: Ask,
    say: Say,
) -> int:
    """Rate summaries sentence by sentence. Output feeds judge calibration and the teacher audit."""
    done: set[tuple[str, str]] = set()
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
        done = {(row["scenario_id"], row["claim"]) for row in existing}
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in rows:
        summary = row.get("summary") or row["card"]["summary"]
        scenario_id = row["scenario_id"]
        shown = False
        for claim in split_claims_simple(summary):
            if (scenario_id, claim) in done:
                continue
            if not shown:
                say(transcripts[scenario_id].render())
                shown = True
            reply = ""
            while reply not in {"y", "n"}:
                reply = ask(f"Supported? {claim} [y/n]: ").strip().lower()
            record = {
                "scenario_id": scenario_id,
                "claim": claim,
                "supported": reply == "y",
                "injected": row.get("injected"),
            }
            with output.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count

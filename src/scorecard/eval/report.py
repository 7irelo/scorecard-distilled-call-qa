"""Generate the comparison report. Numbers are never hand-copied into docs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from scorecard.eval.fields import (
    EvalSet,
    SystemOutput,
    SystemScore,
    paired_field_tests,
    quality_retained,
    score_system,
)
from scorecard.eval.stats import Interval, holm

DRY_RUN_BANNER = (
    "> **DRY RUN. These numbers come from stub models and simulated labels. They exercise the "
    "pipeline and say nothing about real quality.**\n"
)


def _fmt(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value:.3f}"


def score_table(scores: Sequence[SystemScore]) -> str:
    lines = [
        "| System | Q [95% CI] | Compliance | Escalation | Sentiment | Summary | Fail recall | "
        "Esc. recall | Evidence precision | Validity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in scores:
        lines.append(
            f"| {s.name} | {s.q_interval.format()} | {_fmt(s.parts['compliance'])} | "
            f"{_fmt(s.parts['escalation'])} | {_fmt(s.parts['sentiment'])} | "
            f"{_fmt(s.parts['summary'])} | {_fmt(s.fail_recall)} | {_fmt(s.escalation_recall)} | "
            f"{_fmt(s.evidence_precision)} | {_fmt(s.validity.point)} |"
        )
    return "\n".join(lines)


def slice_table(
    eval_set: EvalSet,
    outputs: Sequence[SystemOutput],
    slices: Mapping[str, str],
    resamples: int = 500,
) -> str:
    """Scores per slice. Exploratory by pre-registration, and always shown with counts."""
    lines = ["| Slice | n | " + " | ".join(o.name for o in outputs) + " |"]
    lines.append("|---|---:|" + "---:|" * len(outputs))
    for value in sorted(set(slices.values())):
        ids = tuple(i for i in eval_set.ids if slices[i] == value)
        if not ids:
            continue
        positions = [eval_set.ids.index(i) for i in ids]
        subset = EvalSet(ids, eval_set.gold, eval_set.weights[positions])
        scores = [score_system(subset, o, resamples).q_interval.format(2) for o in outputs]
        lines.append(f"| {value} | {len(ids)} | " + " | ".join(scores) + " |")
    return "\n".join(lines)


def build_report(
    eval_set: EvalSet,
    student: SystemOutput,
    teacher: SystemOutput,
    others: Sequence[SystemOutput],
    slices: Mapping[str, Mapping[str, str]] | None = None,
    dry_run: bool = False,
    post_hoc: Sequence[str] = (),
    resamples: int = 2000,
) -> tuple[str, dict[str, object]]:
    outputs = [student, teacher, *others]
    scores = [score_system(eval_set, output, resamples) for output in outputs]
    student_score = scores[0]
    retained: Interval = quality_retained(eval_set, student, teacher, resamples)
    floors = student_score.floors_met()
    tests = paired_field_tests(eval_set, student, teacher)
    survivors = holm(tests)

    lines: list[str] = []
    if dry_run:
        lines.append(DRY_RUN_BANNER)
    if post_hoc:
        reasons = "; ".join(post_hoc)
        lines.append(f"> **The test set was reopened after first scoring.** Reasons: {reasons}\n")
    failures = {s.name: s.judge_failures for s in scores if s.judge_failures}
    if failures:
        listed = ", ".join(f"{name}: {count}" for name, count in failures.items())
        lines.append(
            f"> **Judge failures, excluded from summary scores:** {listed}. "
            "Re-run the judge if these are more than a handful.\n"
        )
    lines += [
        "# Evaluation report",
        "",
        f"Evaluated on {len(eval_set.ids)} calls.",
        "",
        f"**Quality retained ({student.name} / {teacher.name}): {retained.format()}**",
        "",
        "A result counts only if every floor also holds: "
        + ", ".join(f"{name} {'met' if met else 'NOT met'}" for name, met in floors.items())
        + ".",
        "",
        "## Systems",
        "",
        score_table(scores),
        "",
        "## Student vs teacher, per field",
        "",
        "Exact McNemar on identical calls, with Holm correction across fields.",
        "",
        "| Field | Student accuracy [95% CI] | Teacher accuracy [95% CI] | p | Significant |",
        "|---|---|---|---:|---|",
    ]
    teacher_score = scores[1]
    for name, p_value in tests.items():
        lines.append(
            f"| {name} | {student_score.field_accuracy[name].format()} | "
            f"{teacher_score.field_accuracy[name].format()} | {p_value:.3f} | "
            f"{'yes' if survivors[name] else 'no'} |"
        )
    for title, mapping in (slices or {}).items():
        lines += [
            "",
            f"## Exploratory slice: {title}",
            "",
            slice_table(eval_set, outputs, mapping, min(resamples, 500)),
        ]
    summary: dict[str, object] = {
        "quality_retained": retained.point,
        "quality_retained_ci": [retained.low, retained.high],
        "floors": floors,
        "headline_counts": all(floors.values()),
        "scores": {s.name: {"q": s.q, "structured": s.structured, **s.parts} for s in scores},
        "judge_failures": failures,
        "dry_run": dry_run,
    }
    return "\n".join(lines) + "\n", summary

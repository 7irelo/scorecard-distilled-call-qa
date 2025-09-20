"""Per-field metrics against gold, the composite Q, and the pre-registered floors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from scorecard.eval.stats import Interval, bootstrap, mcnemar_exact, wilson
from scorecard.schema import (
    COMPLIANCE_CHECKS,
    SENTIMENT_ORDER,
    Scorecard,
    SentimentLabel,
    flatten_labels,
)

INVALID = "__invalid__"
SENTIMENT_SEGMENTS = ("sentiment.opening", "sentiment.middle", "sentiment.closing")
COMPLIANCE_FIELDS = tuple(f"compliance.{check}" for check in COMPLIANCE_CHECKS)
Q_WEIGHTS = {"compliance": 0.40, "escalation": 0.20, "sentiment": 0.15, "summary": 0.25}

FLOORS = {
    "fail_recall": 0.85,
    "escalation_recall": 0.85,
    "summary": 0.90,
    "evidence_precision": 0.90,
}


@dataclass(frozen=True)
class SystemOutput:
    """One system's output on the evaluation calls. A missing card counts as wrong, not skipped."""

    name: str
    cards: Mapping[str, Scorecard | None]
    faithfulness: Mapping[str, float] = field(default_factory=dict)
    parse_valid: Mapping[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalSet:
    ids: tuple[str, ...]
    gold: Mapping[str, Scorecard]
    weights: np.ndarray

    def gold_column(self, name: str) -> np.ndarray:
        return np.array([flatten_labels(self.gold[i])[name] for i in self.ids], dtype=object)

    def system_column(self, output: SystemOutput, name: str) -> np.ndarray:
        values = []
        for scenario_id in self.ids:
            card = output.cards.get(scenario_id)
            values.append(flatten_labels(card)[name] if card is not None else INVALID)
        return np.array(values, dtype=object)


def weighted_macro_f1(
    gold: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
    labels: Sequence[str] | None = None,
) -> float:
    """Macro-F1 over the classes in gold. Predictions outside them count only as misses."""
    classes = sorted(set(gold.tolist())) if labels is None else list(labels)
    scores = []
    for label in classes:
        scores.append(_f1(gold == label, predicted == label, weights))
    return float(np.mean(scores)) if scores else float("nan")


def _f1(truth: np.ndarray, guess: np.ndarray, weights: np.ndarray) -> float:
    tp = float(weights[truth & guess].sum())
    fp = float(weights[~truth & guess].sum())
    fn = float(weights[truth & ~guess].sum())
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def weighted_recall(truth: np.ndarray, guess: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights[truth].sum())
    return float(weights[truth & guess].sum()) / total if total else float("nan")


def weighted_precision(truth: np.ndarray, guess: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights[guess].sum())
    return float(weights[truth & guess].sum()) / total if total else float("nan")


@dataclass(frozen=True)
class Columns:
    """Pre-extracted arrays, so thousands of bootstrap resamples stay cheap."""

    gold: dict[str, np.ndarray]
    predicted: dict[str, np.ndarray]
    faithfulness: np.ndarray
    weights: np.ndarray

    @classmethod
    def build(cls, eval_set: EvalSet, output: SystemOutput) -> Columns:
        names = (*COMPLIANCE_FIELDS, *SENTIMENT_SEGMENTS, "escalation.flag")
        return cls(
            gold={name: eval_set.gold_column(name) for name in names},
            predicted={name: eval_set.system_column(output, name) for name in names},
            faithfulness=np.array(
                [output.faithfulness.get(i, 0.0) for i in eval_set.ids], dtype=float
            ),
            weights=eval_set.weights,
        )


def components(columns: Columns, index: np.ndarray | None = None) -> dict[str, float]:
    selected = np.arange(len(columns.weights)) if index is None else index
    weights = columns.weights[selected]

    def pair(name: str) -> tuple[np.ndarray, np.ndarray]:
        return columns.gold[name][selected], columns.predicted[name][selected]

    compliance = np.nanmean([weighted_macro_f1(*pair(name), weights) for name in COMPLIANCE_FIELDS])
    sentiment = np.nanmean([weighted_macro_f1(*pair(name), weights) for name in SENTIMENT_SEGMENTS])
    gold_flag, predicted_flag = pair("escalation.flag")
    escalation = _f1(gold_flag == "true", predicted_flag == "true", weights)
    total = float(weights.sum())
    summary = float((columns.faithfulness[selected] * weights).sum()) / total if total else 0.0
    return {
        "compliance": float(compliance),
        "escalation": escalation,
        "sentiment": float(sentiment),
        "summary": summary,
    }


def composite(parts: Mapping[str, float]) -> float:
    return sum(Q_WEIGHTS[name] * parts[name] for name in Q_WEIGHTS)


def fail_recall(columns: Columns) -> float:
    truth = np.concatenate([columns.gold[name] == "fail" for name in COMPLIANCE_FIELDS])
    guess = np.concatenate([columns.predicted[name] == "fail" for name in COMPLIANCE_FIELDS])
    weights = np.tile(columns.weights, len(COMPLIANCE_FIELDS))
    return weighted_recall(truth, guess, weights)


def sentiment_mae(columns: Columns) -> float:
    errors: list[float] = []
    rank = {label.value: SENTIMENT_ORDER[label] for label in SentimentLabel}
    worst = len(rank) - 1
    for name in SENTIMENT_SEGMENTS:
        for gold, predicted in zip(columns.gold[name], columns.predicted[name], strict=True):
            errors.append(abs(rank[gold] - rank[predicted]) if predicted in rank else worst)
    return float(np.mean(errors)) if errors else float("nan")


def evidence_precision(eval_set: EvalSet, output: SystemOutput) -> float:
    """Share of cited turns that the human also cited, for checks both called a pass."""
    cited = confirmed = 0
    for scenario_id in eval_set.ids:
        card = output.cards.get(scenario_id)
        if card is None:
            continue
        gold = eval_set.gold[scenario_id]
        for check in COMPLIANCE_CHECKS:
            mine, theirs = getattr(card.compliance, check), getattr(gold.compliance, check)
            if mine.status == theirs.status and mine.status.value == "pass":
                cited += len(mine.evidence_turns)
                confirmed += len(set(mine.evidence_turns) & set(theirs.evidence_turns))
    return confirmed / cited if cited else float("nan")


@dataclass(frozen=True)
class SystemScore:
    name: str
    parts: dict[str, float]
    q: float
    q_interval: Interval
    fail_recall: float
    escalation_recall: float
    escalation_precision: float
    sentiment_mae: float
    evidence_precision: float
    validity: Interval
    field_accuracy: dict[str, Interval]

    def floors_met(self) -> dict[str, bool]:
        return {
            "fail_recall": self.fail_recall >= FLOORS["fail_recall"],
            "escalation_recall": self.escalation_recall >= FLOORS["escalation_recall"],
            "summary": self.parts["summary"] >= FLOORS["summary"],
            "evidence_precision": self.evidence_precision >= FLOORS["evidence_precision"],
        }


def score_system(eval_set: EvalSet, output: SystemOutput, resamples: int = 2000) -> SystemScore:
    columns = Columns.build(eval_set, output)
    parts = components(columns)
    flag_gold = columns.gold["escalation.flag"] == "true"
    flag_guess = columns.predicted["escalation.flag"] == "true"
    q_interval = bootstrap(
        lambda index: composite(components(columns, index)), len(eval_set.ids), resamples
    )
    accuracy: dict[str, Interval] = {}
    for name in sorted(flatten_labels(next(iter(eval_set.gold.values())))):
        gold, predicted = eval_set.gold_column(name), eval_set.system_column(output, name)
        accuracy[name] = wilson(int((gold == predicted).sum()), len(gold))
    valid = sum(output.parse_valid.get(i, output.cards.get(i) is not None) for i in eval_set.ids)
    return SystemScore(
        name=output.name,
        parts=parts,
        q=composite(parts),
        q_interval=q_interval,
        fail_recall=fail_recall(columns),
        escalation_recall=weighted_recall(flag_gold, flag_guess, columns.weights),
        escalation_precision=weighted_precision(flag_gold, flag_guess, columns.weights),
        sentiment_mae=sentiment_mae(columns),
        evidence_precision=evidence_precision(eval_set, output),
        validity=wilson(valid, len(eval_set.ids)),
        field_accuracy=accuracy,
    )


def quality_retained(
    eval_set: EvalSet, student: SystemOutput, teacher: SystemOutput, resamples: int = 2000
) -> Interval:
    """Q_student / Q_teacher, with both systems resampled on the same calls."""
    student_columns = Columns.build(eval_set, student)
    teacher_columns = Columns.build(eval_set, teacher)

    def ratio(index: np.ndarray) -> float:
        denominator = composite(components(teacher_columns, index))
        if denominator == 0:
            return float("nan")
        return composite(components(student_columns, index)) / denominator

    return bootstrap(ratio, len(eval_set.ids), resamples)


def paired_field_tests(
    eval_set: EvalSet, first: SystemOutput, second: SystemOutput
) -> dict[str, float]:
    """Exact McNemar p-value for each categorical field, on identical calls."""
    result: dict[str, float] = {}
    for name in sorted(flatten_labels(next(iter(eval_set.gold.values())))):
        gold = eval_set.gold_column(name)
        a = eval_set.system_column(first, name) == gold
        b = eval_set.system_column(second, name) == gold
        result[name] = mcnemar_exact(int((a & ~b).sum()), int((~a & b).sum()))
    return result

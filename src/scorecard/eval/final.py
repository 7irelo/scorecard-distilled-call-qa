"""Phase 8: judge calibration and the single scored opening of the sealed test set."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from scorecard.baselines.llm_system import Prediction
from scorecard.data.models import ScenarioSpec, Transcript
from scorecard.eval.faithfulness import (
    INJECTED_KINDS,
    JudgeCalibration,
    inject_error,
    judge_blinded,
    judge_claim,
)
from scorecard.eval.fields import EvalSet, SystemOutput
from scorecard.eval.report import build_report
from scorecard.gold.sampling import open_test_set, post_hoc_openings
from scorecard.io import read_jsonl, write_jsonl
from scorecard.llm.client import LLMClient
from scorecard.schema import Scorecard


def prediction_row(prediction: Prediction) -> dict[str, Any]:
    row = asdict(prediction)
    row["card"] = prediction.card.model_dump(mode="json") if prediction.card else None
    return row


def load_predictions(path: Path) -> tuple[dict[str, Scorecard | None], dict[str, bool]]:
    cards: dict[str, Scorecard | None] = {}
    valid: dict[str, bool] = {}
    for row in read_jsonl(path):
        cards[row["scenario_id"]] = Scorecard.model_validate(row["card"]) if row["card"] else None
        valid[row["scenario_id"]] = bool(row.get("parse_valid", row["card"] is not None))
    return cards, valid


def make_calibration_set(
    rows: Iterable[dict[str, Any]],
    specs: dict[str, ScenarioSpec],
    n_clean: int = 100,
    n_injected: int = 30,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Summaries for humans to label claim by claim: some as written, some with known errors.

    Draws from non-test data, so the judge is never calibrated on the test set.
    """
    rng = random.Random(seed)
    pool = [row for row in rows if row.get("card")]
    rng.shuffle(pool)
    if len(pool) < n_clean + n_injected:
        raise ValueError(f"need {n_clean + n_injected} labelled calls, have {len(pool)}")
    output: list[dict[str, Any]] = []
    for row in pool[:n_clean]:
        output.append(
            {
                "scenario_id": row["scenario_id"],
                "summary": row["card"]["summary"],
                "injected": None,
            }
        )
    remaining = pool[n_clean:]
    index = 0
    made = 0
    while made < n_injected and index < len(remaining):
        row = remaining[index]
        index += 1
        spec = specs[row["scenario_id"]]
        entities = [spec.customer_name, spec.agent_name, spec.company, "Mr Smith", "Acme Mobile"]
        for kind in rng.sample(INJECTED_KINDS, len(INJECTED_KINDS)):
            changed = inject_error(row["card"]["summary"], kind, rng, entities)
            if changed is not None:
                output.append(
                    {"scenario_id": row["scenario_id"], "summary": changed, "injected": kind}
                )
                made += 1
                break
    return output


async def calibrate_judge(
    client: LLMClient,
    model: str,
    labelled_claims: list[dict[str, Any]],
    transcripts: dict[str, Transcript],
) -> JudgeCalibration:
    """Fit from rows with `scenario_id`, `claim` and the human's `supported` verdict."""
    verdicts = await asyncio.gather(
        *(
            judge_claim(client, model, transcripts[row["scenario_id"]], row["claim"])
            for row in labelled_claims
        )
    )
    pairs = [
        (bool(row["supported"]), bool(verdict))
        for row, verdict in zip(labelled_claims, verdicts, strict=True)
    ]
    return JudgeCalibration.fit(pairs)


def load_calibration(path: Path) -> JudgeCalibration:
    return JudgeCalibration(**json.loads(path.read_text(encoding="utf-8")))


async def _judge_all(
    client: LLMClient,
    model: str,
    systems: dict[str, dict[str, Scorecard | None]],
    transcripts: dict[str, Transcript],
    calibration: JudgeCalibration,
) -> dict[str, dict[str, float]]:
    jobs = [
        (name, scenario_id, transcripts[scenario_id], card.summary)
        for name, cards in systems.items()
        for scenario_id, card in cards.items()
        if card is not None
    ]
    results = await judge_blinded(client, model, jobs)
    scores: dict[str, dict[str, float]] = {name: {} for name in systems}
    for (name, scenario_id), result in results.items():
        scores[name][scenario_id] = calibration.correct(result.score)
    return scores


def run_final_evaluation(
    gold_path: Path,
    lock_path: Path,
    gold_items_path: Path,
    predictions_dir: Path,
    transcripts: dict[str, Transcript],
    specs: dict[str, ScenarioSpec],
    judge: LLMClient,
    judge_model: str,
    calibration: JudgeCalibration,
    output_dir: Path,
    post_hoc_reason: str | None = None,
) -> dict[str, object]:
    """Opens the test set. This is recorded, and a second opening must state a reason."""
    open_test_set(gold_path, lock_path, post_hoc_reason)
    gold = {
        row["scenario_id"]: Scorecard.model_validate(row["card"]) for row in read_jsonl(gold_path)
    }
    weights_by_id = {
        row["scenario_id"]: row["design_weight"] for row in read_jsonl(gold_items_path)
    }
    ids = tuple(sorted(gold))
    eval_set = EvalSet(ids, gold, np.array([weights_by_id[i] for i in ids], dtype=float))

    systems: dict[str, dict[str, Scorecard | None]] = {}
    validity: dict[str, dict[str, bool]] = {}
    for path in sorted(predictions_dir.glob("*.jsonl")):
        cards, valid = load_predictions(path)
        systems[path.stem] = {i: cards.get(i) for i in ids}
        validity[path.stem] = valid
    for required in ("student", "teacher"):
        if required not in systems:
            raise ValueError(f"missing predictions for {required} in {predictions_dir}")

    faithfulness = asyncio.run(_judge_all(judge, judge_model, systems, transcripts, calibration))
    outputs = {
        name: SystemOutput(name, cards, faithfulness[name], validity[name])
        for name, cards in systems.items()
    }
    others = [output for name, output in outputs.items() if name not in {"student", "teacher"}]
    slices = {
        "language mix": {i: specs[i].language_mix.value for i in ids},
        "call type": {i: specs[i].call_type.value for i in ids},
    }
    report, summary = build_report(
        eval_set,
        outputs["student"],
        outputs["teacher"],
        others,
        slices=slices,
        post_hoc=post_hoc_openings(lock_path),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_jsonl(
        output_dir / "faithfulness.jsonl",
        (
            {"system": name, "scenario_id": scenario_id, "score": score}
            for name, scores in faithfulness.items()
            for scenario_id, score in scores.items()
        ),
    )
    return summary

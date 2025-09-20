"""End-to-end pipeline run with stub models. Proves the plumbing, and only the plumbing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np

from scorecard.baselines.llm_system import predict_all
from scorecard.baselines.rules import predict_rules
from scorecard.baselines.tfidf import TfidfBaseline
from scorecard.data.asr_noise import NoiseProfile, apply_noise
from scorecard.data.models import ScenarioSpec, Transcript
from scorecard.data.scenarios import sample_scenarios
from scorecard.data.splits import assert_no_group_leakage, assign_splits, write_manifest
from scorecard.data.verify import render_and_verify_all
from scorecard.eval.faithfulness import judge_blinded
from scorecard.eval.fields import EvalSet, SystemOutput
from scorecard.eval.report import build_report
from scorecard.gold.sampling import lock_test_set, open_test_set, stratified_gold_sample
from scorecard.io import write_jsonl
from scorecard.llm.client import CostLedger, LLMClient, ResponseCache
from scorecard.llm.fake import FakeLLMBackend, corrupt_card
from scorecard.schema import Scorecard
from scorecard.teacher.label import TeacherLabel, label_all, qc_label, qc_report


def _simulate_human(spec: ScenarioSpec, transcript: Transcript) -> Scorecard:
    """Simulated gold: planted truth on top of a rules card. A real run uses `gold label`."""
    body = predict_rules(transcript).model_dump(mode="json")
    for check, status in spec.compliance.items():
        current = body["compliance"][check]
        turns = current["evidence_turns"] or [transcript.turns[0].id]
        body["compliance"][check] = {
            "status": status.value,
            "evidence_turns": turns if status.value == "pass" else [],
        }
    body["escalation"] = spec.escalation.model_dump(mode="json")
    body["call_type"] = spec.call_type.value
    return Scorecard.model_validate(body)


async def _pipeline(n: int, seed: int, out: Path) -> dict[str, Any]:
    specs = sample_scenarios(n, seed)
    backend = FakeLLMBackend({s.seed: s for s in specs})
    cache = ResponseCache(out / "llm_cache.sqlite")
    ledger = CostLedger(out / "ledger.jsonl")
    client = LLMClient(backend, cache=cache, ledger=ledger, concurrency=32)

    rendered = await render_and_verify_all(client, "stub-gen", client, "stub-verify", specs)
    kept = [
        (result.spec, apply_noise(result.transcript, NoiseProfile(), seed))
        for result, verification in rendered
        if result.transcript is not None and verification is not None
    ]
    adherence = sum(1 for _, v in rendered if v is not None and v.passed) / max(1, len(rendered))
    kept_specs = [spec for spec, _ in kept]
    by_id = {spec.scenario_id: spec for spec in kept_specs}
    transcripts = {t.scenario_id: t for _, t in kept}

    splits = assign_splits(
        kept_specs, {"train": 0.7, "silver_val": 0.1, "gold_pool": 0.2}, seed=seed
    )
    assert_no_group_leakage(kept_specs, splits)

    rubric = "Stub rubric for the dry run."
    labels = await label_all(client, "stub-teacher", rubric, list(transcripts.values()))
    label_map: dict[str, TeacherLabel] = {label.scenario_id: label for label in labels}
    qc_results = [
        qc_label(by_id[label.scenario_id], transcripts[label.scenario_id], label.card)
        for label in labels
        if label.card is not None
    ]
    qc_clean = {r.scenario_id: r.clean for r in qc_results}

    rows: dict[str, list[dict[str, Any]]] = {"train": [], "silver_val": [], "gold_pool": []}
    for scenario_id, split in splits.items():
        card = label_map[scenario_id].card
        rows[split].append(
            {
                "scenario_id": scenario_id,
                "transcript": transcripts[scenario_id].model_dump(mode="json"),
                "card": card.model_dump(mode="json") if card else None,
                "language_mix": by_id[scenario_id].language_mix.value,
                "qc_clean": qc_clean.get(scenario_id, False),
            }
        )
    paths = [out / f"{name}.jsonl" for name in rows]
    for path, name in zip(paths, rows, strict=True):
        write_jsonl(path, rows[name])
    write_manifest(out / "manifest.json", paths, {"dry_run": True, "seed": seed})

    pool = [by_id[r["scenario_id"]] for r in rows["gold_pool"]]
    n_test = max(10, len(pool) - max(5, len(pool) // 5))
    n_dev = len(pool) - n_test
    items = stratified_gold_sample(pool, n_dev=n_dev, n_test=n_test, seed=seed)
    test_items = [item for item in items if item.split == "test"]
    gold = {
        i.scenario_id: _simulate_human(by_id[i.scenario_id], transcripts[i.scenario_id])
        for i in test_items
    }
    gold_path = out / "gold_test.jsonl"
    write_jsonl(
        gold_path,
        ({"scenario_id": k, "card": v.model_dump(mode="json")} for k, v in gold.items()),
    )
    lock_test_set(gold_path, out / "gold_test.lock.json")
    open_test_set(gold_path, out / "gold_test.lock.json")

    ids = tuple(item.scenario_id for item in test_items)
    weights = np.array([item.design_weight for item in test_items], dtype=float)
    eval_set = EvalSet(ids=ids, gold=gold, weights=weights)
    eval_transcripts = [transcripts[i] for i in ids]

    student_cards: dict[str, Scorecard | None] = {
        i: corrupt_card(predict_rules(transcripts[i]), 0.08, f"student:{i}") for i in ids
    }
    teacher_cards: dict[str, Scorecard | None] = {i: label_map[i].card for i in ids}
    zero_shot = await predict_all(client, "stub-zero-shot", eval_transcripts)
    zero_shot_cards = {p.scenario_id: p.card for p in zero_shot}
    rules_cards: dict[str, Scorecard | None] = {i: predict_rules(transcripts[i]) for i in ids}

    train_rows = [r for r in rows["train"] if r["card"] is not None]
    tfidf = TfidfBaseline().fit(
        [Transcript.model_validate(r["transcript"]) for r in train_rows],
        [Scorecard.model_validate(r["card"]) for r in train_rows],
    )
    tfidf_cards: dict[str, Scorecard | None] = {i: tfidf.predict(transcripts[i]) for i in ids}

    systems = {
        "student": student_cards,
        "teacher": teacher_cards,
        "zero_shot": zero_shot_cards,
        "rules": rules_cards,
        "tfidf": tfidf_cards,
    }
    jobs = [
        (name, scenario_id, transcripts[scenario_id], card.summary)
        for name, cards in systems.items()
        for scenario_id, card in cards.items()
        if card is not None
    ]
    verdicts = await judge_blinded(client, "stub-judge", jobs, seed=seed)
    outputs = {
        name: SystemOutput(
            name,
            cards,
            faithfulness={
                scenario_id: verdicts[(name, scenario_id)].score
                for scenario_id in cards
                if (name, scenario_id) in verdicts
            },
        )
        for name, cards in systems.items()
    }
    slices = {
        "language mix": {i: by_id[i].language_mix.value for i in ids},
        "call type": {i: by_id[i].call_type.value for i in ids},
    }
    report, summary = build_report(
        eval_set,
        outputs["student"],
        outputs["teacher"],
        [outputs["zero_shot"], outputs["rules"], outputs["tfidf"]],
        slices=slices,
        dry_run=True,
        resamples=200,
    )
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    cache.close()
    result = {
        **summary,
        "scenarios": n,
        "kept": len(kept),
        "spec_adherence": adherence,
        "qc": qc_report(qc_results, labels),
        "gold_test": len(ids),
        "ledger_usd": ledger.total_usd(),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def run_dry_run(n: int, seed: int, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("gold_test.lock.json", "llm_cache.sqlite", "ledger.jsonl"):
        (out / stale).unlink(missing_ok=True)
    return asyncio.run(_pipeline(n, seed, out))

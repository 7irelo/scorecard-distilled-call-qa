"""Command line entry point. Paid steps refuse to run without their config and credentials."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import httpx
import numpy as np
import typer

from scorecard.baselines.llm_system import Prediction, predict_all
from scorecard.baselines.rules import predict_rules
from scorecard.baselines.tfidf import TfidfBaseline
from scorecard.data.asr_noise import NoiseProfile, apply_noise, calibrate
from scorecard.data.models import ScenarioSpec, Transcript
from scorecard.data.quality import (
    check_noise_accuracy,
    dataset_report,
    drop_near_duplicates,
    find_near_duplicates,
    score_review_sheet,
    write_review_sheet,
)
from scorecard.data.reality import (
    SimToRealGap,
    gap_markdown,
    load_reality_rows,
    reality_row,
    utterances_to_turns,
)
from scorecard.data.scenarios import sample_scenarios
from scorecard.data.splits import assert_no_group_leakage, assign_splits, write_manifest
from scorecard.data.verify import render_and_verify_all
from scorecard.eval.fields import Columns, EvalSet, SystemOutput, structured_score
from scorecard.eval.final import (
    calibrate_judge,
    load_calibration,
    load_predictions,
    make_calibration_set,
    prediction_row,
    run_final_evaluation,
    run_selection_evaluation,
)
from scorecard.eval.robustness import (
    bad_input_test,
    client_predictor,
    injection_test,
    noise_sweep,
    robustness_markdown,
)
from scorecard.gold.audit import (
    adjudication_queue,
    apply_decisions,
    audit_markdown,
    audit_teacher,
    label_claims,
    run_adjudication,
)
from scorecard.gold.labelling import inter_rater_agreement, run_blind_session
from scorecard.gold.sampling import (
    assert_not_sealed,
    lock_test_set,
    stratified_gold_sample,
    write_gold_items,
)
from scorecard.io import read_jsonl, write_jsonl
from scorecard.llm.config import build_client, load_llm_config
from scorecard.prompts import load_rubric, small_model_messages
from scorecard.schema import Scorecard, flatten_labels
from scorecard.serve.check import check_server
from scorecard.serve.cost_chart import render_chart
from scorecard.serve.cost_model import load_cost_inputs, report_markdown, volume_table
from scorecard.serve.loadgen import (
    GpuSampler,
    body_from_row,
    ledger_stats,
    load_markdown,
    probe_ttft,
    run_load,
    saturation_point,
)
from scorecard.teacher.label import (
    is_unstable,
    label_all,
    planted_labels,
    qc_label,
    qc_report,
    stability,
)
from scorecard.train.bakeoff import (
    compare_runs,
    freeze_run,
    load_bakeoff,
    load_run_summaries,
    token_report,
    write_bakeoff_configs,
)
from scorecard.train.config import TrainConfig, load_train_config, write_sweep

app = typer.Typer(no_args_is_help=True, help="Distilled call QA pipeline.")
data_app = typer.Typer(no_args_is_help=True, help="Phase 2: synthetic data.")
teacher_app = typer.Typer(no_args_is_help=True, help="Phase 3: teacher labels.")
gold_app = typer.Typer(no_args_is_help=True, help="Phase 4: human gold set.")
train_app = typer.Typer(no_args_is_help=True, help="Phases 6-7: student training.")
bench_app = typer.Typer(no_args_is_help=True, help="Phase 10: economics.")
predict_app = typer.Typer(no_args_is_help=True, help="Phases 5 and 8: system predictions.")
eval_app = typer.Typer(no_args_is_help=True, help="Phase 8: final evaluation.")
for name, sub in (
    ("data", data_app),
    ("teacher", teacher_app),
    ("gold", gold_app),
    ("train", train_app),
    ("bench", bench_app),
    ("predict", predict_app),
    ("eval", eval_app),
):
    app.add_typer(sub, name=name)

LLM_CONFIG = Path("configs/llm.yaml")
DataDir = Annotated[Path, typer.Option(help="Dataset directory.")]


def _specs(path: Path) -> list[ScenarioSpec]:
    return [ScenarioSpec.model_validate(row) for row in read_jsonl(path)]


def _transcripts(path: Path) -> list[Transcript]:
    return [Transcript.model_validate(row) for row in read_jsonl(path)]


def _source_transcripts(path: Path) -> list[Transcript]:
    """Reads dataset rows holding a `transcript`, or flat reality-check rows."""
    transcripts = []
    for row in read_jsonl(path):
        body = row.get("transcript") or {k: v for k, v in row.items() if k != "consent_id"}
        transcripts.append(Transcript.model_validate(body))
    return transcripts


def _examples(path: Path | None, limit: int = 5) -> tuple[tuple[Transcript, str], ...]:
    """Worked examples for few-shot prompts, from rows holding a `transcript` and a `card`."""
    if path is None:
        return ()
    return tuple(
        (
            Transcript.model_validate(row["transcript"]),
            Scorecard.model_validate(row["card"]).canonical_json(),
        )
        for row in list(read_jsonl(path))[:limit]
    )


@data_app.command("sample")
def data_sample(n: int = 5000, seed: int = 1, out: DataDir = Path("data")) -> None:
    """Sample scenario specs. Free, and fixes the planted truth."""
    specs = sample_scenarios(n, seed)
    write_jsonl(out / "specs.jsonl", (s.model_dump(mode="json") for s in specs))
    typer.echo(f"wrote {len(specs)} specs")


@data_app.command("generate")
def data_generate(limit: int | None = None, out: DataDir = Path("data")) -> None:
    """PAID. Render and verify transcripts. Use --limit 50 for the pilot."""
    config = load_llm_config(LLM_CONFIG)
    specs = _specs(out / "specs.jsonl")[:limit]
    generator, generator_model = build_client(config, "generator")
    verifier, verifier_model = build_client(config, "verifier")
    results = asyncio.run(
        render_and_verify_all(generator, generator_model, verifier, verifier_model, specs)
    )
    passed = [(r, v) for r, v in results if r.transcript and v and v.passed]
    write_jsonl(
        out / "clean.jsonl",
        (r.transcript.model_dump(mode="json") for r, _ in passed),  # type: ignore[union-attr]
    )
    write_jsonl(
        out / "rejected.jsonl",
        (
            {
                "scenario_id": r.spec.scenario_id,
                "failure": r.failure,
                "mismatches": list(v.mismatches) if v else [],
            }
            for r, v in results
            if not (r.transcript and v and v.passed)
        ),
    )
    adherence = len(passed) / max(1, len(results))
    typer.echo(f"kept {len(passed)}/{len(results)}, adherence {adherence:.1%}, gate is 95%")


@data_app.command("calibrate")
def data_calibrate(pairs: Path, out: DataDir = Path("data")) -> None:
    """Fit a noise profile from JSONL rows holding `reference` and `hypothesis`."""
    rows = [(r["reference"], r["hypothesis"]) for r in read_jsonl(pairs)]
    profile = calibrate(rows)
    (out / "noise_profile.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        f"sub={profile.substitution:.3f} del={profile.deletion:.3f} ins={profile.insertion:.3f}"
    )


@data_app.command("noise")
def data_noise(seed: int = 1, out: DataDir = Path("data")) -> None:
    """Apply the calibrated noise profile, or the uncalibrated default if there is none."""
    path = out / "noise_profile.json"
    if path.exists():
        profile = NoiseProfile.model_validate_json(path.read_text(encoding="utf-8"))
    else:
        typer.secho("No calibrated profile: using uncalibrated defaults.", fg="yellow")
        profile = NoiseProfile()
    noisy = (apply_noise(t, profile, seed) for t in _transcripts(out / "clean.jsonl"))
    count = write_jsonl(out / "transcripts.jsonl", (t.model_dump(mode="json") for t in noisy))
    typer.echo(f"wrote {count} noisy transcripts with profile {profile.name}")


@data_app.command("split")
def data_split(seed: int = 1, out: DataDir = Path("data")) -> None:
    """Group-aware splits, with a leakage check. Run after teacher labelling."""
    transcripts = {t.scenario_id: t for t in _transcripts(out / "transcripts.jsonl")}
    specs = [s for s in _specs(out / "specs.jsonl") if s.scenario_id in transcripts]
    labels = {r["scenario_id"]: r for r in read_jsonl(out / "teacher.jsonl")}
    repeat_path = out / "teacher_repeat.jsonl"
    repeats = (
        {r["scenario_id"]: r.get("card") for r in read_jsonl(repeat_path)}
        if repeat_path.exists()
        else {}
    )
    total = len(specs)
    fractions = {
        "gold_pool": 500 / total,
        "silver_val": 300 / total,
        "train": (total - 800) / total,
    }
    assignment = assign_splits(specs, fractions, seed)
    assert_no_group_leakage(specs, assignment)
    by_split: dict[str, list[dict[str, object]]] = {name: [] for name in fractions}
    for spec in specs:
        card = labels.get(spec.scenario_id, {}).get("card")
        issues_clean = False
        if card:
            parsed = Scorecard.model_validate(card)
            result = qc_label(spec, transcripts[spec.scenario_id], parsed)
            repeat = repeats.get(spec.scenario_id)
            second = Scorecard.model_validate(repeat) if repeat else None
            issues_clean = result.clean and not is_unstable(parsed, second)
        by_split[assignment[spec.scenario_id]].append(
            {
                "scenario_id": spec.scenario_id,
                "transcript": transcripts[spec.scenario_id].model_dump(mode="json"),
                "card": card,
                "language_mix": spec.language_mix.value,
                "qc_clean": issues_clean,
            }
        )
    paths = []
    for name, rows in by_split.items():
        path = out / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths.append(path)
        typer.echo(f"{name}: {len(rows)}")
    write_manifest(out / "manifest.json", paths, {"seed": seed})


@teacher_app.command("label")
def teacher_label(
    rubric: Path = Path("docs/RUBRIC.md"),
    limit: int | None = None,
    stability_n: int = 200,
    examples: Annotated[Path | None, typer.Option(help="Worked examples for the prompt.")] = None,
    role: str = "teacher",
    out: DataDir = Path("data"),
) -> None:
    """PAID. Label transcripts, plus a second pass on a subset to measure teacher stability."""
    config = load_llm_config(LLM_CONFIG)
    client, model = build_client(config, role)
    transcripts = _transcripts(out / "transcripts.jsonl")[:limit]
    shots = _examples(examples)
    rubric_text = load_rubric(rubric)
    labels = asyncio.run(label_all(client, model, rubric_text, transcripts, shots))
    write_jsonl(out / "teacher.jsonl", (label.to_row() for label in labels))
    repeat = asyncio.run(
        label_all(client, model, rubric_text, transcripts[:stability_n], shots, salt="stability")
    )
    write_jsonl(out / "teacher_repeat.jsonl", (label.to_row() for label in repeat))
    specs = {s.scenario_id: s for s in _specs(out / "specs.jsonl")}
    by_id = {t.scenario_id: t for t in transcripts}
    results = [
        qc_label(specs[label.scenario_id], by_id[label.scenario_id], label.card)
        for label in labels
        if label.card is not None
    ]
    report = {**qc_report(results, labels), "stability": stability(labels, repeat)}
    (out / "teacher_qc.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


@gold_app.command("sample")
def gold_sample(
    n_dev: int = 50, n_test: int = 200, seed: int = 1, out: DataDir = Path("data")
) -> None:
    """Choose gold calls from the gold pool, stratified with design weights."""
    pool_ids = {r["scenario_id"] for r in read_jsonl(out / "gold_pool.jsonl")}
    specs = [s for s in _specs(out / "specs.jsonl") if s.scenario_id in pool_ids]
    items = stratified_gold_sample(specs, n_dev, n_test, seed)
    write_gold_items(out / "gold_items.jsonl", items)
    typer.echo(f"chose {len(items)} gold calls")


@gold_app.command("label")
def gold_label(
    split: str = "test",
    rater: str = "owner",
    limit: Annotated[int | None, typer.Option(help="Second rater: about 40 calls.")] = None,
    reality: Annotated[bool, typer.Option(help="Label the reality-check set instead.")] = False,
    out: DataDir = Path("data"),
) -> None:
    """Interactive blind labelling. Reads transcripts only, and resumes where you stopped."""
    if reality:
        transcripts = load_reality_rows(read_jsonl(out / "reality.jsonl"))
        output = out / f"gold_reality_{rater}.jsonl"
    else:
        wanted = {
            r["scenario_id"] for r in read_jsonl(out / "gold_items.jsonl") if r["split"] == split
        }
        transcripts = [
            t for t in _transcripts(out / "transcripts.jsonl") if t.scenario_id in wanted
        ]
        output = out / f"gold_{split}_{rater}.jsonl"
    count = run_blind_session(transcripts[:limit], output, rater, input, typer.echo)
    typer.echo(f"labelled {count} calls into {output}")


@gold_app.command("agreement")
def gold_agreement(first: Path, second: Path) -> None:
    """Cohen's kappa per field, between two raters' files. The gate is 0.7."""

    def load(path: Path) -> dict[str, Scorecard]:
        return {r["scenario_id"]: Scorecard.model_validate(r["card"]) for r in read_jsonl(path)}

    typer.echo(json.dumps(inter_rater_agreement(load(first), load(second)), indent=2))


@gold_app.command("lock")
def gold_lock(labels: Path, lock: Path = Path("docs/gold_test.lock.json")) -> None:
    """Seal the finished test labels. Commit the lock file, never the labels."""
    typer.echo(f"locked {lock_test_set(labels, lock)}")


@train_app.command("run")
def train_run(config: Path, resume: bool = False) -> None:
    """Train one configuration. Needs the `train` extra, and a GPU unless quantization is none."""
    from scorecard.train.run import run_training

    summary = run_training(load_train_config(config), resume=resume)
    typer.echo(json.dumps({k: summary.get(k) for k in ("name", "wall_hours", "gpu_cost_usd")}))


@train_app.command("plan-sweep")
def train_plan_sweep(
    base: Path = Path("configs/train/baseline.yaml"),
    directory: Path = Path("configs/sweep/generated"),
    seeds: str = "42",
) -> None:
    """Write one config per ablation run, without running anything."""
    seed_values = tuple(int(seed) for seed in seeds.split(","))
    paths = write_sweep(load_train_config(base), directory, seed_values)
    typer.echo(f"wrote {len(paths)} run configs to {directory}")


@train_app.command("defaults")
def train_defaults() -> None:
    """Print the default training config."""
    typer.echo(TrainConfig().model_dump_json(indent=2))


@bench_app.command("costs")
def bench_costs(inputs: Path = Path("benchmarks/costs.example.yaml")) -> None:
    """Print the break-even table and save the volume table next to the inputs."""
    costs = load_cost_inputs(inputs)
    typer.echo(report_markdown(costs))
    output = inputs.with_suffix(".volumes.jsonl")
    write_jsonl(output, volume_table(costs))
    typer.echo(f"volume table: {output}")


@data_app.command("round-trip")
def data_round_trip(
    turns: int = 2000, language: str = "multi", out: DataDir = Path("data")
) -> None:
    """PAID. Send clean turns through TTS, a phone-line simulation and real STT."""
    from scorecard.data.speech import AzureSynthesizer, DeepgramTranscriber, round_trip_turn

    synth = AzureSynthesizer(os.environ["AZURE_SPEECH_KEY"], os.environ["AZURE_SPEECH_REGION"])
    stt = DeepgramTranscriber(os.environ["DEEPGRAM_API_KEY"], language=language)
    pending = [t.text for call in _transcripts(out / "clean.jsonl") for t in call.turns][:turns]
    rows = (
        {"reference": text, "hypothesis": round_trip_turn(text, synth, stt, seed)}
        for seed, text in enumerate(pending)
    )
    typer.echo(f"wrote {write_jsonl(out / 'asr_pairs.jsonl', rows)} pairs")


@predict_app.command("run")
def predict_run(
    role: Annotated[str, typer.Argument(help="Role name in configs/llm.yaml, such as student.")],
    name: Annotated[str, typer.Option(help="System name, which becomes the output file name.")],
    source: Annotated[Path, typer.Option(help="JSONL of rows with a `transcript`.")],
    constrained: bool = True,
    per_field: Annotated[bool, typer.Option(help="One call per field (ablation C).")] = False,
    examples: Annotated[Path | None, typer.Option(help="JSONL rows for few-shot prompts.")] = None,
    limit_examples: int = 3,
    out: Path = Path("runs/predictions"),
) -> None:
    """Run a model over transcripts. Closed providers are allowed here, on evaluation data only."""
    config = load_llm_config(LLM_CONFIG)
    client, model = build_client(config, role)
    transcripts = _source_transcripts(source)
    shots = _examples(examples, limit_examples)
    predictions = asyncio.run(
        predict_all(client, model, transcripts, shots, constrained, per_field)
    )
    write_jsonl(out / f"{name}.jsonl", (prediction_row(p) for p in predictions))
    valid = sum(p.parse_valid for p in predictions)
    typer.echo(f"{name}: {valid}/{len(predictions)} valid")


@predict_app.command("offline")
def predict_offline(
    system: Annotated[str, typer.Argument(help="rules or tfidf")],
    source: Annotated[Path, typer.Option(help="JSONL of rows with a `transcript`.")],
    train: Path = Path("data/train.jsonl"),
    out: Path = Path("runs/predictions"),
) -> None:
    """Run a baseline that needs no model server."""
    transcripts = _source_transcripts(source)
    if system == "rules":
        predict = predict_rules
    elif system == "tfidf":
        rows = [r for r in read_jsonl(train) if r.get("card")]
        predict = (
            TfidfBaseline()
            .fit(
                [Transcript.model_validate(r["transcript"]) for r in rows],
                [Scorecard.model_validate(r["card"]) for r in rows],
            )
            .predict
        )
    else:
        raise typer.BadParameter("choose rules or tfidf")
    rows_out = (
        prediction_row(Prediction(t.scenario_id, predict(t), True, "ok")) for t in transcripts
    )
    typer.echo(f"wrote {write_jsonl(out / f'{system}.jsonl', rows_out)} predictions")


@eval_app.command("calibration-set")
def eval_calibration_set(
    source: Path = Path("data/silver_val.jsonl"), out: DataDir = Path("data")
) -> None:
    """Choose summaries for humans to label, including 30 with injected errors."""
    specs = {s.scenario_id: s for s in _specs(out / "specs.jsonl")}
    rows = make_calibration_set(read_jsonl(source), specs)
    typer.echo(f"wrote {write_jsonl(out / 'judge_calibration_set.jsonl', rows)} summaries")


@eval_app.command("calibrate")
def eval_calibrate(
    claims: Annotated[Path, typer.Argument(help="Rows of scenario_id, claim, human `supported`.")],
    out: DataDir = Path("data"),
) -> None:
    """PAID. Measure judge sensitivity and specificity against human claim labels."""
    client, model = build_client(load_llm_config(LLM_CONFIG), "judge")
    transcripts = {t.scenario_id: t for t in _transcripts(out / "transcripts.jsonl")}
    calibration = asyncio.run(calibrate_judge(client, model, list(read_jsonl(claims)), transcripts))
    (out / "judge_calibration.json").write_text(
        json.dumps(asdict(calibration), indent=2), encoding="utf-8"
    )
    typer.echo(json.dumps(asdict(calibration)))


@eval_app.command("final")
def eval_final(
    gold: Path,
    lock: Path = Path("docs/gold_test.lock.json"),
    predictions: Path = Path("runs/predictions"),
    frozen: Path = Path("configs/train/frozen.yaml"),
    post_hoc_reason: str | None = None,
    out: DataDir = Path("data"),
    report_dir: Path = Path("runs/final"),
) -> None:
    """PAID, and ONE TIME. Opens the sealed test set and writes the report."""
    client, model = build_client(load_llm_config(LLM_CONFIG), "judge")
    summary = run_final_evaluation(
        gold,
        lock,
        out / "gold_items.jsonl",
        predictions,
        {t.scenario_id: t for t in _transcripts(out / "transcripts.jsonl")},
        {s.scenario_id: s for s in _specs(out / "specs.jsonl")},
        client,
        model,
        load_calibration(out / "judge_calibration.json"),
        report_dir,
        frozen,
        post_hoc_reason,
    )
    typer.echo(json.dumps(summary, indent=2))


@data_app.command("transcribe-reality")
def data_transcribe_reality(
    manifest: Annotated[
        Path, typer.Argument(help="Rows of file, call_id, consent_id, agent_speaker.")
    ],
    language: str = "multi",
    out: DataDir = Path("data"),
) -> None:
    """PAID. Transcribe consented recordings with real STT. They stay out of training."""
    from scorecard.data.speech import DeepgramTranscriber

    stt = DeepgramTranscriber(os.environ["DEEPGRAM_API_KEY"], language=language)
    rows = []
    for entry in read_jsonl(manifest):
        audio = (manifest.parent / entry["file"]).read_bytes()
        turns = utterances_to_turns(
            stt.transcribe_utterances(audio), int(entry.get("agent_speaker", 0))
        )
        rows.append(reality_row(entry["call_id"], entry.get("consent_id", ""), turns))
    typer.echo(f"wrote {write_jsonl(out / 'reality.jsonl', rows)} reality-check calls")


@data_app.command("import-reality")
def data_import_reality(source: Path, out: DataDir = Path("data")) -> None:
    """Validate an already transcribed reality-check file. Refuses calls without consent."""
    rows = list(read_jsonl(source))
    load_reality_rows(rows)
    typer.echo(f"wrote {write_jsonl(out / 'reality.jsonl', rows)} reality-check calls")


def _blind_cards(path: Path) -> dict[str, Scorecard]:
    return {r["scenario_id"]: Scorecard.model_validate(r["card"]) for r in read_jsonl(path)}


def _teacher_cards(path: Path) -> dict[str, Scorecard | None]:
    return {
        r["scenario_id"]: Scorecard.model_validate(r["card"]) if r.get("card") else None
        for r in read_jsonl(path)
    }


@gold_app.command("adjudicate")
def gold_adjudicate(
    blind: Annotated[Path, typer.Argument(help="Your finished blind labels.")],
    split: str = "test",
    out: DataDir = Path("data"),
) -> None:
    """Resolve disagreements with teacher and planted truth. Refuses an unfinished blind pass."""
    wanted = {r["scenario_id"] for r in read_jsonl(out / "gold_items.jsonl") if r["split"] == split}
    human = _blind_cards(blind)
    missing = wanted - set(human)
    if missing:
        raise typer.BadParameter(f"finish the blind pass first: {len(missing)} calls left")
    specs = {s.scenario_id: s for s in _specs(out / "specs.jsonl")}
    queue = adjudication_queue(
        human,
        _teacher_cards(out / "teacher.jsonl"),
        {i: planted_labels(specs[i]) for i in human},
    )
    transcripts = {t.scenario_id: t for t in _transcripts(out / "transcripts.jsonl")}
    output = out / f"adjudication_{split}.jsonl"
    count = run_adjudication(queue, transcripts, output, input, typer.echo)
    typer.echo(f"resolved {count} of {len(queue)} disagreements into {output}")


@gold_app.command("finalize")
def gold_finalize(blind: Path, split: str = "test", out: DataDir = Path("data")) -> None:
    """Apply adjudication decisions to produce final gold. Lock the result if it's the test set."""
    decisions = list(read_jsonl(out / f"adjudication_{split}.jsonl"))
    cards, issues = apply_decisions(_blind_cards(blind), decisions)
    if issues:
        typer.echo(json.dumps(issues, indent=2))
        raise typer.Exit(code=1)
    path = out / f"gold_{split}_final.jsonl"
    rows = ({"scenario_id": k, "card": v.model_dump(mode="json")} for k, v in cards.items())
    typer.echo(f"wrote {write_jsonl(path, rows)} final labels to {path}")


@gold_app.command("audit")
def gold_audit(
    gold: Annotated[Path, typer.Argument(help="Final, adjudicated gold labels.")],
    split: str = "test",
    claims: Annotated[
        Path | None, typer.Option(help="Your claim labels on teacher summaries.")
    ] = None,
    out: DataDir = Path("data"),
    report_dir: Path = Path("runs/audit"),
) -> None:
    """Report the teacher's error rate per field, with intervals and the adjudication record."""
    cards = _blind_cards(gold)
    items = {r["scenario_id"]: r for r in read_jsonl(out / "gold_items.jsonl")}
    specs = {s.scenario_id: s for s in _specs(out / "specs.jsonl")}
    audits = audit_teacher(
        cards,
        _teacher_cards(out / "teacher.jsonl"),
        {i: items[i]["design_weight"] for i in cards},
        {i: specs[i].language_mix.value for i in cards},
    )
    decision_path = out / f"adjudication_{split}.jsonl"
    decisions = list(read_jsonl(decision_path)) if decision_path.exists() else []
    claim_rows = list(read_jsonl(claims)) if claims else []
    report = audit_markdown(audits, decisions, claim_rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"TEACHER_AUDIT_{split}.md").write_text(report, encoding="utf-8")
    typer.echo(report)


@eval_app.command("label-claims")
def eval_label_claims(
    source: Annotated[Path, typer.Argument(help="Rows with scenario_id and a summary or card.")],
    output: Path = Path("data/judge_claims.jsonl"),
    split: Annotated[str | None, typer.Option(help="Only this gold split's calls.")] = None,
    out: DataDir = Path("data"),
) -> None:
    """Rate summaries sentence by sentence, for judge calibration or the teacher audit."""
    transcripts = {t.scenario_id: t for t in _transcripts(out / "transcripts.jsonl")}
    rows = [row for row in read_jsonl(source) if row.get("summary") or row.get("card")]
    if split is not None:
        wanted = {
            r["scenario_id"] for r in read_jsonl(out / "gold_items.jsonl") if r["split"] == split
        }
        rows = [row for row in rows if row["scenario_id"] in wanted]
    count = label_claims(rows, transcripts, output, input, typer.echo)
    typer.echo(f"labelled {count} claims into {output}")


@eval_app.command("select")
def eval_select(
    gold: Annotated[Path, typer.Argument(help="Dev gold labels. Sealed test labels are refused.")],
    focus: str = "student",
    lock: Path = Path("docs/gold_test.lock.json"),
    predictions: Path = Path("runs/predictions"),
    out: DataDir = Path("data"),
    report_dir: Path = Path("runs/select"),
) -> None:
    """PAID. Compare systems on dev gold, as often as needed, for choosing a configuration."""
    client, model = build_client(load_llm_config(LLM_CONFIG), "judge")
    summary = run_selection_evaluation(
        gold,
        lock,
        out / "gold_items.jsonl",
        predictions,
        {t.scenario_id: t for t in _transcripts(out / "transcripts.jsonl")},
        {s.scenario_id: s for s in _specs(out / "specs.jsonl")},
        client,
        model,
        load_calibration(out / "judge_calibration.json"),
        report_dir,
        focus,
    )
    typer.echo(json.dumps(summary, indent=2))


@eval_app.command("robustness")
def eval_robustness(
    role: Annotated[str, typer.Argument(help="Role name in configs/llm.yaml.")] = "student",
    gold: Annotated[Path, typer.Option(help="Gold labels for the noise sweep.")] = Path(
        "data/gold_dev_final.jsonl"
    ),
    calls: int = 100,
    lock: Path = Path("docs/gold_test.lock.json"),
    out: DataDir = Path("data"),
    report_dir: Path = Path("runs/robustness"),
) -> None:
    """Noise sweep, prompt injection and bad inputs. Sealed test labels wait for `eval final`."""
    assert_not_sealed(gold, lock)
    client, model = build_client(load_llm_config(LLM_CONFIG), role)
    predict = client_predictor(client, model)
    cards = _blind_cards(gold)
    items = {r["scenario_id"]: r["design_weight"] for r in read_jsonl(out / "gold_items.jsonl")}
    profile_path = out / "noise_profile.json"
    profile = (
        NoiseProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))
        if profile_path.exists()
        else NoiseProfile()
    )
    clean = _transcripts(out / "clean.jsonl")
    silver = [
        Transcript.model_validate(r["transcript"]) for r in read_jsonl(out / "silver_val.jsonl")
    ]

    async def run() -> str:
        noise = await noise_sweep(predict, clean, cards, items, profile)
        injection = await injection_test(predict, silver[:calls])
        bad = await bad_input_test(predict)
        return robustness_markdown(noise, injection, bad, role)

    report = asyncio.run(run())
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{role}.md").write_text(report, encoding="utf-8")
    typer.echo(report)


@eval_app.command("reality")
def eval_reality(
    gold: Annotated[Path, typer.Argument(help="Your labels for the reality-check calls.")],
    predictions: Path = Path("runs/predictions_reality"),
    final_summary: Path = Path("runs/final/summary.json"),
    report_dir: Path = Path("runs/final"),
) -> None:
    """Report the gap between synthetic test scores and consented recorded calls."""
    cards = _blind_cards(gold)
    ids = tuple(sorted(cards))
    eval_set = EvalSet(ids, cards, np.ones(len(ids)))
    synthetic = json.loads(final_summary.read_text(encoding="utf-8"))["scores"]
    gaps = []
    for path in sorted(predictions.glob("*.jsonl")):
        predicted, valid = load_predictions(path)
        columns = Columns.build(eval_set, SystemOutput(path.stem, predicted, parse_valid=valid))
        if path.stem in synthetic:
            gaps.append(
                SimToRealGap(
                    path.stem,
                    synthetic[path.stem]["structured"],
                    structured_score(columns),
                    len(ids),
                )
            )
    report = gap_markdown(gaps)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "SIM_TO_REAL.md").write_text(report, encoding="utf-8")
    typer.echo(report)


@predict_app.command("teacher")
def predict_teacher(
    out: DataDir = Path("data"), output: Path = Path("runs/predictions/teacher.jsonl")
) -> None:
    """Use the teacher's labels as a system to score."""
    rows = (
        {
            "scenario_id": r["scenario_id"],
            "card": r.get("card"),
            "parse_valid": r.get("card") is not None,
        }
        for r in read_jsonl(out / "teacher.jsonl")
    )
    typer.echo(f"wrote {write_jsonl(output, rows)} teacher predictions")


@predict_app.command("heads")
def predict_heads_cmd(
    config: Path,
    source: Annotated[Path, typer.Option(help="JSONL of rows with a `transcript`.")],
    out: Path = Path("runs/predictions"),
) -> None:
    """Predict with a trained heads-mode run. Needs the `train` extra."""
    from scorecard.train.heads import load_heads_model, predict_heads

    train_config = load_train_config(config)
    model, tokenizer = load_heads_model(train_config)
    rows = []
    for row in read_jsonl(source):
        transcript = Transcript.model_validate(row["transcript"])
        card, valid = predict_heads(model, tokenizer, transcript, train_config)
        outcome = "ok" if valid else "invalid"
        rows.append(prediction_row(Prediction(transcript.scenario_id, card, valid, outcome)))
    typer.echo(f"wrote {write_jsonl(out / f'{train_config.name}.jsonl', rows)} predictions")


@train_app.command("bakeoff-plan")
def train_bakeoff_plan(
    plan: Path = Path("configs/train/bakeoff.yaml"),
    base: Path = Path("configs/train/baseline.yaml"),
    directory: Path = Path("configs/sweep/bakeoff"),
) -> None:
    """Write a short run config per candidate. Refuses any unverified licence."""
    paths = write_bakeoff_configs(load_bakeoff(plan), load_train_config(base), directory)
    typer.echo(f"wrote {len(paths)} configs to {directory}")


@train_app.command("tokens")
def train_tokens(
    base_model: str,
    revision: str | None = None,
    source: Path = Path("data/silver_val.jsonl"),
    max_seq_length: int = 8192,
) -> None:
    """Measure tokens per word by language mix. Needs the `train` extra."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
    typer.echo(json.dumps(token_report(tokenizer, read_jsonl(source), max_seq_length), indent=2))


@train_app.command("overfit")
def train_overfit(config: Path, steps: int = 200) -> None:
    """Run this before any long run. Exits non-zero if 10 examples can't be learned."""
    from scorecard.train.run import overfit_config, run_training

    summary = run_training(overfit_config(load_train_config(config), steps))
    metrics = summary["final_metrics"]
    typer.echo(json.dumps(metrics, indent=2))
    if metrics["val/validity"] < 0.9 or metrics["val/field_agreement"] < 0.9:
        typer.secho("Overfit check FAILED: look for template or masking bugs.", fg="red")
        raise typer.Exit(code=1)


@train_app.command("inspect")
def train_inspect(config: Path) -> None:
    """Show one decoded example and its label mask. Needs the `train` extra."""
    from scorecard.train.run import inspect_example

    train_config = load_train_config(config)
    row = next(r for r in read_jsonl(train_config.train_path) if r.get("card"))
    typer.echo(json.dumps(inspect_example(train_config, row), indent=2, ensure_ascii=False))


@train_app.command("compare")
def train_compare(runs: Path = Path("runs"), prefix: str = "") -> None:
    """Rank finished runs by validation score."""
    typer.echo(compare_runs(load_run_summaries(runs, prefix)))


@train_app.command("freeze")
def train_freeze(run: Path, output: Path = Path("configs/train/frozen.yaml")) -> None:
    """Record the chosen run. `eval final` refuses to open the test set until this exists."""
    frozen = freeze_run(run, output)
    typer.echo(f"froze {frozen['run']} to {output}. Commit this before opening the test set.")


@bench_app.command("load")
def bench_load(
    url: str = "http://127.0.0.1:8000",
    rates: str = "0.25,0.5,1,2,4",
    duration: float = 120.0,
    source: Path = Path("data/silver_val.jsonl"),
    output: Path = Path("runs/bench/load.jsonl"),
) -> None:
    """Sweep arrival rates against a running API until it saturates."""
    bodies = [body_from_row(row) for row in read_jsonl(source)]
    key = os.environ["SCORECARD_API_KEY"]
    levels = [float(rate) for rate in rates.split(",")]

    async def run() -> tuple[list, dict[str, float] | None]:  # type: ignore[type-arg]
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=100)
        async with (
            httpx.AsyncClient(base_url=url, timeout=600, limits=limits) as client,
            GpuSampler() as sampler,
        ):
            results = [await run_load(client, bodies, rate, duration, key) for rate in levels]
        return results, sampler.summary()

    results, gpu = asyncio.run(run())
    write_jsonl(output, (result.to_row() for result in results))
    typer.echo(load_markdown(results, gpu))
    peak = saturation_point(results)
    if peak:
        typer.echo(f"Set peak_calls_per_hour: {peak.calls_per_hour:.0f} in your cost inputs.")


@bench_app.command("ttft")
def bench_ttft(
    url: str = "http://127.0.0.1:8001/v1",
    model: str = "student",
    calls: int = 20,
    source: Path = Path("data/silver_val.jsonl"),
) -> None:
    """Measure time to first token directly against vLLM, one call at a time."""
    rows = list(read_jsonl(source))[:calls]

    async def run() -> list[dict[str, float]]:
        async with httpx.AsyncClient(base_url=url.rstrip("/") + "/", timeout=600) as client:
            measured = []
            for row in rows:
                messages = small_model_messages(Transcript.model_validate(row["transcript"]))
                body = [{"role": role, "content": text} for role, text in messages]
                measured.append(await probe_ttft(client, model, body))
            return measured

    measured = asyncio.run(run())
    first = [m["ttft_s"] for m in measured]
    typer.echo(
        f"TTFT p50 {np.nanpercentile(first, 50):.2f}s, p95 {np.nanpercentile(first, 95):.2f}s"
    )


@bench_app.command("ledger")
def bench_ledger(path: Path = Path("runs/cost_ledger.jsonl"), model: str | None = None) -> None:
    """Measured tokens, latency and cost per model, to use as cost-model inputs."""
    typer.echo(json.dumps(ledger_stats(path, model), indent=2))


@bench_app.command("vllm-check")
def bench_vllm_check(
    url: str = "http://127.0.0.1:8001/v1",
    student: str = "student",
    base: str = "base",
) -> None:
    """Check a real vLLM server. Exits non-zero if adapters, schemas or overflow handling fail."""
    results = asyncio.run(
        check_server(url, student, base or None, os.environ.get("SCORECARD_VLLM_KEY", ""))
    )
    for result in results:
        typer.echo(f"{'PASS' if result.passed else 'FAIL'} {result.name}: {result.detail}")
    if not all(result.passed for result in results):
        raise typer.Exit(code=1)


@bench_app.command("chart")
def bench_chart(
    inputs: Path = Path("benchmarks/costs.example.yaml"),
    output: Path = Path("runs/bench/break_even.png"),
) -> None:
    """Draw the break-even chart. Needs the `report` extra."""
    typer.echo(f"wrote {render_chart(load_cost_inputs(inputs), output)}")


@data_app.command("quality")
def data_quality(threshold: float = 0.9, drop: bool = False, out: DataDir = Path("data")) -> None:
    """Report distributions and near-duplicates. With --drop, remove duplicates from clean.jsonl."""
    transcripts = _transcripts(out / "clean.jsonl")
    specs = {s.scenario_id: s for s in _specs(out / "specs.jsonl")}
    pairs = find_near_duplicates(transcripts, threshold)
    report = {**dataset_report(specs, transcripts), "near_duplicate_pairs": len(pairs)}
    (out / "quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))
    if drop and pairs:
        kept = drop_near_duplicates(transcripts, pairs)
        write_jsonl(out / "clean.jsonl", (t.model_dump(mode="json") for t in kept))
        typer.echo(f"kept {len(kept)} of {len(transcripts)} calls")


@data_app.command("review-sheet")
def data_review_sheet(language: str, n: int = 50, out: DataDir = Path("data")) -> None:
    """Export a CSV for a native reviewer. `language` is zu or af."""
    specs = {s.scenario_id: s for s in _specs(out / "specs.jsonl")}
    path = out / f"review_{language}.csv"
    count = write_review_sheet(path, specs, _transcripts(out / "clean.jsonl"), language, n)
    typer.echo(f"wrote {count} calls to {path}")


@data_app.command("review-score")
def data_review_score(sheet: Path) -> None:
    """Score a returned sheet. Exits non-zero below the 3.5 of 5 gate."""
    result = score_review_sheet(sheet)
    typer.echo(f"{result.reviewed} rated, {result.blank} blank, mean {result.mean:.2f}")
    if not result.passed:
        typer.secho("Below the naturalness gate: improve prompts before scaling.", fg="red")
        raise typer.Exit(code=1)


@data_app.command("check-noise")
def data_check_noise(
    target_wer: Annotated[float | None, typer.Option(help="WER measured from STT.")] = None,
    calls: int = 200,
    out: DataDir = Path("data"),
) -> None:
    """Check the injector's real WER against the measured target. Exits non-zero if it's off."""
    profile = NoiseProfile.model_validate_json(
        (out / "noise_profile.json").read_text(encoding="utf-8")
    )
    check = check_noise_accuracy(_transcripts(out / "clean.jsonl")[:calls], profile, target_wer)
    typer.echo(f"achieved {check.achieved_wer:.3f}, target {check.target_wer:.3f}")
    if not check.within_tolerance:
        raise typer.Exit(code=1)


@teacher_app.command("bakeoff")
def teacher_bakeoff(
    roles: Annotated[str, typer.Argument(help="Comma-separated roles from configs/llm.yaml.")],
    gold: Annotated[Path, typer.Option(help="Dev gold labels.")] = Path(
        "data/gold_dev_owner.jsonl"
    ),
    rubric: Path = Path("docs/RUBRIC.md"),
    examples: Path | None = None,
    out: DataDir = Path("data"),
) -> None:
    """PAID. Score candidate teachers on dev gold, reporting accuracy and cost per 1k calls."""
    config = load_llm_config(LLM_CONFIG)
    cards = _blind_cards(gold)
    transcripts = [t for t in _transcripts(out / "transcripts.jsonl") if t.scenario_id in cards]
    rubric_text = load_rubric(rubric)
    shots = _examples(examples)
    typer.echo("| Role | Model | Mean field accuracy | Failed labels | $/1k calls |")
    typer.echo("|---|---|---:|---:|---:|")
    for role in roles.split(","):
        client, model = build_client(config, role)
        ledger = config.ledger_path
        before = sum(1 for _ in read_jsonl(ledger)) if ledger.exists() else 0
        labels = asyncio.run(label_all(client, model, rubric_text, transcripts, shots))
        fields = correct = 0
        for label in labels:
            gold_flat = flatten_labels(cards[label.scenario_id])
            guess = flatten_labels(label.card) if label.card else {}
            fields += len(gold_flat)
            correct += sum(guess.get(name) == value for name, value in gold_flat.items())
        rows = list(read_jsonl(ledger))[before:]
        spent = sum(row["cost_usd"] for row in rows)
        failed = sum(label.card is None for label in labels)
        per_1k = spent / max(1, len(labels)) * 1000
        typer.echo(f"| {role} | {model} | {correct / fields:.3f} | {failed} | {per_1k:.2f} |")


@gold_app.command("examples")
def gold_examples(
    gold: Path = Path("data/gold_dev_final.jsonl"),
    n: int = 5,
    out: DataDir = Path("data"),
) -> None:
    """Build worked examples for the teacher prompt from dev gold. Test gold is never used."""
    transcripts = {t.scenario_id: t for t in _transcripts(out / "transcripts.jsonl")}
    rows = (
        {"transcript": transcripts[r["scenario_id"]].model_dump(mode="json"), "card": r["card"]}
        for r in list(read_jsonl(gold))[:n]
    )
    typer.echo(f"wrote {write_jsonl(out / 'teacher_examples.jsonl', rows)} examples")


@gold_app.command("rubric-check")
def gold_rubric_check(
    calls: Path = Path("examples/rubric_calls.jsonl"),
    rater: str = "second",
    out: DataDir = Path("data"),
) -> None:
    """Label the edge-case calls blind, then see every disagreement with the draft labels."""
    rows = list(read_jsonl(calls))
    transcripts = [Transcript.model_validate(row["transcript"]) for row in rows]
    output = out / f"rubric_{rater}.jsonl"
    run_blind_session(transcripts, output, rater, input, typer.echo)
    draft = {r["transcript"]["scenario_id"]: Scorecard.model_validate(r["card"]) for r in rows}
    labelled = _blind_cards(output)
    for scenario_id, card in labelled.items():
        mine, theirs = flatten_labels(draft[scenario_id]), flatten_labels(card)
        for name, value in mine.items():
            if theirs[name] != value:
                typer.echo(f"{scenario_id} {name}: draft={value} {rater}={theirs[name]}")
    typer.echo(json.dumps(inter_rater_agreement(draft, labelled), indent=2))


@train_app.command("merge")
def train_merge(base_model: str, adapter: Path, output: Path, revision: str | None = None) -> None:
    """Write merged weights, to compare with serving the adapter. Needs the `train` extra."""
    from scorecard.train.run import merge_adapter

    merge_adapter(base_model, adapter, output, revision)
    typer.echo(f"wrote merged model to {output}")


@app.command("demo")
def demo(
    call: Path = Path("examples/code_switched_call.json"),
    url: str = "http://127.0.0.1:8000",
    usd_per_1k: Annotated[float | None, typer.Option(help="Your measured cost per 1k.")] = None,
) -> None:
    """Send a code-switched call to a running API, and show the scorecard and latency."""
    body = json.loads(call.read_text(encoding="utf-8"))
    start = time.perf_counter()
    reply = httpx.post(
        f"{url}/v1/scorecards",
        json=body,
        headers={"x-api-key": os.environ["SCORECARD_API_KEY"]},
        timeout=300,
    )
    elapsed = time.perf_counter() - start
    if reply.status_code != 200:
        typer.secho(f"HTTP {reply.status_code}: {reply.text}", fg="red")
        raise typer.Exit(code=1)
    typer.echo(json.dumps(reply.json()["scorecard"], indent=2, ensure_ascii=False))
    typer.echo(f"Latency: {elapsed:.2f}s")
    typer.echo(
        f"Cost: ${usd_per_1k / 1000:.5f} per call"
        if usd_per_1k is not None
        else "Cost: pass --usd-per-1k with your measured figure"
    )


@app.command("dry-run")
def dry_run(n: int = 300, seed: int = 1, out: Path = Path("runs/dryrun")) -> None:
    """Run the whole pipeline on stub models. Free, and proves plumbing rather than quality."""
    from scorecard.dryrun import run_dry_run

    result = run_dry_run(n, seed, out)
    typer.echo(json.dumps(result, indent=2, default=str))
    typer.echo(f"report: {out / 'REPORT.md'}")


if __name__ == "__main__":
    app()

"""Command line entry point. Paid steps refuse to run without their config and credentials."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from scorecard.baselines.llm_system import Prediction, predict_all
from scorecard.baselines.rules import predict_rules
from scorecard.baselines.tfidf import TfidfBaseline
from scorecard.data.asr_noise import NoiseProfile, apply_noise, calibrate
from scorecard.data.models import ScenarioSpec, Transcript
from scorecard.data.scenarios import sample_scenarios
from scorecard.data.splits import assert_no_group_leakage, assign_splits, write_manifest
from scorecard.data.verify import render_and_verify_all
from scorecard.eval.final import (
    calibrate_judge,
    load_calibration,
    make_calibration_set,
    prediction_row,
    run_final_evaluation,
)
from scorecard.gold.labelling import inter_rater_agreement, run_blind_session
from scorecard.gold.sampling import (
    lock_test_set,
    stratified_gold_sample,
    write_gold_items,
)
from scorecard.io import read_jsonl, write_jsonl
from scorecard.llm.config import build_client, load_llm_config
from scorecard.prompts import load_rubric
from scorecard.schema import Scorecard
from scorecard.serve.cost_model import load_cost_inputs, report_markdown, volume_table
from scorecard.teacher.label import label_all, qc_label, qc_report, stability
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
            result = qc_label(spec, transcripts[spec.scenario_id], Scorecard.model_validate(card))
            issues_clean = result.clean
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
    out: DataDir = Path("data"),
) -> None:
    """PAID. Label transcripts, plus a second pass on a subset to measure teacher stability."""
    config = load_llm_config(LLM_CONFIG)
    client, model = build_client(config, "teacher")
    transcripts = _transcripts(out / "transcripts.jsonl")[:limit]
    rubric_text = load_rubric(rubric)
    labels = asyncio.run(label_all(client, model, rubric_text, transcripts))
    write_jsonl(out / "teacher.jsonl", (label.to_row() for label in labels))
    repeat = asyncio.run(
        label_all(client, model, rubric_text, transcripts[:stability_n], salt="stability")
    )
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
def gold_label(split: str = "test", rater: str = "owner", out: DataDir = Path("data")) -> None:
    """Interactive blind labelling. Reads transcripts only, and resumes where you stopped."""
    wanted = {r["scenario_id"] for r in read_jsonl(out / "gold_items.jsonl") if r["split"] == split}
    transcripts = [t for t in _transcripts(out / "transcripts.jsonl") if t.scenario_id in wanted]
    output = out / f"gold_{split}_{rater}.jsonl"
    count = run_blind_session(transcripts, output, rater, input, typer.echo)
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
    examples: Annotated[Path | None, typer.Option(help="JSONL rows for few-shot prompts.")] = None,
    limit_examples: int = 3,
    out: Path = Path("runs/predictions"),
) -> None:
    """Run a model over transcripts. Closed providers are allowed here, on evaluation data only."""
    config = load_llm_config(LLM_CONFIG)
    client, model = build_client(config, role)
    transcripts = [Transcript.model_validate(r["transcript"]) for r in read_jsonl(source)]
    shots: tuple[tuple[Transcript, str], ...] = ()
    if examples is not None:
        shots = tuple(
            (
                Transcript.model_validate(r["transcript"]),
                Scorecard.model_validate(r["card"]).canonical_json(),
            )
            for r in list(read_jsonl(examples))[:limit_examples]
        )
    predictions = asyncio.run(predict_all(client, model, transcripts, shots, constrained))
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
    transcripts = [Transcript.model_validate(r["transcript"]) for r in read_jsonl(source)]
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
        post_hoc_reason,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("dry-run")
def dry_run(n: int = 300, seed: int = 1, out: Path = Path("runs/dryrun")) -> None:
    """Run the whole pipeline on stub models. Free, and proves plumbing rather than quality."""
    from scorecard.dryrun import run_dry_run

    result = run_dry_run(n, seed, out)
    typer.echo(json.dumps(result, indent=2, default=str))
    typer.echo(f"report: {out / 'REPORT.md'}")


if __name__ == "__main__":
    app()

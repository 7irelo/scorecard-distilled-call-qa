"""Tests for dataset quality checks, review sheets, cost additions and the rubric call file."""

import csv
import json

import pytest
from typer.testing import CliRunner

from scorecard.cli import app as cli_app
from scorecard.data.asr_noise import NoiseProfile
from scorecard.data.models import Speaker, Transcript, Turn
from scorecard.data.quality import (
    check_noise_accuracy,
    dataset_report,
    drop_near_duplicates,
    find_near_duplicates,
    score_review_sheet,
    write_review_sheet,
)
from scorecard.data.scenarios import sample_scenarios
from scorecard.io import read_jsonl
from scorecard.schema import Scorecard
from scorecard.serve.app import Settings, create_app
from scorecard.serve.cost_model import (
    ApiOption,
    CostInputs,
    Fixed,
    SelfHosted,
    length_sensitivity,
    quality_adjusted_rows,
    report_markdown,
    serverless_monthly_usd,
    volume_table,
)
from scorecard.serve.inference import FakeBackend
from tests.fixtures.rubric_calls import FIXTURES

runner = CliRunner()


def _call(name, text):
    return Transcript(scenario_id=name, turns=(Turn(id=1, speaker=Speaker.AGENT, text=text),))


def test_near_duplicates_are_found_and_only_one_of_each_pair_is_dropped():
    calls = [
        _call("a", "my bill is wrong and i want a refund for this month please"),
        _call("b", "my bill is wrong and i want a refund for this month please"),
        _call("c", "the fibre line is completely down since last tuesday evening"),
    ]
    pairs = find_near_duplicates(calls)
    assert [(first, second) for first, second, _ in pairs] == [("a", "b")]
    assert [t.scenario_id for t in drop_near_duplicates(calls, pairs)] == ["a", "c"]
    assert find_near_duplicates(calls[:1]) == []


def _with_specs(n=40):
    specs = sample_scenarios(n, seed=21)
    transcripts = [_call(s.scenario_id, f"sawubona dankie call {i}") for i, s in enumerate(specs)]
    return {s.scenario_id: s for s in specs}, transcripts


def test_dataset_report_covers_distributions():
    specs, transcripts = _with_specs()
    report = dataset_report(specs, transcripts)
    assert report["calls"] == 40
    assert sum(report["call_types"].values()) == 40
    assert report["turns"]["max"] == 1
    assert report["language_mix"]


def test_review_sheet_has_only_the_right_languages_and_no_labels(tmp_path):
    specs, transcripts = _with_specs(200)
    path = tmp_path / "review.csv"
    assert write_review_sheet(path, specs, transcripts, "zu", n=10) == 10
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["language_mix"] for row in rows} <= {"en_zu", "en_zu_af"}
    assert "compliance" not in rows[0]
    assert rows[0]["naturalness_1_to_5"] == ""


def _sheet(path, ratings):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario_id", "naturalness_1_to_5"])
        writer.writeheader()
        for index, rating in enumerate(ratings):
            writer.writerow({"scenario_id": str(index), "naturalness_1_to_5": rating})
    return path


def test_review_scoring_applies_the_gate_and_ignores_blanks(tmp_path):
    passing = score_review_sheet(_sheet(tmp_path / "a.csv", ["4", "4", ""]))
    assert (passing.reviewed, passing.blank, passing.mean, passing.passed) == (2, 1, 4.0, True)
    assert not score_review_sheet(_sheet(tmp_path / "b.csv", ["3", "3"])).passed
    assert not score_review_sheet(_sheet(tmp_path / "c.csv", [""])).passed
    with pytest.raises(ValueError):
        score_review_sheet(_sheet(tmp_path / "d.csv", ["9"]))


def test_noise_check_compares_achieved_wer_with_target():
    text = " ".join(["the bill is wrong and i want help now please"] * 80)
    calls = [_call(str(i), text) for i in range(3)]
    profile = NoiseProfile(
        substitution=0.10,
        deletion=0.05,
        insertion=0.0,
        filler=0.0,
        interruption=0.0,
        number_garble=0.0,
        code_switch_multiplier=1.0,
    )
    assert check_noise_accuracy(calls, profile).within_tolerance
    assert not check_noise_accuracy(calls, profile, target_wer=0.60).within_tolerance


def _gpu(**extra):
    return SelfHosted(name="G", gpu_usd_per_hour=0.59, peak_calls_per_hour=1800, **extra)


def _api(**extra):
    return ApiOption(
        name="A",
        input_per_mtok=4,
        output_per_mtok=20,
        batch_discount=0.5,
        input_tokens=6000,
        output_tokens=500,
        **extra,
    )


def test_serverless_charges_active_and_cold_start_time():
    option = _gpu(serverless_usd_per_hour=0.75, cold_start_seconds=60, cold_starts_per_month=60)
    # 1800 calls is one active hour, and 60 one-minute cold starts add another.
    assert serverless_monthly_usd(option, 1800, Fixed()) == pytest.approx(1.5)
    assert serverless_monthly_usd(_gpu(), 1800, Fixed()) is None
    rows = volume_table(CostInputs(self_hosted=[option], apis=[], volumes_per_month=[1800]))
    assert "G serverless" in {row["option"] for row in rows}


def test_length_sensitivity_raises_cost_with_length():
    rows = length_sensitivity(CostInputs(self_hosted=[_gpu()], apis=[_api()]), (1.0, 2.0))
    gpu = [r["usd_per_1k"] for r in rows if r["option"] == "G"]
    api = [r["usd_per_1k"] for r in rows if r["option"] == "A batch"]
    assert gpu[1] == pytest.approx(2 * gpu[0], rel=0.01)
    assert api[1] > api[0]


def test_quality_adjusted_rows_only_use_measured_quality():
    assert quality_adjusted_rows(CostInputs(self_hosted=[_gpu()], apis=[_api()])) == []
    rows = quality_adjusted_rows(
        CostInputs(self_hosted=[_gpu(quality_retained=0.5)], apis=[_api()]), utilisation=1.0
    )
    assert len(rows) == 1
    assert rows[0]["quality_adjusted_usd_per_1k"] == pytest.approx(0.6556, abs=0.001)
    report = report_markdown(CostInputs(self_hosted=[_gpu(quality_retained=0.5)], apis=[_api()]))
    assert "Quality-adjusted" in report
    assert "Length factor" in report


def test_repo_rubric_call_file_matches_the_fixtures():
    rows = list(read_jsonl(__import__("pathlib").Path("examples/rubric_calls.jsonl")))
    assert len(rows) == len(FIXTURES)
    for row, (transcript, card) in zip(rows, FIXTURES, strict=True):
        assert Transcript.model_validate(row["transcript"]) == transcript
        assert Scorecard.model_validate(row["card"]) == card


def test_demo_prints_a_scorecard_and_latency(monkeypatch):
    from fastapi.testclient import TestClient

    client = TestClient(create_app(Settings(api_key="k"), FakeBackend()))
    monkeypatch.setenv("SCORECARD_API_KEY", "k")
    monkeypatch.setattr(
        "scorecard.cli.httpx.post",
        lambda url, **kwargs: client.post(
            "/v1/scorecards", **{k: kwargs[k] for k in ("json", "headers")}
        ),
    )
    result = runner.invoke(cli_app, ["demo", "--usd-per-1k", "1.0"])
    assert result.exit_code == 0, result.output
    assert "Latency:" in result.output
    assert "$0.00100 per call" in result.output
    assert json.loads(result.output[: result.output.index("Latency:")])["schema_version"] == "1.0"

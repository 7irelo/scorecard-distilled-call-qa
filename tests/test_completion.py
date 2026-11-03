"""Tests for the teacher audit, robustness, reality-check set, bake-off and fixed bugs."""

import asyncio
import json
import math

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from scorecard.baselines.llm_system import predict_per_field
from scorecard.baselines.rules import predict_rules
from scorecard.cli import app as cli_app
from scorecard.data.models import Speaker, Transcript, Turn
from scorecard.data.reality import (
    ConsentMissing,
    RealityLeak,
    SimToRealGap,
    assert_no_reality_rows,
    gap_markdown,
    load_reality_rows,
    reality_row,
    utterances_to_turns,
)
from scorecard.data.speech import DeepgramTranscriber
from scorecard.eval.fields import Columns, EvalSet, SystemOutput, components, structured_score
from scorecard.eval.final import (
    run_final_evaluation,
    run_selection_evaluation,
    sample_errors,
)
from scorecard.eval.report import build_report
from scorecard.eval.robustness import (
    ATTACKS,
    BAD_INPUTS,
    bad_input_test,
    injection_test,
    noise_sweep,
    robustness_markdown,
    with_attack,
)
from scorecard.gold.audit import (
    adjudication_queue,
    apply_decisions,
    audit_markdown,
    audit_teacher,
    label_claims,
    run_adjudication,
    split_claims_simple,
    verdict_counts,
)
from scorecard.gold.sampling import TestSetSealed, lock_test_set
from scorecard.io import read_jsonl, write_jsonl
from scorecard.llm.backends import AnthropicBackend, OpenAICompatibleBackend
from scorecard.llm.client import LLMClient, ResponseCache, cache_key
from scorecard.llm.config import ModelConfig, build_backend, load_llm_config
from scorecard.llm.fake import FakeLLMBackend
from scorecard.llm.types import LLMRequest, LLMResponse, Outcome, ProviderClass, Purpose
from scorecard.schema import Scorecard, flatten_labels
from scorecard.serve.app import Settings, create_app
from scorecard.serve.cost_chart import chart_series
from scorecard.serve.cost_model import ApiOption, CostInputs, SelfHosted
from scorecard.serve.inference import (
    BackendUnavailable,
    FakeBackend,
    TranscriptTooLong,
    VLLMBackend,
)
from scorecard.serve.loadgen import (
    GpuSampler,
    LoadResult,
    body_from_row,
    ledger_stats,
    load_markdown,
    probe_ttft,
    saturation_point,
)
from scorecard.teacher.label import is_unstable
from scorecard.train.bakeoff import (
    BakeoffPlan,
    Candidate,
    LicenceNotVerified,
    compare_runs,
    freeze_run,
    load_bakeoff,
    load_run_summaries,
    token_report,
    write_bakeoff_configs,
)
from scorecard.train.config import TrainConfig
from scorecard.train.format import PER_FIELD_NAMES, per_field_targets, select_training_rows
from tests.fixtures.rubric_calls import FIXTURES
from tests.test_train_format import CharTokenizer

CARD = FIXTURES[0][1]
OTHER = FIXTURES[1][1]


def _eval_set():
    ids = tuple(t.scenario_id for t, _ in FIXTURES)
    return EvalSet(ids, {t.scenario_id: c for t, c in FIXTURES}, np.ones(len(ids)))


# Bug: thinking mode must match between training, evaluation and serving.
def test_repo_config_disables_thinking_for_served_models():
    config = load_llm_config(__import__("pathlib").Path("configs/llm.yaml"))
    for role in ("student", "zero_shot"):
        extra = config.model_for(role).extra_body
        assert extra["chat_template_kwargs"] == {"enable_thinking": False}


def test_openai_backend_sends_extra_body():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    backend = OpenAICompatibleBackend(
        "http://x/v1",
        transport=httpx.MockTransport(handler),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    request = LLMRequest("m", (("user", "hi"),), Purpose.BASELINE)
    asyncio.run(backend.complete(request))
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


def test_build_backend_passes_extra_body_and_namespace_splits_cache():
    model = ModelConfig(
        model_id="m",
        family="f",
        provider_class=ProviderClass.OPEN_WEIGHT,
        base_url="http://x/v1",
        extra_body={"a": 1},
    )
    assert build_backend(model)._extra_body == {"a": 1}
    request = LLMRequest("m", (("user", "hi"),), Purpose.BASELINE)
    assert cache_key(request) != cache_key(request, "thinking-off")


# Bug: per-field runs must predict with per-field prompts.
class _FieldBackend:
    provider_class = ProviderClass.OPEN_WEIGHT

    def __init__(self, card, broken=None):
        self.targets = per_field_targets(card)
        self.broken = broken
        self.prompts = []

    async def complete(self, request):
        system = request.messages[0][1]
        self.prompts.append(system)
        field = system.rsplit("as JSON: ", 1)[1].rstrip(".")
        text = "{bad" if field == self.broken else self.targets[field]
        return LLMResponse(text, Outcome.OK, input_tokens=10, output_tokens=2)


def test_per_field_prediction_rebuilds_the_card_with_one_call_per_field():
    backend = _FieldBackend(CARD)
    result = asyncio.run(predict_per_field(LLMClient(backend), "m", FIXTURES[0][0]))
    assert result.card == CARD
    assert result.parse_valid
    assert len(backend.prompts) == len(PER_FIELD_NAMES) == 9
    assert result.input_tokens == 90


def test_per_field_prediction_marks_one_bad_field_invalid():
    backend = _FieldBackend(CARD, broken="sentiment")
    result = asyncio.run(predict_per_field(LLMClient(backend), "m", FIXTURES[0][0]))
    assert result.card is None
    assert result.outcome == "invalid"


# Bug: judge failures are gaps, not low scores, and calibration applies to the mean.
def test_judge_failures_are_excluded_and_counted():
    eval_set = _eval_set()
    faith = dict.fromkeys(eval_set.ids, 1.0)
    faith[eval_set.ids[0]] = float("nan")
    columns = Columns.build(eval_set, SystemOutput("s", eval_set.gold, faith))
    assert components(columns)["summary"] == pytest.approx(1.0)
    report, summary = build_report(
        eval_set,
        SystemOutput("s", eval_set.gold, faith),
        SystemOutput("t", eval_set.gold, dict.fromkeys(eval_set.ids, 1.0)),
        [],
        resamples=20,
    )
    assert "Judge failures" in report
    assert summary["judge_failures"] == {"s": 1}


def test_missing_card_still_scores_zero_on_summary():
    eval_set = _eval_set()
    faith = {i: 1.0 for i in eval_set.ids[1:]}
    cards = {i: eval_set.gold[i] for i in eval_set.ids[1:]}
    columns = Columns.build(eval_set, SystemOutput("s", cards, faith))
    assert components(columns)["summary"] == pytest.approx(0.9)


def test_summary_transform_applies_to_the_mean():
    eval_set = _eval_set()
    faith = dict.fromkeys(eval_set.ids, 0.5)
    output = SystemOutput("s", eval_set.gold, faith, summary_transform=lambda value: value * 2)
    assert components(Columns.build(eval_set, output))["summary"] == pytest.approx(1.0)


def test_structured_score_ignores_summary_and_is_one_when_perfect():
    eval_set = _eval_set()
    columns = Columns.build(eval_set, SystemOutput("s", eval_set.gold))
    assert structured_score(columns) == pytest.approx(1.0)


def test_sample_errors_lists_only_real_mistakes():
    eval_set = _eval_set()
    transcripts = {t.scenario_id: t for t, _ in FIXTURES}
    assert sample_errors(eval_set, SystemOutput("s", eval_set.gold), transcripts) == []
    errors = sample_errors(eval_set, SystemOutput("s", {}), transcripts, n=5)
    assert len(errors) == 5
    assert errors[0]["predicted"] is None
    assert errors[0]["category"] == ""


# Bug: the clean filter must use the relabelling pass.
def test_is_unstable():
    assert not is_unstable(CARD, CARD)
    assert is_unstable(CARD, OTHER)
    assert not is_unstable(CARD, None)


# Teacher audit.
def test_adjudication_queue_and_session(tmp_path):
    transcript = FIXTURES[0][0]
    name = transcript.scenario_id
    queue = adjudication_queue({name: CARD}, {name: OTHER}, {name: {"call_type": "outage"}})
    assert "call_type" in {item.field for item in queue}
    output = tmp_path / "decisions.jsonl"
    item = next(i for i in queue if i.field == "call_type")
    answers = iter(["nonsense", "teacher", "teacher was right"])
    said = []
    assert (
        run_adjudication([item], {name: transcript}, output, lambda _: next(answers), said.append)
        == 1
    )
    assert transcript.render() in said[0]
    row = next(read_jsonl(output))
    assert row["verdict"] == "teacher"
    assert row["value"] == OTHER.call_type.value
    assert run_adjudication([item], {name: transcript}, output, lambda _: "", said.append) == 0


def test_apply_decisions_changes_only_the_decided_field():
    name = "x"
    cards, issues = apply_decisions(
        {name: CARD}, [{"scenario_id": name, "field": "call_type", "value": "outage"}]
    )
    assert cards[name].call_type.value == "outage"
    assert cards[name].summary == CARD.summary
    assert issues == {}


def test_apply_decisions_clears_evidence_on_fail_and_uses_it_on_pass():
    name = "x"
    decisions = [
        {
            "scenario_id": name,
            "field": "compliance.agent_introduced",
            "value": "fail",
            "evidence_turns": None,
        },
        {
            "scenario_id": name,
            "field": "compliance.identity_verified",
            "value": "pass",
            "evidence_turns": [3, 4],
        },
    ]
    cards, _ = apply_decisions({name: CARD}, decisions)
    assert cards[name].compliance.agent_introduced.evidence_turns == []
    assert cards[name].compliance.identity_verified.evidence_turns == [3, 4]


def test_apply_decisions_reports_incoherent_cards_rather_than_repairing_them():
    decisions = [{"scenario_id": "x", "field": "sentiment.closing", "value": "very_negative"}]
    _, issues = apply_decisions({"x": CARD}, decisions)
    assert issues == {"x": ["trajectory_contradicts_segments"]}


def test_audit_counts_missing_teacher_labels_as_errors_and_splits_by_language():
    gold = {"a": CARD, "b": CARD, "c": CARD, "d": CARD}
    teacher = {"a": CARD, "b": OTHER, "c": None, "d": CARD}
    weights = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    mix = {"a": "en", "b": "en", "c": "en_zu", "d": "en_zu"}
    audits = {a.field: a for a in audit_teacher(gold, teacher, weights, mix)}
    call_type = audits["call_type"]
    assert call_type.errors == 2
    assert call_type.n == 4
    assert call_type.error_rate.point == pytest.approx(0.5)
    assert call_type.english_error == pytest.approx(0.5)
    assert call_type.code_switched_error == pytest.approx(0.5)


def test_audit_weighted_estimate_follows_weights():
    gold = {"a": CARD, "b": CARD}
    audits = {
        a.field: a
        for a in audit_teacher(
            gold, {"a": CARD, "b": OTHER}, {"a": 9.0, "b": 1.0}, {"a": "en", "b": "en"}
        )
    }
    assert audits["call_type"].weighted_error_rate.point == pytest.approx(0.1)
    assert audits["call_type"].error_rate.point == pytest.approx(0.5)


def test_audit_report_includes_intervals_verdicts_and_claims():
    audits = audit_teacher({"a": CARD}, {"a": OTHER}, {"a": 1.0}, {"a": "en"})
    decisions = [
        {"field": "call_type", "verdict": "teacher"},
        {"field": "call_type", "verdict": "human"},
    ]
    assert verdict_counts(decisions) == {"call_type": {"teacher": 1, "human": 1}}
    report = audit_markdown(audits, decisions, [{"supported": True}, {"supported": False}])
    assert "# Teacher audit" in report
    assert "call_type" in report
    assert "0.500" in report
    assert "the teacher was right in 1" in report
    assert not report.startswith("> **DRY RUN")
    assert audit_markdown(audits, dry_run=True).startswith("> **DRY RUN")


def test_audit_needs_gold():
    with pytest.raises(ValueError):
        audit_teacher({}, {}, {}, {})


def test_claim_labelling_is_resumable(tmp_path):
    transcript = FIXTURES[0][0]
    rows = [
        {"scenario_id": transcript.scenario_id, "summary": "One thing. Another thing!"},
    ]
    output = tmp_path / "claims.jsonl"
    answers = iter(["?", "y", "n"])
    count = label_claims(
        rows, {transcript.scenario_id: transcript}, output, lambda _: next(answers), lambda _: None
    )
    assert count == 2
    assert [r["supported"] for r in read_jsonl(output)] == [True, False]
    assert (
        label_claims(rows, {transcript.scenario_id: transcript}, output, lambda _: "", print) == 0
    )
    assert split_claims_simple("A. B? C!") == ["A.", "B?", "C!"]


# Evaluation flow.
class _NoJudge:
    provider_class = ProviderClass.OPEN_WEIGHT

    async def complete(self, request):
        if "<claim>" in request.messages[-1][1]:
            return LLMResponse(json.dumps({"supported": True}), Outcome.OK)
        return LLMResponse(json.dumps({"claims": ["c"]}), Outcome.OK)


def _evaluation_files(tmp_path):
    from scorecard.data.scenarios import sample_scenarios
    from scorecard.eval.faithfulness import JudgeCalibration

    transcripts = {t.scenario_id: t for t, _ in FIXTURES}
    specs_list = sample_scenarios(len(FIXTURES), seed=1)
    specs = {
        name: spec.model_copy(update={"scenario_id": name})
        for name, spec in zip(transcripts, specs_list, strict=True)
    }
    gold = tmp_path / "gold.jsonl"
    write_jsonl(
        gold,
        ({"scenario_id": t.scenario_id, "card": c.model_dump(mode="json")} for t, c in FIXTURES),
    )
    items = tmp_path / "items.jsonl"
    write_jsonl(items, ({"scenario_id": n, "design_weight": 1.0} for n in transcripts))
    predictions = tmp_path / "predictions"
    for system in ("student", "teacher"):
        write_jsonl(
            predictions / f"{system}.jsonl",
            (
                {
                    "scenario_id": t.scenario_id,
                    "card": c.model_dump(mode="json"),
                    "parse_valid": True,
                }
                for t, c in FIXTURES
            ),
        )
    return {
        "gold_path": gold,
        "gold_items_path": items,
        "predictions_dir": predictions,
        "transcripts": transcripts,
        "specs": specs,
        "judge": LLMClient(_NoJudge()),
        "judge_model": "j",
        "calibration": JudgeCalibration(1.0, 1.0, 10),
        "resamples": 20,
    }


def test_final_evaluation_requires_a_frozen_run_and_records_opening(tmp_path):
    files = _evaluation_files(tmp_path)
    lock = tmp_path / "lock.json"
    lock_test_set(files["gold_path"], lock)
    frozen = tmp_path / "frozen.yaml"
    with pytest.raises(FileNotFoundError):
        run_final_evaluation(lock_path=lock, output_dir=tmp_path / "o", frozen_path=frozen, **files)
    assert json.loads(lock.read_text())["openings"] == []
    frozen.write_text("run: x\n")
    summary = run_final_evaluation(
        lock_path=lock, output_dir=tmp_path / "o", frozen_path=frozen, **files
    )
    assert summary["quality_retained"] == pytest.approx(1.0)
    assert len(json.loads(lock.read_text())["openings"]) == 1
    assert (tmp_path / "o" / "errors_to_read.jsonl").exists()
    with pytest.raises(TestSetSealed):
        run_final_evaluation(lock_path=lock, output_dir=tmp_path / "o", frozen_path=frozen, **files)


def test_selection_refuses_the_sealed_test_set_but_accepts_dev(tmp_path):
    files = _evaluation_files(tmp_path)
    lock = tmp_path / "lock.json"
    assert run_selection_evaluation(lock_path=lock, output_dir=tmp_path / "s", **files)
    lock_test_set(files["gold_path"], lock)
    with pytest.raises(TestSetSealed):
        run_selection_evaluation(lock_path=lock, output_dir=tmp_path / "s", **files)


def test_selection_can_focus_on_a_baseline_before_any_student_exists(tmp_path):
    files = _evaluation_files(tmp_path)
    (files["predictions_dir"] / "student.jsonl").rename(
        files["predictions_dir"] / "zero_shot.jsonl"
    )
    summary = run_selection_evaluation(
        lock_path=tmp_path / "none.json", output_dir=tmp_path / "s", focus="zero_shot", **files
    )
    assert "zero_shot" in summary["scores"]


# Robustness.
async def _rules(transcripts):
    return {t.scenario_id: predict_rules(t) for t in transcripts}


def test_noise_sweep_reports_each_level_and_rising_wer():
    gold = {t.scenario_id: c for t, c in FIXTURES}
    clean = [t for t, _ in FIXTURES]
    results = asyncio.run(
        noise_sweep(
            _rules,
            clean,
            gold,
            dict.fromkeys(gold, 1.0),
            __import__("scorecard.data.asr_noise", fromlist=["NoiseProfile"]).NoiseProfile(),
            (0.5, 3.0),
        )
    )
    assert [r.factor for r in results] == [0.5, 3.0]
    assert results[1].wer > results[0].wer
    assert all(r.validity == 1.0 for r in results)


def test_with_attack_adds_a_customer_turn_at_the_end():
    transcript = FIXTURES[0][0]
    attacked = with_attack(transcript, "attack", "direct")
    assert len(attacked.turns) == len(transcript.turns) + 1
    assert attacked.turns[-1].speaker is Speaker.CUSTOMER
    assert attacked.turns[-1].id == transcript.turns[-1].id + 1


def test_injection_detects_a_vulnerable_model_and_clears_a_robust_one():
    calls = [FIXTURES[8][0]]  # The rules baseline gives this call a failed check.

    async def obeys(transcripts):
        cards = {}
        for t in transcripts:
            card = predict_rules(t)
            if t.noise.startswith("attack"):
                body = card.model_dump(mode="json")
                for check in body["compliance"].values():
                    check.update(status="pass", evidence_turns=[1])
                card = Scorecard.model_validate(body)
            cards[t.scenario_id] = card
        return cards

    vulnerable = asyncio.run(injection_test(obeys, calls))
    assert len(vulnerable) == len(ATTACKS)
    assert all(r.fail_flip_rate == 1.0 for r in vulnerable)

    async def robust(transcripts):
        return {t.scenario_id: predict_rules(t) for t in transcripts}

    assert all(r.fail_flip_rate == 0.0 for r in asyncio.run(injection_test(robust, calls)))


def test_injection_counts_invalid_outputs():
    calls = [FIXTURES[8][0]]

    async def breaks(transcripts):
        return {
            t.scenario_id: None if t.noise.startswith("attack") else predict_rules(t)
            for t in transcripts
        }

    assert all(r.invalid_rate == 1.0 for r in asyncio.run(injection_test(breaks, calls)))


def test_bad_inputs_cover_each_case_and_count_invented_passes():
    async def always_passes(transcripts):
        body = CARD.model_dump(mode="json")
        for check in body["compliance"].values():
            check.update(status="pass", evidence_turns=[1])
        return {t.scenario_id: Scorecard.model_validate(body) for t in transcripts}

    results = asyncio.run(bad_input_test(always_passes))
    assert len(results) == len(BAD_INPUTS)
    assert all(r.valid and r.invented_passes == 5 for r in results)


def test_robustness_report_renders_every_section():
    async def run():
        return await injection_test(_rules, [FIXTURES[8][0]]), await bad_input_test(_rules)

    injection, bad = asyncio.run(run())
    report = robustness_markdown([], injection, bad, "x")
    assert "Prompt injection" in report
    assert "Bad inputs" in report


# Reality-check set.
def test_utterances_merge_by_speaker_and_map_agent():
    turns = utterances_to_turns(
        [
            {"speaker": 1, "transcript": "Hello"},
            {"speaker": 1, "transcript": "there"},
            {"speaker": 0, "transcript": "Hi"},
            {"speaker": 0, "transcript": " "},
        ],
        agent_speaker=1,
    )
    assert [(t.id, t.speaker, t.text) for t in turns] == [
        (1, Speaker.AGENT, "Hello there"),
        (2, Speaker.CUSTOMER, "Hi"),
    ]


def test_reality_rows_need_consent_and_prefix():
    turn = Turn(id=1, speaker=Speaker.AGENT, text="Hi")
    with pytest.raises(ConsentMissing):
        reality_row("c1", " ", [turn])
    row = reality_row("c1", "consent-1", [turn])
    assert load_reality_rows([row])[0].scenario_id == "reality-c1"
    without = {k: v for k, v in row.items() if k != "consent_id"}
    with pytest.raises(ConsentMissing):
        load_reality_rows([without])
    with pytest.raises(ValueError):
        load_reality_rows([{**row, "scenario_id": "s1"}])


def test_reality_calls_cannot_reach_training():
    with pytest.raises(RealityLeak):
        assert_no_reality_rows([{"scenario_id": "reality-c1"}])
    row = {"scenario_id": "reality-c1", "card": {}, "language_mix": "en"}
    with pytest.raises(RealityLeak):
        select_training_rows([row], TrainConfig())


def test_gap_report():
    gap = SimToRealGap("student", 0.9, 0.7, 25)
    assert gap.gap == pytest.approx(0.2)
    assert "+0.200" in gap_markdown([gap])


def test_deepgram_requests_diarized_utterances():
    seen = {}

    def handler(request):
        seen.update(request.url.params)
        utterances = [{"speaker": 0, "transcript": "hi"}]
        return httpx.Response(200, json={"results": {"utterances": utterances}})

    stt = DeepgramTranscriber("k", transport=httpx.MockTransport(handler))
    assert stt.transcribe_utterances(b"x")[0]["transcript"] == "hi"
    assert seen["diarize"] == "true"


# Serving.
KEY = {"x-api-key": "k"}


def _body(call_id="c1", text="Hello, my name is Thandi."):
    return {"call_id": call_id, "turns": [{"id": 1, "speaker": "AGENT", "text": text}]}


def _client(backend=None, **settings):
    return TestClient(create_app(Settings(api_key="k", **settings), backend or FakeBackend()))


def test_batch_scores_each_call_and_isolates_failures():
    reply = _client(max_transcript_chars=100).post(
        "/v1/scorecards/batch",
        json={"calls": [_body("a"), _body("b", "x" * 200)]},
        headers=KEY,
    )
    assert reply.status_code == 200
    results = {item["call_id"]: item for item in reply.json()["results"]}
    assert results["a"]["status"] == 200
    assert results["a"]["result"]["scorecard"]["schema_version"] == "1.0"
    assert results["b"]["status"] == 413
    assert results["b"]["result"] is None


def test_batch_limits_and_auth():
    client = _client(max_batch=1)
    calls = {"calls": [_body("a"), _body("b")]}
    assert client.post("/v1/scorecards/batch", json=calls, headers=KEY).status_code == 413
    client = _client()
    assert client.post("/v1/scorecards/batch", json=calls).status_code == 401
    same = {"calls": [_body("a"), _body("a")]}
    assert client.post("/v1/scorecards/batch", json=same, headers=KEY).status_code == 422
    assert client.post("/v1/scorecards/batch", json={"calls": []}, headers=KEY).status_code == 422


class _TooLong(FakeBackend):
    async def generate(self, transcript):
        raise TranscriptTooLong("x")


def test_context_overflow_returns_413_rather_than_503():
    reply = _client(_TooLong()).post("/v1/scorecards", json=_body(), headers=KEY)
    assert reply.status_code == 413


def _vllm(handler):
    return VLLMBackend("http://v/v1", "s", transport=httpx.MockTransport(handler))


def test_vllm_maps_context_errors_but_not_other_bad_requests():
    too_long = httpx.Response(400, json={"message": "maximum context length is 8192 tokens"})
    with pytest.raises(TranscriptTooLong):
        asyncio.run(_vllm(lambda _: too_long).generate(FIXTURES[0][0]))
    other = httpx.Response(400, json={"message": "bad schema"})
    with pytest.raises(BackendUnavailable):
        asyncio.run(_vllm(lambda _: other).generate(FIXTURES[0][0]))


# Load tools.
def _load(target, achieved, errors):
    return LoadResult(target, 60, 100, 100, {}, achieved, achieved * 3600, 1, 2, 3, errors)


def test_saturation_point_picks_highest_sustained_rate():
    results = [_load(1, 1, 0), _load(2, 1.95, 0), _load(4, 2.1, 0), _load(8, 7.9, 0.5)]
    assert saturation_point(results).target_rps == 2
    assert saturation_point([_load(8, 1, 0.5)]) is None


def test_load_report_states_peak_or_its_absence():
    assert "Peak sustained: 3,600" in load_markdown([_load(1, 1, 0)])
    assert "No tested rate" in load_markdown([_load(8, 1, 0.5)])
    gpu = {"gpu_util_mean": 50, "gpu_util_max": 90, "gpu_memory_mb_max": 8000}
    assert "GPU utilisation 50%" in load_markdown([_load(1, 1, 0)], gpu)


def test_body_from_row_builds_a_valid_request():
    transcript = FIXTURES[0][0]
    row = {"scenario_id": "x", "transcript": transcript.model_dump(mode="json")}
    reply = _client().post("/v1/scorecards", json=body_from_row(row), headers=KEY)
    assert reply.status_code == 200


def test_gpu_sampler_tolerates_machines_without_a_gpu(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)

    async def run():
        async with GpuSampler(interval_s=0.01) as sampler:
            await asyncio.sleep(0.03)
        return sampler.summary()

    assert asyncio.run(run()) is None


def test_ttft_probe_reads_a_stream():
    def handler(request):
        assert json.loads(request.content)["stream"] is True
        body = (
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"{"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body)

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(base_url="http://v/v1/", transport=transport) as client:
            return await probe_ttft(client, "s", [{"role": "user", "content": "x"}])

    result = asyncio.run(run())
    assert 0 <= result["ttft_s"] <= result["total_s"]


def test_ledger_stats_skip_cached_calls(tmp_path):
    path = tmp_path / "ledger.jsonl"
    base = {
        "model": "m",
        "input_tokens": 100,
        "output_tokens": 10,
        "elapsed_s": 1.0,
        "cost_usd": 0.002,
        "outcome": "ok",
    }
    write_jsonl(path, [{**base, "cached": False}, {**base, "cached": True, "input_tokens": 9999}])
    stats = ledger_stats(path)["m"]
    assert stats["calls"] == 1
    assert stats["mean_input_tokens"] == 100
    assert stats["usd_per_1k"] == pytest.approx(2.0)


def test_chart_series_has_a_line_per_option():
    inputs = CostInputs(
        self_hosted=[SelfHosted(name="G", gpu_usd_per_hour=1, peak_calls_per_hour=1000)],
        apis=[
            ApiOption(
                name="A",
                input_per_mtok=1,
                output_per_mtok=1,
                batch_discount=0.5,
                input_tokens=1000,
                output_tokens=0,
            )
        ],
        volumes_per_month=[1000, 2000],
    )
    series = chart_series(inputs)
    assert set(series) == {"G, always on", "G, scheduled batches", "A, batch API"}
    assert series["A, batch API"] == [pytest.approx(0.5), pytest.approx(1.0)]


# Bake-off.
def _candidate(verified=True):
    return Candidate(slug="a", base_model="org/a", licence="MIT", licence_verified=verified)


def test_bakeoff_refuses_unverified_licences(tmp_path):
    plan = BakeoffPlan(candidates=[_candidate(False)])
    with pytest.raises(LicenceNotVerified):
        write_bakeoff_configs(plan, TrainConfig(), tmp_path)


def test_bakeoff_writes_short_runs(tmp_path):
    from scorecard.train.config import load_train_config

    paths = write_bakeoff_configs(BakeoffPlan(candidates=[_candidate()]), TrainConfig(), tmp_path)
    config = load_train_config(paths[0])
    assert config.name == "bakeoff-a"
    assert config.base_model == "org/a"
    assert config.train_size == 500


def test_repo_bakeoff_marks_the_unconfirmed_licence():
    plan = load_bakeoff(__import__("pathlib").Path("configs/train/bakeoff.yaml"))
    verified = {c.slug: c.licence_verified for c in plan.candidates}
    assert verified["qwen3-4b"] is True
    assert verified["phi4-mini"] is False


def test_token_report_flags_prompts_over_the_limit():
    transcript = FIXTURES[0][0].model_dump(mode="json")
    rows = [{"transcript": transcript, "language_mix": "en"}]
    report = token_report(CharTokenizer(), rows, max_seq_length=10)
    assert report["en"]["share_over_limit"] == 1.0
    assert report["en"]["tokens_per_word"] > 1
    assert token_report(CharTokenizer(), rows, 10**7)["en"]["share_over_limit"] == 0.0


def test_compare_and_freeze(tmp_path):
    run = tmp_path / "good"
    (run / "final_adapter").mkdir(parents=True)
    (run / "final_adapter" / "adapter.bin").write_bytes(b"weights")
    good = {
        "name": "good",
        "final_metrics": {"val/score": 0.9, "val/validity": 1.0},
        "wall_hours": 1.0,
        "gpu_cost_usd": 0.59,
        "global_step": 10,
        "config": {"lora_rank": 16},
    }
    (run / "run_summary.json").write_text(json.dumps(good))
    other = tmp_path / "other"
    other.mkdir()
    (other / "run_summary.json").write_text(
        json.dumps({**good, "name": "other", "final_metrics": {"val/score": 0.5}})
    )
    table = compare_runs(load_run_summaries(tmp_path))
    assert table.index("| good") < table.index("| other")
    assert "n/a" in table
    output = tmp_path / "frozen.yaml"
    frozen = freeze_run(run, output)
    assert frozen["run"] == "good"
    assert len(frozen["adapter_sha256"]) == 64
    with pytest.raises(FileExistsError):
        freeze_run(run, output)


# Closed baseline through the real SDK types.
def test_anthropic_backend_reads_output_and_refusals():
    pytest.importorskip("anthropic")

    class _Usage:
        input_tokens = 12
        output_tokens = 3

    class _Block:
        type = "text"
        text = "{}"

    class _Message:
        def __init__(self, stop):
            self.stop_reason = stop
            self.usage = _Usage()
            self.content = [_Block()]
            self.stop_details = "frontier_llm"

    class _Messages:
        def __init__(self, stop):
            self.stop = stop
            self.params = None

        async def create(self, **params):
            self.params = params
            return _Message(self.stop)

    class _Client:
        def __init__(self, stop):
            self.messages = _Messages(stop)

    request = LLMRequest(
        "claude-haiku-5-5",
        (("system", "S"), ("user", "U")),
        Purpose.BASELINE,
        json_schema={"type": "object"},
    )
    client = _Client("end_turn")
    response = asyncio.run(AnthropicBackend(client).complete(request))
    assert response.outcome is Outcome.OK
    assert response.input_tokens == 12
    assert client.messages.params["system"] == "S"
    assert client.messages.params["output_config"]["format"]["schema"] == {"type": "object"}
    assert "temperature" not in client.messages.params
    refused = asyncio.run(AnthropicBackend(_Client("refusal")).complete(request))
    assert refused.outcome is Outcome.REFUSED
    capped = asyncio.run(AnthropicBackend(_Client("max_tokens")).complete(request))
    assert capped.outcome is Outcome.TRUNCATED


# Command line.
runner = CliRunner()


@pytest.mark.parametrize("group", ["data", "teacher", "gold", "train", "bench", "predict", "eval"])
def test_each_command_group_has_help(group):
    assert runner.invoke(cli_app, [group, "--help"]).exit_code == 0


def test_cli_adjudicate_refuses_an_unfinished_blind_pass(tmp_path):
    write_jsonl(tmp_path / "gold_items.jsonl", [{"scenario_id": "a", "split": "test"}])
    blind = tmp_path / "blind.jsonl"
    write_jsonl(blind, [])
    result = runner.invoke(cli_app, ["gold", "adjudicate", str(blind), "--out", str(tmp_path)])
    assert result.exit_code != 0


def test_cli_import_reality_refuses_missing_consent(tmp_path):
    source = tmp_path / "r.jsonl"
    write_jsonl(
        source,
        [{"scenario_id": "reality-1", "turns": [{"id": 1, "speaker": "AGENT", "text": "hi"}]}],
    )
    result = runner.invoke(cli_app, ["data", "import-reality", str(source), "--out", str(tmp_path)])
    assert result.exit_code != 0
    assert not (tmp_path / "reality.jsonl").exists()


def test_cli_compare_and_costs(tmp_path):
    assert runner.invoke(cli_app, ["train", "compare", "--runs", str(tmp_path)]).exit_code == 0
    assert (
        runner.invoke(cli_app, ["train", "plan-sweep", "--directory", str(tmp_path)]).exit_code == 0
    )
    assert any(tmp_path.glob("*.yaml"))


def test_dry_run_writes_audit_and_robustness_reports(tmp_path):
    from scorecard.dryrun import run_dry_run

    run_dry_run(120, seed=4, out=tmp_path)
    audit = (tmp_path / "TEACHER_AUDIT.md").read_text(encoding="utf-8")
    robustness = (tmp_path / "ROBUSTNESS.md").read_text(encoding="utf-8")
    assert audit.startswith("> **DRY RUN")
    assert robustness.startswith("> **DRY RUN")
    assert "Noise sweep" in robustness
    assert math.isfinite(json.loads((tmp_path / "summary.json").read_text())["quality_retained"])


def test_fake_backend_still_answers_baselines():
    transcript = Transcript(
        scenario_id="x", turns=(Turn(id=1, speaker=Speaker.AGENT, text="Hello"),)
    )
    client = LLMClient(FakeLLMBackend())
    response = asyncio.run(
        client.complete(LLMRequest("m", (("user", transcript.render()),), Purpose.BASELINE))
    )
    assert flatten_labels(Scorecard.model_validate_json(response.text))
    assert ResponseCache

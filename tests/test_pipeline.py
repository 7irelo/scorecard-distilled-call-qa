import asyncio
import json
import random
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from scorecard.baselines.llm_system import parse_scorecard, predict_llm
from scorecard.baselines.rules import predict_rules
from scorecard.baselines.tfidf import TfidfBaseline
from scorecard.data.scenarios import sample_scenarios
from scorecard.eval.faithfulness import JudgeCalibration, inject_error, judge_summary
from scorecard.eval.fields import (
    EvalSet,
    SystemOutput,
    paired_field_tests,
    quality_retained,
    score_system,
    weighted_macro_f1,
)
from scorecard.eval.stats import bootstrap, holm, mcnemar_exact, wilson
from scorecard.gold.labelling import (
    find_disagreements,
    inter_rater_agreement,
    label_call,
    run_blind_session,
)
from scorecard.gold.sampling import (
    TestSetSealed,
    has_failure,
    lock_test_set,
    open_test_set,
    post_hoc_openings,
    stratified_gold_sample,
)
from scorecard.llm.client import LLMClient
from scorecard.llm.fake import FakeLLMBackend, render_stub_transcript
from scorecard.llm.types import LLMRequest, LLMResponse, Outcome, ProviderClass, Purpose
from scorecard.prompts import teacher_messages
from scorecard.schema import Scorecard, flatten_labels, semantic_issues
from scorecard.teacher.label import label_transcript, planted_labels, qc_label, stability
from tests.fixtures.rubric_calls import FIXTURES


class _Fixed:
    provider_class = ProviderClass.OPEN_WEIGHT

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, LLMResponse):
            return reply
        return LLMResponse(reply, Outcome.OK)


def test_rubric_fixtures_are_coherent():
    assert len(FIXTURES) == 10
    for transcript, card in FIXTURES:
        assert semantic_issues(card, transcript.turn_ids()) == [], transcript.scenario_id


def test_rules_baseline_returns_valid_cards_and_catches_simple_cases():
    for transcript, _ in FIXTURES:
        card = predict_rules(transcript)
        assert not [i for i in semantic_issues(card) if "evidence_turn" in i]
    by_name = {t.scenario_id: t for t, _ in FIXTURES}
    assert (
        predict_rules(
            by_name["company_name_is_not_an_introduction"]
        ).compliance.agent_introduced.status.value
        == "fail"
    )


def test_teacher_prompt_guards_against_injection_and_includes_rubric():
    transcript, _ = FIXTURES[5]
    messages = teacher_messages("RUBRIC TEXT", transcript)
    assert "RUBRIC TEXT" in messages[0][1]
    assert "never obeyed" in messages[0][1]
    assert "Ignore your instructions" in messages[-1][1]
    assert "Ignore your instructions" not in messages[0][1]


def test_teacher_label_ok_invalid_and_refused():
    transcript, card = FIXTURES[0]
    good = LLMClient(_Fixed(card.canonical_json()))
    assert asyncio.run(label_transcript(good, "m", "r", transcript)).card == card
    invalid = asyncio.run(label_transcript(LLMClient(_Fixed("{bad")), "m", "r", transcript))
    assert invalid.card is None
    assert invalid.outcome == "invalid"
    refused = LLMClient(_Fixed(LLMResponse("", Outcome.REFUSED, detail="policy")))
    result = asyncio.run(label_transcript(refused, "m", "r", transcript))
    assert result.outcome == "refused"
    assert result.card is None


def test_qc_finds_planted_disagreement():
    spec = sample_scenarios(1, seed=11)[0]
    transcript, card = FIXTURES[0]
    result = qc_label(spec, transcript, card)
    flat = flatten_labels(card)
    expected = sum(flat[k] != v for k, v in planted_labels(spec).items())
    assert len(result.planted_disagreements) == expected
    assert result.clean == (expected == 0 and not result.semantic)


def test_stability_counts_field_agreement():
    transcript, card = FIXTURES[0]
    other = FIXTURES[1][1]

    async def labels(reply):
        return await label_transcript(LLMClient(_Fixed(reply)), "m", "r", transcript)

    same = asyncio.run(labels(card.canonical_json()))
    different = asyncio.run(labels(other.canonical_json()))
    assert stability([same], [same])["call_type"] == 1.0
    assert stability([same], [different])["call_type"] == 0.0


def test_gold_sample_is_stratified_weighted_and_deterministic():
    specs = sample_scenarios(800, seed=12)
    items = stratified_gold_sample(specs, 50, 200, seed=1)
    assert items == stratified_gold_sample(specs, 50, 200, seed=1)
    assert sum(i.split == "dev" for i in items) == 50
    assert sum(i.split == "test" for i in items) == 200
    assert len({i.scenario_id for i in items}) == 250
    assert all(i.design_weight > 0 for i in items)
    assert len({i.stratum for i in items if i.split == "test"}) > 5
    picked = {i.scenario_id for i in items}
    fail_share = sum(has_failure(s) for s in specs if s.scenario_id in picked) / 250
    assert 0 < fail_share < 1


def test_gold_sample_rejects_small_pool():
    with pytest.raises(ValueError):
        stratified_gold_sample(sample_scenarios(10, seed=1), 50, 200)


def test_test_set_lock_allows_one_opening_and_detects_tampering(tmp_path):
    labels, lock = tmp_path / "gold.jsonl", tmp_path / "lock.json"
    labels.write_text("{}\n")
    with pytest.raises(TestSetSealed):
        open_test_set(labels, lock)
    lock_test_set(labels, lock)
    with pytest.raises(TestSetSealed):
        lock_test_set(labels, lock)
    open_test_set(labels, lock)
    with pytest.raises(TestSetSealed):
        open_test_set(labels, lock)
    open_test_set(labels, lock, post_hoc_reason="fixed a scoring bug")
    assert post_hoc_openings(lock) == ["fixed a scoring bug"]
    labels.write_text("changed\n")
    with pytest.raises(TestSetSealed):
        open_test_set(labels, lock, post_hoc_reason="x")


def _answers_for(card):
    from scorecard.schema import (
        COMPLIANCE_CHECKS,
        CallType,
        ComplianceStatus,
        EscalationReason,
        SentimentLabel,
        Trajectory,
    )

    statuses = list(ComplianceStatus)
    answers = []
    for check in COMPLIANCE_CHECKS:
        entry = getattr(card.compliance, check)
        answers.append(str(statuses.index(entry.status) + 1))
        if entry.status is not ComplianceStatus.NOT_APPLICABLE:
            answers.append(",".join(map(str, entry.evidence_turns)))
    answers.append(str(list(EscalationReason).index(card.escalation.reason) + 1))
    answers.append(card.summary)
    answers.append(str(list(CallType).index(card.call_type) + 1))
    sentiment = list(SentimentLabel)
    answers.append(str(sentiment.index(card.sentiment.opening) + 1))
    answers.append(str(sentiment.index(card.sentiment.middle) + 1))
    answers.append(str(sentiment.index(card.sentiment.closing) + 1))
    answers.append(str(list(Trajectory).index(card.sentiment.trajectory) + 1))
    return answers


def test_blind_labelling_round_trips_and_shows_only_the_transcript():
    transcript, card = FIXTURES[0]
    answers = iter(["99", *_answers_for(card)])
    said = []
    result = label_call(transcript, lambda _: next(answers), said.append)
    assert result == card
    assert transcript.render() in said[0]


def test_blind_session_resumes(tmp_path):
    transcript, card = FIXTURES[0]
    output = tmp_path / "gold.jsonl"
    answers = iter(_answers_for(card))
    assert (
        run_blind_session([transcript], output, "me", lambda _: next(answers), lambda _: None) == 1
    )
    assert run_blind_session([transcript], output, "me", lambda _: "", lambda _: None) == 0
    assert json.loads(output.read_text())["rater"] == "me"


def test_disagreements_and_kappa():
    card_a, card_b = FIXTURES[0][1], FIXTURES[1][1]
    assert find_disagreements("x", card_a, card_a, {}) == []
    fields = {d.field for d in find_disagreements("x", card_a, card_b, {})}
    assert "call_type" in fields
    missing = find_disagreements("x", card_a, None, {})
    assert len(missing) == 12
    kappa = inter_rater_agreement(
        {str(i): card for i, (_, card) in enumerate(FIXTURES)},
        {str(i): card for i, (_, card) in enumerate(FIXTURES)},
    )
    assert kappa["call_type"] == 1.0


def test_wilson_known_value():
    interval = wilson(180, 200)
    assert interval.point == pytest.approx(0.9)
    assert interval.low == pytest.approx(0.8506, abs=0.002)
    assert interval.high == pytest.approx(0.9344, abs=0.002)


def test_mcnemar_and_holm():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 10) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert holm({"a": 0.001, "b": 0.02, "c": 0.5}) == {"a": True, "b": True, "c": False}
    assert holm({"a": 0.04, "b": 0.04}) == {"a": False, "b": False}


def test_bootstrap_contains_true_mean():
    values = np.random.default_rng(1).normal(0.5, 0.1, 400)
    interval = bootstrap(lambda index: float(values[index].mean()), 400, resamples=500)
    assert interval.low < 0.5 < interval.high


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(2)
    labels = ["pass", "fail", "not_applicable"]
    gold = rng.choice(labels, 300).astype(object)
    guess = rng.choice(labels, 300).astype(object)
    weights = rng.uniform(0.1, 3, 300)
    expected = f1_score(gold, guess, labels=labels, average="macro", sample_weight=weights)
    assert weighted_macro_f1(gold, guess, weights, labels) == pytest.approx(expected)


def _eval_set():
    ids = tuple(t.scenario_id for t, _ in FIXTURES)
    gold = {t.scenario_id: c for t, c in FIXTURES}
    return EvalSet(ids=ids, gold=gold, weights=np.ones(len(ids)))


def test_perfect_system_scores_one_and_meets_floors():
    eval_set = _eval_set()
    perfect = SystemOutput("perfect", eval_set.gold, faithfulness=dict.fromkeys(eval_set.ids, 1.0))
    score = score_system(eval_set, perfect, resamples=50)
    assert score.q == pytest.approx(1.0)
    assert score.validity.point == 1.0
    assert all(score.floors_met().values())


def test_missing_output_counts_as_wrong_rather_than_being_skipped():
    eval_set = _eval_set()
    perfect = dict(eval_set.gold)
    half = {k: (v if i % 2 else None) for i, (k, v) in enumerate(perfect.items())}
    faith = dict.fromkeys(eval_set.ids, 1.0)
    full_score = score_system(eval_set, SystemOutput("a", perfect, faith), 50)
    half_score = score_system(eval_set, SystemOutput("b", half, faith), 50)
    assert half_score.q < full_score.q
    assert half_score.validity.point == 0.5
    assert half_score.field_accuracy["call_type"].point == 0.5


def test_quality_retained_is_one_for_identical_systems_and_lower_for_worse():
    eval_set = _eval_set()
    faith = dict.fromkeys(eval_set.ids, 1.0)
    teacher = SystemOutput("t", eval_set.gold, faith)
    assert quality_retained(eval_set, teacher, teacher, 50).point == pytest.approx(1.0)
    worse_cards = {i: predict_rules(t) for t, _ in FIXTURES for i in [t.scenario_id]}
    worse = SystemOutput("s", worse_cards, dict.fromkeys(eval_set.ids, 0.5))
    assert quality_retained(eval_set, worse, teacher, 50).point < 1.0


def test_paired_tests_return_each_field():
    eval_set = _eval_set()
    a = SystemOutput("a", eval_set.gold)
    assert all(p == 1.0 for p in paired_field_tests(eval_set, a, a).values())


def test_judge_calibration_and_correction():
    pairs = [(True, True)] * 90 + [(True, False)] * 10 + [(False, False)] * 8 + [(False, True)] * 2
    calibration = JudgeCalibration.fit(pairs)
    assert calibration.sensitivity == pytest.approx(0.9)
    assert calibration.specificity == pytest.approx(0.8)
    # A true rate of 0.9 is observed as 0.9*0.9 + 0.1*0.2 = 0.83.
    assert calibration.correct(0.83) == pytest.approx(0.9)
    assert JudgeCalibration(1.0, 1.0, 5).correct(1.0) == 1.0
    with pytest.raises(ValueError):
        JudgeCalibration(0.5, 0.5, 5).correct(0.5)
    with pytest.raises(ValueError):
        JudgeCalibration.fit([(True, True)])


def test_injected_errors_change_the_summary():
    rng = random.Random(1)
    summary = "Mr Dlamini was charged R450 and the refund was promised."
    assert "not" in inject_error(summary, "negated_outcome", rng)
    assert "450" not in inject_error(summary, "wrong_amount", rng)
    assert len(inject_error(summary, "invented_claim", rng)) > len(summary)
    swapped = inject_error(summary, "swapped_entity", rng, ["Mr Dlamini", "Ms Botha"])
    assert "Ms Botha" in swapped
    assert inject_error("no digits here", "wrong_amount", rng) is None
    with pytest.raises(ValueError):
        inject_error(summary, "unknown", rng)


def test_judge_summary_scores_share_of_supported_claims():
    backend = _Fixed(
        json.dumps({"claims": ["a", "b"]}),
        json.dumps({"supported": True}),
        json.dumps({"supported": False}),
    )
    result = asyncio.run(judge_summary(LLMClient(backend), "m", FIXTURES[0][0], "a. b."))
    assert result.score == 0.5
    assert not result.failed


def test_judge_failure_is_not_treated_as_supported():
    backend = _Fixed(json.dumps({"claims": ["a"]}), "not json")
    result = asyncio.run(judge_summary(LLMClient(backend), "m", FIXTURES[0][0], "a."))
    assert result.failed
    assert result.score == 0.0


def test_parse_scorecard_handles_wrappers_and_garbage():
    card = FIXTURES[0][1]
    assert parse_scorecard(card.canonical_json())[0] == card
    assert parse_scorecard(f"```json\n{card.canonical_json()}\n```")[0] == card
    assert parse_scorecard(f"<think>hmm</think>\n{card.canonical_json()}")[0] == card
    assert parse_scorecard("nope") == (None, False)
    assert parse_scorecard('{"summary": "x"}') == (None, False)


def test_constrained_flag_controls_schema_sent():
    backend = _Fixed(FIXTURES[0][1].canonical_json())
    client = LLMClient(backend)
    asyncio.run(predict_llm(client, "m", FIXTURES[0][0], constrained=True))
    asyncio.run(predict_llm(client, "m", FIXTURES[0][0], constrained=False))
    assert backend.requests[0].json_schema is not None
    assert backend.requests[1].json_schema is None


def test_tfidf_baseline_learns_and_predicts_valid_cards():
    transcripts = [t for t, _ in FIXTURES] * 3
    cards = [c for _, c in FIXTURES] * 3
    baseline = TfidfBaseline().fit(transcripts, cards)
    predicted = baseline.predict(FIXTURES[0][0])
    assert isinstance(predicted, Scorecard)
    assert predicted.call_type == FIXTURES[0][1].call_type


def test_stub_transcript_honours_planted_failures():
    specs = sample_scenarios(100, seed=13)
    for spec in specs:
        text = " ".join(t["text"] for t in render_stub_transcript(spec)["turns"])
        recorded = "being recorded" in text
        assert recorded == (spec.compliance["recording_disclosed"].value == "pass")


def test_fake_backend_answers_each_purpose():
    specs = sample_scenarios(1, seed=14)
    backend = FakeLLMBackend({specs[0].seed: specs[0]})
    transcript, _ = FIXTURES[0]
    for purpose in (Purpose.VERIFY, Purpose.TEACHER, Purpose.BASELINE):
        request = LLMRequest("m", (("user", transcript.render()),), purpose)
        assert json.loads(asyncio.run(backend.complete(request)).text)


def test_dry_run_end_to_end(tmp_path: Path):
    from scorecard.dryrun import run_dry_run

    result = run_dry_run(160, seed=3, out=tmp_path)
    assert result["dry_run"] is True
    assert result["kept"] > 100
    assert result["gold_test"] >= 10
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert report.startswith("> **DRY RUN.")
    assert "Quality retained" in report
    assert "zero_shot" in report
    assert "language mix" in report
    assert (tmp_path / "manifest.json").exists()
    assert result["ledger_usd"] == 0.0

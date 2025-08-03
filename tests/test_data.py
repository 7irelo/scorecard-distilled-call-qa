import asyncio
import io
import json
import wave

import httpx
import numpy as np
import pytest

from scorecard.data.asr_noise import NoiseProfile, apply_noise, calibrate, word_error_rate
from scorecard.data.codeswitch import mix_is_present, switch_stats
from scorecard.data.generate import behaviour_lines, parse_rendered, render_messages
from scorecard.data.models import LanguageMix, Speaker, Transcript, Turn
from scorecard.data.scenarios import sample_scenario, sample_scenarios
from scorecard.data.speech import (
    AzureSynthesizer,
    DeepgramTranscriber,
    degrade_for_telephony,
    round_trip_turn,
)
from scorecard.data.splits import (
    LeakageError,
    assert_no_group_leakage,
    assign_splits,
    check_manifest,
    write_manifest,
)
from scorecard.data.verify import SpecReading, compare_reading, verify_transcript
from scorecard.llm.client import LLMClient
from scorecard.llm.types import LLMResponse, Outcome, ProviderClass
from scorecard.schema import COMPLIANCE_CHECKS, ComplianceStatus


def transcript(text="Good morning, this call may be recorded. Ngiyabonga, the fee is R450."):
    turns = tuple(
        Turn(id=i + 1, speaker=Speaker.AGENT if i % 2 else Speaker.CUSTOMER, text=text)
        for i in range(30)
    )
    return Transcript(scenario_id="s0", turns=turns)


def test_sampler_is_deterministic_and_seed_sensitive():
    assert sample_scenario(3, 7) == sample_scenario(3, 7)
    assert sample_scenario(3, 7) != sample_scenario(3, 8)


def test_sampler_covers_every_check_and_oversamples_failures():
    specs = sample_scenarios(2000, seed=1)
    assert all(set(s.compliance) == set(COMPLIANCE_CHECKS) for s in specs)
    applicable = [
        status
        for s in specs
        for status in s.compliance.values()
        if status is not ComplianceStatus.NOT_APPLICABLE
    ]
    rate = sum(status is ComplianceStatus.FAIL for status in applicable) / len(applicable)
    assert 0.27 < rate < 0.33


def test_weights_recover_realistic_failure_rate():
    specs = sample_scenarios(6000, seed=2)
    check = "recording_disclosed"
    total = sum(s.weight for s in specs)
    failed = sum(s.weight for s in specs if s.compliance[check] is ComplianceStatus.FAIL)
    assert failed / total == pytest.approx(0.08, abs=0.02)


def test_hang_up_makes_closing_checks_not_applicable():
    specs = [s for s in sample_scenarios(2000, seed=3) if s.hung_up_early]
    assert specs
    for spec in specs:
        assert spec.compliance["next_steps_confirmed"] is ComplianceStatus.NOT_APPLICABLE
        assert spec.outcome == "unresolved"


def test_escalation_flag_matches_reason():
    for spec in sample_scenarios(500, seed=4):
        assert spec.escalation.flag == (spec.escalation.reason.value != "none")


def test_prompt_states_failure_behaviour_and_real_issue():
    spec = next(
        s
        for s in sample_scenarios(200, seed=5)
        if s.compliance["recording_disclosed"] is ComplianceStatus.FAIL
    )
    assert "Nobody ever mentions that the call is being recorded." in behaviour_lines(spec)
    assert spec.issue in render_messages(spec)[1][1]


def test_parse_rendered_numbers_turns_and_rejects_empty():
    body = {"turns": [{"speaker": "AGENT", "text": " Hi "}, {"speaker": "CUSTOMER", "text": "Yes"}]}
    parsed = parse_rendered("x", json.dumps(body))
    assert [t.id for t in parsed.turns] == [1, 2]
    assert parsed.turns[0].text == "Hi"
    with pytest.raises(ValueError):
        parse_rendered("x", json.dumps({"turns": []}))


def test_switch_stats_and_mix_detection():
    text = "Sawubona, my rekening is wrong, ngicela usizo"
    stats = switch_stats(text)
    assert stats.zulu == 3
    assert stats.afrikaans == 1
    assert mix_is_present(text, LanguageMix.EN_ZU_AF)
    assert not mix_is_present("hello there", LanguageMix.EN_ZU)
    assert mix_is_present("hello there", LanguageMix.EN)


def test_noise_is_deterministic_and_keeps_turn_ids():
    clean = transcript()
    profile = NoiseProfile()
    first, second = apply_noise(clean, profile, 1), apply_noise(clean, profile, 1)
    assert first == second
    assert [t.id for t in first.turns] == [t.id for t in clean.turns]
    assert first.noise == "default"
    assert apply_noise(clean, profile, 2) != first


def test_heavier_noise_gives_higher_wer():
    clean = transcript()
    plain = " ".join(t.text for t in clean.turns)
    errors = []
    for factor in (0.5, 3.0):
        noisy = apply_noise(clean, NoiseProfile().scaled(factor, f"x{factor}"), 1)
        errors.append(word_error_rate(plain, " ".join(t.text for t in noisy.turns)))
    assert errors[1] > errors[0]


def test_calibration_recovers_rates_and_reproduces_them():
    reference = " ".join(["the bill is wrong and i want help now please"] * 200)
    profile = NoiseProfile(
        strip_punctuation=True,
        substitution=0.10,
        deletion=0.05,
        insertion=0.0,
        filler=0.0,
        number_garble=0.0,
        code_switch_multiplier=1.0,
    )
    source = Transcript(scenario_id="c", turns=(Turn(id=1, speaker=Speaker.AGENT, text=reference),))
    noisy = apply_noise(source, profile, 3).turns[0].text
    fitted = calibrate([(reference, noisy)])
    assert fitted.deletion == pytest.approx(0.05, abs=0.02)
    assert fitted.substitution == pytest.approx(0.10, abs=0.03)
    assert fitted.confusions


def test_calibration_finds_code_switch_multiplier():
    reference = "the bill ngiyabonga is wrong ngiyabonga for me"
    hypothesis = "the bill giya bonga is wrong yabonga for me"
    assert calibrate([(reference, hypothesis)] * 5).code_switch_multiplier > 1.0


def test_calibration_requires_reference_words():
    with pytest.raises(ValueError):
        calibrate([("", "text")])


def _wav(rate=16000, seconds=1.0):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(samples.tobytes())
    return buffer.getvalue()


def test_degrade_outputs_8khz_mono_of_same_duration():
    with wave.open(io.BytesIO(degrade_for_telephony(_wav()))) as reader:
        assert reader.getframerate() == 8000
        assert reader.getnchannels() == 1
        assert reader.getnframes() == pytest.approx(8000, abs=2)


def test_speech_round_trip_with_mock_vendors():
    seen = {}

    def azure(request):
        seen["ssml"] = request.content.decode()
        return httpx.Response(200, content=_wav())

    def deepgram(request):
        seen["language"] = request.url.params["language"]
        body = {"results": {"channels": [{"alternatives": [{"transcript": "sawubona"}]}]}}
        return httpx.Response(200, json=body)

    synth = AzureSynthesizer("k", "southafricanorth", transport=httpx.MockTransport(azure))
    stt = DeepgramTranscriber("k", transport=httpx.MockTransport(deepgram))
    assert round_trip_turn("Sawubona ngiyabonga <ok>", synth, stt) == "sawubona"
    assert "zu-ZA" in seen["ssml"]
    assert "&lt;ok&gt;" in seen["ssml"]
    assert seen["language"] == "multi"


def test_splits_have_no_leakage_and_approximate_targets():
    specs = sample_scenarios(3000, seed=6)
    fractions = {"train": 0.8, "silver_val": 0.1, "gold_pool": 0.1}
    assignment = assign_splits(specs, fractions, seed=1)
    assert_no_group_leakage(specs, assignment)
    counts = {name: list(assignment.values()).count(name) for name in fractions}
    assert counts["train"] == pytest.approx(2400, rel=0.1)
    assert counts["gold_pool"] == pytest.approx(300, rel=0.4)
    for split in fractions:
        members = {s.call_type for s in specs if assignment[s.scenario_id] == split}
        assert len(members) >= 3


def test_leakage_detected():
    specs = sample_scenarios(200, seed=7)
    group = specs[0].group_id
    twins = [s for s in specs if s.group_id == group]
    if len(twins) < 2:
        twins = [specs[0], specs[0].model_copy(update={"scenario_id": "other"})]
    assignment = {s.scenario_id: "train" for s in specs}
    assignment[twins[1].scenario_id] = "test"
    with pytest.raises(LeakageError):
        assert_no_group_leakage([*specs, *twins], assignment)


def test_manifest_detects_tampering(tmp_path):
    data = tmp_path / "a.jsonl"
    data.write_text("one\n")
    manifest = tmp_path / "m.json"
    write_manifest(manifest, [data])
    assert check_manifest(manifest, tmp_path) == []
    data.write_text("two\n")
    assert check_manifest(manifest, tmp_path) == ["a.jsonl"]


class _Reader:
    provider_class = ProviderClass.OPEN_WEIGHT

    def __init__(self, reading):
        self.reading = reading

    async def complete(self, request):
        return LLMResponse(self.reading, Outcome.OK)


def _reading_for(spec, **overrides):
    values = {check: status.value for check, status in spec.compliance.items()}
    values["escalation_trigger_present"] = spec.escalation.flag
    values.update(overrides)
    return SpecReading.model_validate(values)


def test_compare_reading_reports_each_mismatch():
    spec = sample_scenario(1, 9)
    assert compare_reading(spec, _reading_for(spec)) == []
    flipped = _reading_for(spec, escalation_trigger_present=not spec.escalation.flag)
    assert compare_reading(spec, flipped) == [f"escalation:planted_{spec.escalation.flag}"]


def test_verify_transcript_fails_on_unparseable_reader():
    spec = sample_scenario(1, 9).model_copy(
        update={"language_mix": LanguageMix.EN, "target_turns": 30}
    )
    client = LLMClient(_Reader("not json"))
    result = asyncio.run(verify_transcript(client, "m", spec, transcript()))
    assert not result.passed
    assert "verifier_unparseable" in result.mismatches


def test_verify_transcript_passes_when_reading_agrees():
    spec = sample_scenario(1, 9).model_copy(
        update={"language_mix": LanguageMix.EN, "target_turns": 30}
    )
    client = LLMClient(_Reader(_reading_for(spec).model_dump_json()))
    assert asyncio.run(verify_transcript(client, "m", spec, transcript())).passed

"""Deterministic stand-in for paid models, for tests and the dry run.

Its outputs are not evidence of anything. Every report built from them is watermarked.
"""

from __future__ import annotations

import hashlib
import json
import random
import re

from scorecard.baselines.rules import predict_rules
from scorecard.data.codeswitch import AFRIKAANS_LEXICON, ZULU_LEXICON
from scorecard.data.generate import CHECK_BEHAVIOUR
from scorecard.data.models import LanguageMix, ScenarioSpec, Speaker, Transcript, Turn
from scorecard.data.verify import SpecReading
from scorecard.llm.types import LLMRequest, LLMResponse, Outcome, ProviderClass, Purpose
from scorecard.schema import COMPLIANCE_CHECKS, ComplianceStatus, Scorecard

_LINE = {
    "agent_introduced": "Good day, my name is {agent} from {company}.",
    "recording_disclosed": "Please note this call is being recorded for quality purposes.",
    "identity_verified": "Could I have your ID number please? And your address and date of birth?",
    "terms_explained": "I must explain the notice period and the exit fee in your contract terms.",
    "next_steps_confirmed": "The next step: you will receive an SMS and a reference number today.",
}
_ESCALATION = {
    "customer_threatened_regulator": "If this is not fixed I am going to the ombudsman.",
    "unresolved_repeat_contact": "This is the third time I have called about this.",
    "agent_error": "Sorry, that was wrong, my mistake on the account.",
    "safety": "My alarm is offline and that is a safety danger for us.",
}
_TURN_LINE = re.compile(r"^\[(\d+)\] (AGENT|CUSTOMER): (.*)$", re.M)


def render_stub_transcript(spec: ScenarioSpec) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(spec.seed)
    zulu, afrikaans = sorted(ZULU_LEXICON), sorted(AFRIKAANS_LEXICON)

    def switch(text: str) -> str:
        words = []
        if spec.language_mix in {LanguageMix.EN_ZU, LanguageMix.EN_ZU_AF}:
            words.append(rng.choice(zulu))
        if spec.language_mix in {LanguageMix.EN_AF, LanguageMix.EN_ZU_AF}:
            words.append(rng.choice(afrikaans))
        return f"{text} {' '.join(words)}".strip()

    turns: list[dict[str, str]] = []

    def add(speaker: Speaker, text: str) -> None:
        turns.append({"speaker": speaker.value, "text": text})

    passed = {c for c, s in spec.compliance.items() if s is ComplianceStatus.PASS}
    opening = ["Hello."]
    if "agent_introduced" in passed:
        opening.append(
            _LINE["agent_introduced"].format(agent=spec.agent_name, company=spec.company)
        )
    if "recording_disclosed" in passed:
        opening.append(_LINE["recording_disclosed"])
    add(Speaker.AGENT, " ".join(opening))
    add(Speaker.CUSTOMER, switch(f"I am calling because of {spec.issue}."))
    if "identity_verified" in passed:
        add(Speaker.AGENT, "Could I have your ID number please?")
        add(Speaker.CUSTOMER, "Yes, it is on the system.")
        add(Speaker.AGENT, "And your address and date of birth?")
        add(Speaker.CUSTOMER, "Sure, I can confirm them.")
    if "terms_explained" in passed:
        add(Speaker.AGENT, _LINE["terms_explained"])
    if spec.escalation.flag:
        speaker = (
            Speaker.AGENT if spec.escalation.reason.value == "agent_error" else Speaker.CUSTOMER
        )
        add(speaker, _ESCALATION[spec.escalation.reason.value])
    while len(turns) < spec.target_turns - 2:
        add(Speaker.AGENT, "Let me look into that for you.")
        add(Speaker.CUSTOMER, switch("Okay, I am waiting."))
    add(
        Speaker.AGENT,
        _LINE["next_steps_confirmed"] if "next_steps_confirmed" in passed else "Alright then.",
    )
    add(
        Speaker.CUSTOMER,
        "Thank you." if spec.sentiment.closing.value.endswith("positive") else "Fine.",
    )
    return {"turns": turns}


def parse_prompt_transcript(user_text: str, scenario_id: str = "prompt") -> Transcript:
    turns = tuple(
        Turn(id=int(i), speaker=Speaker(s), text=t) for i, s, t in _TURN_LINE.findall(user_text)
    )
    if not turns:
        turns = (Turn(id=1, speaker=Speaker.CUSTOMER, text=user_text),)
    return Transcript(scenario_id=scenario_id, turns=turns)


def corrupt_card(card: Scorecard, error_rate: float, key: str) -> Scorecard:
    """Flip some compliance statuses at a fixed rate, simulating an imperfect model."""
    rng = random.Random(key)
    body = card.model_dump(mode="json")
    for check in COMPLIANCE_CHECKS:
        if rng.random() < error_rate:
            entry = body["compliance"][check]
            if entry["status"] == "pass":
                body["compliance"][check] = {"status": "fail", "evidence_turns": []}
            elif entry["status"] == "fail":
                body["compliance"][check] = {"status": "pass", "evidence_turns": [1]}
    return Scorecard.model_validate(body)


class FakeLLMBackend:
    """Answers each purpose in the expected format, from rules and planted specs."""

    provider_class = ProviderClass.OPEN_WEIGHT

    def __init__(
        self,
        specs: dict[int, ScenarioSpec] | None = None,
        teacher_error: float = 0.03,
        baseline_error: float = 0.10,
        judge_supported: float = 0.93,
    ) -> None:
        self.specs = specs or {}
        self.teacher_error = teacher_error
        self.baseline_error = baseline_error
        self.judge_supported = judge_supported

    async def complete(self, request: LLMRequest) -> LLMResponse:
        user = request.messages[-1][1]
        text = self._answer(request, user)
        return LLMResponse(
            text, Outcome.OK, input_tokens=len(user) // 4, output_tokens=len(text) // 4
        )

    def _answer(self, request: LLMRequest, user: str) -> str:
        if request.purpose is Purpose.GENERATE:
            return json.dumps(render_stub_transcript(self.specs[int(request.salt.split(":")[0])]))
        transcript = parse_prompt_transcript(user)
        if request.purpose is Purpose.VERIFY:
            return self._read_spec(transcript)
        if request.purpose is Purpose.JUDGE:
            return self._judge(request, user)
        rate = self.teacher_error if request.purpose is Purpose.TEACHER else self.baseline_error
        return corrupt_card(predict_rules(transcript), rate, user + request.salt).canonical_json()

    @staticmethod
    def _read_spec(transcript: Transcript) -> str:
        card = predict_rules(transcript)
        values = {c: getattr(card.compliance, c).status.value for c in CHECK_BEHAVIOUR}
        return SpecReading.model_validate(
            {**values, "escalation_trigger_present": card.escalation.flag}
        ).model_dump_json()

    def _judge(self, request: LLMRequest, user: str) -> str:
        if "<claim>" not in user:
            claims = [part.strip() for part in re.split(r"(?<=[.!?])\s+", user) if part.strip()]
            return json.dumps({"claims": claims or [user]})
        digest = hashlib.sha256(user.encode()).digest()[0] / 255
        return json.dumps({"supported": digest < self.judge_supported})

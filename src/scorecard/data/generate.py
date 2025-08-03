"""Render a scenario spec into a clean transcript with the generator model."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

from scorecard.data.codeswitch import MIX_NAMES, STYLE_GUIDANCE
from scorecard.data.models import ScenarioSpec, Speaker, Transcript, Turn
from scorecard.llm.client import LLMClient
from scorecard.llm.types import LLMRequest, Outcome, Purpose
from scorecard.schema import ComplianceStatus

SYSTEM = """You write realistic transcripts of South African contact-centre phone calls for a \
fictional-data research dataset. All people, companies and account details are invented. Never use \
real ID numbers, real phone numbers or real company names. Write speech as people actually talk: \
hesitations, interruptions, repetition, and people answering a different question from the one \
asked. Follow the behaviour requirements exactly, including the ones where the agent makes \
mistakes. Never mention these requirements in the dialogue."""

CHECK_BEHAVIOUR = {
    "agent_introduced": (
        "Early on, the agent gives their name and the company.",
        "The agent never gives their own name.",
    ),
    "recording_disclosed": (
        "Near the start, the agent says the call may be recorded.",
        "Nobody ever mentions that the call is being recorded.",
    ),
    "identity_verified": (
        "Before discussing account details, the agent verifies identity by asking for at least "
        "two details, such as ID digits and an address.",
        "The agent discusses account details without properly verifying who the caller is.",
    ),
    "terms_explained": (
        "The agent clearly explains the relevant fees, notice period or contract terms.",
        "The agent proceeds without explaining the fees, notice period or contract terms.",
    ),
    "next_steps_confirmed": (
        "Before ending, the agent confirms what happens next and when.",
        "The call ends without the agent confirming what happens next.",
    ),
}
ESCALATION_BEHAVIOUR = {
    "customer_threatened_regulator": "The customer threatens to go to a regulator or ombudsman.",
    "unresolved_repeat_contact": "The customer has called several times already without a fix, "
    "and this call does not fix it either.",
    "agent_error": "The agent makes a clear factual error that harms the customer.",
    "safety": "The customer mentions a safety risk that the outage or issue creates.",
}


class _RenderedTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speaker: Speaker
    text: str


class RenderedCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turns: list[_RenderedTurn]


def behaviour_lines(spec: ScenarioSpec) -> list[str]:
    lines: list[str] = []
    for check, status in spec.compliance.items():
        does, does_not = CHECK_BEHAVIOUR[check]
        if status is ComplianceStatus.PASS:
            lines.append(does)
        elif status is ComplianceStatus.FAIL:
            lines.append(does_not)
    if spec.escalation.flag:
        lines.append(ESCALATION_BEHAVIOUR[spec.escalation.reason.value])
    if spec.hung_up_early:
        lines.append("The customer hangs up abruptly before the call can conclude.")
    return lines


def render_messages(spec: ScenarioSpec) -> tuple[tuple[str, str], ...]:
    requirements = "\n".join(f"- {line}" for line in behaviour_lines(spec))
    user = f"""Write one call of about {spec.target_turns} turns.

Company: {spec.company}
Agent: {spec.agent_name}
Customer: {spec.customer_name}, temperament: {spec.temperament}
Reason for the call: {spec.issue}
How the call ends: {spec.outcome.replace("_", " ")}

Customer sentiment: starts {spec.sentiment.opening.value}, is {spec.sentiment.middle.value} midway, \
and ends {spec.sentiment.closing.value}.

Languages: {MIX_NAMES[spec.language_mix]}.
Style: {STYLE_GUIDANCE[spec.switch_style]}

Behaviour requirements:
{requirements}"""
    return (("system", SYSTEM), ("user", user))


@dataclass(frozen=True)
class RenderResult:
    spec: ScenarioSpec
    transcript: Transcript | None
    failure: str = ""


def parse_rendered(scenario_id: str, text: str) -> Transcript:
    rendered = RenderedCall.model_validate_json(text)
    turns = tuple(
        Turn(id=index, speaker=turn.speaker, text=turn.text.strip())
        for index, turn in enumerate(rendered.turns, start=1)
        if turn.text.strip()
    )
    if not turns:
        raise ValueError("no turns")
    return Transcript(scenario_id=scenario_id, turns=turns)


async def render_scenario(
    client: LLMClient, model: str, spec: ScenarioSpec, attempts: int = 2, feedback: str = ""
) -> RenderResult:
    messages = render_messages(spec)
    if feedback:
        system, user = messages
        messages = (system, ("user", f"{user[1]}\n\nA previous draft failed: {feedback}"))
    failure = ""
    for attempt in range(attempts):
        response = await client.complete(
            LLMRequest(
                model=model,
                messages=messages,
                purpose=Purpose.GENERATE,
                json_schema=RenderedCall.model_json_schema(),
                max_tokens=8192,
                temperature=0.9,
                salt=f"{spec.seed}:{attempt}:{feedback}",
            )
        )
        if response.outcome is not Outcome.OK:
            failure = f"{response.outcome.value}: {response.detail}"
            continue
        try:
            return RenderResult(spec, parse_rendered(spec.scenario_id, response.text))
        except (ValidationError, ValueError) as exc:
            failure = f"parse: {exc}"
    return RenderResult(spec, None, failure)


async def render_all(
    client: LLMClient, model: str, specs: Iterable[ScenarioSpec]
) -> list[RenderResult]:
    return list(await asyncio.gather(*(render_scenario(client, model, s) for s in specs)))

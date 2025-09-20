"""Summary faithfulness: decompose into claims, judge each, and correct for judge error."""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

from scorecard.data.models import Transcript
from scorecard.llm.client import LLMClient
from scorecard.llm.types import LLMRequest, Outcome, Purpose
from scorecard.prompts import INJECTION_GUARD, user_message


class Claims(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[str]


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    supported: bool


CLAIMS_SYSTEM = (
    "Split this summary of a call into its separate factual claims, one per item. "
    "Keep each claim self-contained. Add nothing."
)
JUDGE_SYSTEM = (
    "Decide whether the transcript supports the claim. The transcript comes from speech "
    "recognition and may mix English, isiZulu and Afrikaans. Allow for recognition errors, but a "
    "claim is unsupported if the transcript contradicts it or never establishes it. "
    + INJECTION_GUARD
)


@dataclass(frozen=True)
class FaithfulnessResult:
    claims: tuple[str, ...]
    verdicts: tuple[bool, ...]
    failed: bool = False

    @property
    def score(self) -> float:
        return sum(self.verdicts) / len(self.verdicts) if self.verdicts else 0.0


async def split_claims(client: LLMClient, model: str, summary: str) -> list[str] | None:
    response = await client.complete(
        LLMRequest(
            model=model,
            messages=(("system", CLAIMS_SYSTEM), ("user", summary)),
            purpose=Purpose.JUDGE,
            json_schema=Claims.model_json_schema(),
            max_tokens=1024,
        )
    )
    if response.outcome is not Outcome.OK:
        return None
    try:
        return Claims.model_validate_json(response.text).claims
    except ValidationError:
        return None


async def judge_claim(
    client: LLMClient, model: str, transcript: Transcript, claim: str
) -> bool | None:
    prompt = f"{user_message(transcript)}\n\n<claim>{claim}</claim>"
    response = await client.complete(
        LLMRequest(
            model=model,
            messages=(("system", JUDGE_SYSTEM), ("user", prompt)),
            purpose=Purpose.JUDGE,
            json_schema=ClaimVerdict.model_json_schema(),
            max_tokens=64,
        )
    )
    if response.outcome is not Outcome.OK:
        return None
    try:
        return ClaimVerdict.model_validate_json(response.text).supported
    except ValidationError:
        return None


async def judge_summary(
    client: LLMClient, model: str, transcript: Transcript, summary: str
) -> FaithfulnessResult:
    """The judge receives only a summary and a transcript, never the system that produced it."""
    claims = await split_claims(client, model, summary)
    if not claims:
        return FaithfulnessResult((), (), failed=True)
    verdicts = await asyncio.gather(*(judge_claim(client, model, transcript, c) for c in claims))
    # A claim the judge could not assess is not treated as supported.
    return FaithfulnessResult(
        tuple(claims),
        tuple(bool(v) for v in verdicts),
        failed=any(v is None for v in verdicts),
    )


async def judge_blinded(
    client: LLMClient,
    model: str,
    jobs: Sequence[tuple[str, str, Transcript, str]],
    seed: int = 0,
) -> dict[tuple[str, str], FaithfulnessResult]:
    """Judge (system, scenario_id, transcript, summary) jobs in shuffled order."""
    ordered = list(jobs)
    random.Random(seed).shuffle(ordered)
    results = await asyncio.gather(
        *(
            judge_summary(client, model, transcript, summary)
            for _, _, transcript, summary in ordered
        )
    )
    return {
        (system, scenario_id): result
        for (system, scenario_id, _, _), result in zip(ordered, results, strict=True)
    }


@dataclass(frozen=True)
class JudgeCalibration:
    sensitivity: float
    """Share of truly supported claims the judge accepts."""
    specificity: float
    """Share of truly unsupported claims the judge rejects."""
    n: int

    @classmethod
    def fit(cls, pairs: Sequence[tuple[bool, bool]]) -> JudgeCalibration:
        """Fit from (human says supported, judge says supported) pairs."""
        supported = [judge for human, judge in pairs if human]
        unsupported = [judge for human, judge in pairs if not human]
        if not supported or not unsupported:
            raise ValueError("calibration needs both supported and unsupported claims")
        return cls(
            sensitivity=sum(supported) / len(supported),
            specificity=sum(not judge for judge in unsupported) / len(unsupported),
            n=len(pairs),
        )

    def correct(self, observed: float) -> float:
        """Rogan-Gladen estimate of the true supported rate, given judge error rates."""
        informedness = self.sensitivity + self.specificity - 1
        if informedness <= 0:
            raise ValueError("judge is no better than chance, so its scores cannot be corrected")
        return min(1.0, max(0.0, (observed + self.specificity - 1) / informedness))


_NEGATABLE = re.compile(r"\b(was|were|is|has|have|did)\b(?! not)")
_AMOUNT = re.compile(r"\d+")
INVENTED_CLAIMS = (
    "The agent offered a full year of free service.",
    "The customer said they had already contacted the regulator twice.",
    "A technician was sent out during the call and fixed the fault.",
)
INJECTED_KINDS = ("negated_outcome", "wrong_amount", "invented_claim", "swapped_entity")


def inject_error(
    summary: str, kind: str, rng: random.Random, entities: Sequence[str] = ()
) -> str | None:
    """Make a summary unfaithful in a known way. None means this kind does not apply here."""
    if kind == "negated_outcome":
        match = _NEGATABLE.search(summary)
        if not match:
            return None
        return f"{summary[: match.end()]} not{summary[match.end() :]}"
    if kind == "wrong_amount":
        match = _AMOUNT.search(summary)
        if not match:
            return None
        wrong = str(int(match.group(0)) + rng.randrange(100, 900))
        return f"{summary[: match.start()]}{wrong}{summary[match.end() :]}"
    if kind == "invented_claim":
        return f"{summary} {rng.choice(INVENTED_CLAIMS)}"
    if kind == "swapped_entity":
        present = [entity for entity in entities if entity in summary]
        absent = [entity for entity in entities if entity not in summary]
        if not present or not absent:
            return None
        return summary.replace(present[0], rng.choice(absent))
    raise ValueError(f"unknown error kind: {kind}")

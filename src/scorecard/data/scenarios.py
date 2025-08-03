"""Seeded scenario sampler. Ground truth is fixed here, before any model writes a word."""

from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict

from scorecard.data.models import (
    LanguageMix,
    PlantedEscalation,
    PlantedSentiment,
    ScenarioSpec,
    SwitchStyle,
)
from scorecard.schema import (
    COMPLIANCE_CHECKS,
    CallType,
    ComplianceStatus,
    EscalationReason,
    SentimentLabel,
)

# Invented brands. Check against real SA companies before publishing the dataset.
COMPANIES = (
    "Kopano Fibre",
    "Zolani Mobile",
    "Highveld Connect",
    "Karoo Telecom",
    "Umoya Networks",
    "Silverleaf Prepaid",
)
AGENT_NAMES = ("Thandi", "Sipho", "Annelien", "Pieter", "Nomsa", "Bongani", "Chantel", "Ayanda")
CUSTOMER_NAMES = (
    "Mr Dlamini",
    "Ms Naidoo",
    "Mrs van Wyk",
    "Mr Mokoena",
    "Ms Botha",
    "Mr Khumalo",
    "Mrs Pillay",
    "Ms Ndlovu",
    "Mr Fourie",
    "Mrs Mthembu",
)
TEMPERAMENTS = ("calm", "frustrated", "rushed", "confused", "polite but firm", "angry")

ISSUES: dict[CallType, tuple[tuple[str, str], ...]] = {
    CallType.BILLING_DISPUTE: (
        ("double_debit", "a debit order ran twice this month, about R{amount} each"),
        ("wrong_plan", "billed R{amount} for a plan the customer never agreed to"),
        ("late_fee", "a R{amount} late fee after a payment made on time"),
        ("post_cancel_charge", "still charged R{amount} after cancelling two months ago"),
        ("promo_expired", "a promotion ended early and the bill jumped to R{amount}"),
        ("unknown_addon", "an unrecognised R{amount} add-on on the statement"),
        ("failed_refund", "a promised R{amount} refund has not arrived"),
    ),
    CallType.OUTAGE: (
        ("fibre_down", "the fibre line has been down for {days} days"),
        ("load_shedding", "connectivity does not return after load-shedding ends"),
        ("slow_speed", "speeds far below the paid package for {days} days"),
        ("area_outage", "a suspected area outage with no communication for {days} days"),
        ("router_fault", "a router fault and a missed technician visit"),
        ("mobile_no_signal", "no mobile signal at home for {days} days"),
        ("work_from_home", "an outage threatening the customer's work from home"),
    ),
    CallType.CANCELLATION: (
        ("moving", "cancelling because the customer is moving to an uncovered area"),
        ("price", "cancelling because a competitor is R{amount} cheaper per month"),
        ("poor_service", "cancelling after repeated service failures"),
        ("contract_end", "wants to confirm cancellation at contract end"),
        ("deceased", "cancelling the account of a family member who passed away"),
        ("hardship", "cancelling because of job loss and financial hardship"),
        ("early_exit", "wants out of the contract early and disputes the R{amount} exit fee"),
    ),
    CallType.OTHER: (
        ("upgrade_query", "asking about upgrading to a faster package"),
        ("change_details", "changing the banking details on the account"),
        ("wrong_number", "reached the wrong department"),
    ),
}

OUTCOMES = ("resolved", "partially_resolved", "unresolved", "callback_scheduled")
SWITCH_STYLES = (
    SwitchStyle.SINGLE_INSERTIONS,
    SwitchStyle.INTRA_SENTENCE,
    SwitchStyle.WHOLE_TURN,
    SwitchStyle.WHEN_UPSET,
)
SENTIMENTS = tuple(SentimentLabel)


class SamplerConfig(BaseModel):
    """Sampled rates are oversampled against a realistic base rate. Weights undo this later."""

    model_config = ConfigDict(extra="forbid")
    call_type_shares: dict[CallType, float] = {
        CallType.BILLING_DISPUTE: 0.35,
        CallType.OUTAGE: 0.32,
        CallType.CANCELLATION: 0.28,
        CallType.OTHER: 0.05,
    }
    language_mix_shares: dict[LanguageMix, float] = {
        LanguageMix.EN: 0.40,
        LanguageMix.EN_ZU: 0.30,
        LanguageMix.EN_AF: 0.20,
        LanguageMix.EN_ZU_AF: 0.10,
    }
    sampled_fail_rate: float = 0.30
    realistic_fail_rate: float = 0.08
    sampled_escalation_rate: float = 0.25
    realistic_escalation_rate: float = 0.08
    hang_up_rate: float = 0.05
    min_turns: int = 16
    max_turns: int = 90


def _weighted(rng: random.Random, shares: dict[CallType, float] | dict[LanguageMix, float]):  # type: ignore[no-untyped-def]
    keys = list(shares)
    return rng.choices(keys, weights=[shares[k] for k in keys])[0]  # type: ignore[index]


def _applicable(check: str, call_type: CallType, hung_up: bool, rng: random.Random) -> bool:
    if hung_up and check in {"terms_explained", "next_steps_confirmed"}:
        return False
    if check == "terms_explained":
        return call_type is CallType.CANCELLATION or (
            call_type is CallType.BILLING_DISPUTE and rng.random() < 0.5
        )
    if check == "identity_verified":
        return not (call_type is CallType.OTHER and rng.random() < 0.5)
    return True


def _sentiment(
    rng: random.Random, outcome: str, escalated: bool, temperament: str
) -> PlantedSentiment:
    start = {"angry": 0, "frustrated": 1, "rushed": 1, "confused": 2}.get(temperament, 2)
    start = min(4, max(0, start + rng.choice((-1, 0, 0, 1))))
    drift = {"resolved": 2, "callback_scheduled": 1, "partially_resolved": 0, "unresolved": -1}
    end = start + drift[outcome] + rng.choice((-1, 0, 0, 1))
    if escalated:
        end = min(end, 1)
    end = min(4, max(0, end))
    return PlantedSentiment(
        opening=SENTIMENTS[start],
        middle=SENTIMENTS[(start + end) // 2],
        closing=SENTIMENTS[end],
    )


def sample_scenario(index: int, seed: int, config: SamplerConfig | None = None) -> ScenarioSpec:
    config = config or SamplerConfig()
    scenario_seed = seed * 1_000_003 + index
    rng = random.Random(scenario_seed)

    call_type: CallType = _weighted(rng, config.call_type_shares)
    issue_id, issue_template = rng.choice(ISSUES[call_type])
    issue = issue_template.format(amount=rng.randrange(89, 2400), days=rng.randrange(2, 15))
    company = rng.choice(COMPANIES)
    temperament = rng.choice(TEMPERAMENTS)
    mix: LanguageMix = _weighted(rng, config.language_mix_shares)
    style = SwitchStyle.NONE if mix is LanguageMix.EN else rng.choice(SWITCH_STYLES)
    hung_up = rng.random() < config.hang_up_rate
    outcome = "unresolved" if hung_up else rng.choice(OUTCOMES)

    weight = 1.0
    compliance: dict[str, ComplianceStatus] = {}
    for check in COMPLIANCE_CHECKS:
        if not _applicable(check, call_type, hung_up, rng):
            compliance[check] = ComplianceStatus.NOT_APPLICABLE
        elif rng.random() < config.sampled_fail_rate:
            compliance[check] = ComplianceStatus.FAIL
            weight *= config.realistic_fail_rate / config.sampled_fail_rate
        else:
            compliance[check] = ComplianceStatus.PASS
            weight *= (1 - config.realistic_fail_rate) / (1 - config.sampled_fail_rate)

    escalated = rng.random() < config.sampled_escalation_rate
    reason: EscalationReason
    if escalated:
        weight *= config.realistic_escalation_rate / config.sampled_escalation_rate
        reason = rng.choice([r for r in EscalationReason if r is not EscalationReason.NONE])
    else:
        weight *= (1 - config.realistic_escalation_rate) / (1 - config.sampled_escalation_rate)
        reason = EscalationReason.NONE

    low, high = (6, 14) if hung_up else (config.min_turns, config.max_turns)
    return ScenarioSpec(
        scenario_id=f"s{seed}-{index:06d}",
        group_id=f"{call_type.value}:{issue_id}:{company}",
        seed=scenario_seed,
        call_type=call_type,
        issue_id=issue_id,
        issue=issue,
        company=company,
        agent_name=rng.choice(AGENT_NAMES),
        customer_name=rng.choice(CUSTOMER_NAMES),
        temperament=temperament,
        language_mix=mix,
        switch_style=style,
        target_turns=rng.randint(low, high),
        outcome=outcome,
        hung_up_early=hung_up,
        compliance=compliance,
        sentiment=_sentiment(rng, outcome, escalated, temperament),
        escalation=PlantedEscalation(flag=escalated, reason=reason),
        weight=weight,
    )


def sample_scenarios(n: int, seed: int, config: SamplerConfig | None = None) -> list[ScenarioSpec]:
    return [sample_scenario(i, seed, config) for i in range(n)]

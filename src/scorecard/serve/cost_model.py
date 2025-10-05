"""Cost model. Compares across volume and utilisation, not one ratio."""

from __future__ import annotations

import math
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

HOURS_PER_MONTH = 730.0


class SelfHosted(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    gpu_usd_per_hour: float
    peak_calls_per_hour: float
    measured: bool = False
    """False marks throughput as a placeholder. Reports refuse to present those as results."""


class ApiOption(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    input_per_mtok: float
    output_per_mtok: float
    batch_discount: float = 0.0
    input_tokens: int
    output_tokens: int
    retry_rate: float = 0.0
    measured: bool = False

    def usd_per_1k(self, batch: bool = False) -> float:
        per_call = (
            self.input_tokens * self.input_per_mtok + self.output_tokens * self.output_per_mtok
        ) / 1e6
        discount = 1 - self.batch_discount if batch else 1.0
        return per_call * discount * (1 + self.retry_rate) * 1000


class Fixed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    training_and_labelling_usd: float = 0.0
    amortisation_months: int = 12
    ops_hours_per_month: float = 0.0
    ops_usd_per_hour: float = 0.0
    usd_to_zar: float | None = None


class CostInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    self_hosted: list[SelfHosted]
    apis: list[ApiOption]
    fixed: Fixed = Fixed()
    volumes_per_month: list[int] = [10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]

    @property
    def fully_measured(self) -> bool:
        return all(o.measured for o in self.self_hosted) and all(a.measured for a in self.apis)


def load_cost_inputs(path: Path) -> CostInputs:
    return CostInputs.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def self_hosted_usd_per_1k(option: SelfHosted, utilisation: float) -> float:
    """GPU cost alone, as a function of how busy the GPU is kept."""
    if not 0 < utilisation <= 1:
        raise ValueError("utilisation must be in (0, 1]")
    return option.gpu_usd_per_hour * 1000 / (option.peak_calls_per_hour * utilisation)


def fixed_monthly_usd(fixed: Fixed) -> float:
    return (
        fixed.training_and_labelling_usd / fixed.amortisation_months
        + fixed.ops_hours_per_month * fixed.ops_usd_per_hour
    )


def gpus_needed(option: SelfHosted, volume: int) -> int:
    return max(1, math.ceil(volume / (option.peak_calls_per_hour * HOURS_PER_MONTH)))


def always_on_monthly_usd(option: SelfHosted, volume: int, fixed: Fixed) -> float:
    return gpus_needed(
        option, volume
    ) * option.gpu_usd_per_hour * HOURS_PER_MONTH + fixed_monthly_usd(fixed)


def scheduled_monthly_usd(option: SelfHosted, volume: int, fixed: Fixed) -> float:
    """Batch windows: pay only for the hours needed, as post-call QA can usually wait."""
    return volume / option.peak_calls_per_hour * option.gpu_usd_per_hour + fixed_monthly_usd(fixed)


def always_on_break_even_calls(
    option: SelfHosted, api: ApiOption, fixed: Fixed, batch: bool
) -> float:
    """Monthly volume where one always-on GPU matches the API. inf if it exceeds GPU capacity."""
    per_call = api.usd_per_1k(batch) / 1000
    monthly = option.gpu_usd_per_hour * HOURS_PER_MONTH + fixed_monthly_usd(fixed)
    calls = monthly / per_call if per_call else math.inf
    capacity = option.peak_calls_per_hour * HOURS_PER_MONTH
    return calls if calls <= capacity else math.inf


def break_even_table(inputs: CostInputs) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for option in inputs.self_hosted:
        capacity = option.peak_calls_per_hour * HOURS_PER_MONTH
        for api in inputs.apis:
            for batch in (False, True):
                calls = always_on_break_even_calls(option, api, inputs.fixed, batch)
                rows.append(
                    {
                        "self_hosted": option.name,
                        "api": api.name,
                        "api_mode": "batch" if batch else "standard",
                        "api_usd_per_1k": round(api.usd_per_1k(batch), 4),
                        "break_even_calls_per_month": None if math.isinf(calls) else round(calls),
                        "utilisation_at_break_even": None
                        if math.isinf(calls)
                        else round(calls / capacity, 3),
                    }
                )
    return rows


def volume_table(inputs: CostInputs) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for volume in inputs.volumes_per_month:
        for option in inputs.self_hosted:
            rows.append(
                {
                    "calls_per_month": volume,
                    "option": f"{option.name} always-on",
                    "monthly_usd": round(always_on_monthly_usd(option, volume, inputs.fixed), 2),
                }
            )
            rows.append(
                {
                    "calls_per_month": volume,
                    "option": f"{option.name} scheduled batch",
                    "monthly_usd": round(scheduled_monthly_usd(option, volume, inputs.fixed), 2),
                }
            )
        for api in inputs.apis:
            for batch in (False, True):
                mode = "batch" if batch else "standard"
                rows.append(
                    {
                        "calls_per_month": volume,
                        "option": f"{api.name} {mode}",
                        "monthly_usd": round(api.usd_per_1k(batch) * volume / 1000, 2),
                    }
                )
    if inputs.fixed.usd_to_zar:
        for row in rows:
            row["monthly_zar"] = round(float(str(row["monthly_usd"])) * inputs.fixed.usd_to_zar, 2)
    return rows


def quality_adjusted_usd_per_1k(usd_per_1k: float, quality_retained: float) -> float:
    """Cost per 1k calls, per unit of quality. Cheap and wrong must not look like a win."""
    if quality_retained <= 0:
        return math.inf
    return usd_per_1k / quality_retained


def report_markdown(inputs: CostInputs) -> str:
    lines: list[str] = []
    if not inputs.fully_measured:
        lines.append("> **PLACEHOLDER INPUTS. Do not quote these numbers as results.**\n")
    lines += [
        "| Self-hosted | API | Mode | API $/1k | Break-even calls/month | Utilisation needed |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in break_even_table(inputs):
        calls = row["break_even_calls_per_month"]
        utilisation = row["utilisation_at_break_even"]
        calls_text = f"{calls:,}" if isinstance(calls, int) else "never on one GPU"
        utilisation_text = f"{utilisation:.0%}" if isinstance(utilisation, float) else "n/a"
        lines.append(
            f"| {row['self_hosted']} | {row['api']} | {row['api_mode']} | "
            f"{row['api_usd_per_1k']} | {calls_text} | {utilisation_text} |"
        )
    return "\n".join(lines) + "\n"

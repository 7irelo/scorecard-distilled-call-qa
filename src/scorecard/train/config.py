"""Training configuration and the ablation plan."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class OutputMode(StrEnum):
    FULL_JSON = "full_json"
    PER_FIELD = "per_field"
    HEADS = "heads"


class Quantization(StrEnum):
    NF4 = "nf4"
    NONE = "none"


class LabelFilter(StrEnum):
    RAW = "raw"
    CLEAN = "clean"
    """Drops labels flagged as unstable or that disagree with planted truth."""


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    name: str = "baseline"
    base_model: str = "Qwen/Qwen3-4B"
    revision: str | None = None
    """Pin to a commit hash for real runs."""
    train_path: Path = Path("data/train.jsonl")
    val_path: Path = Path("data/silver_val.jsonl")
    output_dir: Path = Path("runs")
    max_seq_length: int = 8192
    quantization: Quantization = Quantization.NF4
    lora_rank: int = 16
    lora_alpha: int | None = None
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: float = 3.0
    max_steps: int = -1
    batch_size: int = 1
    grad_accum: int = 8
    warmup_ratio: float = 0.05
    seed: int = 42
    output_mode: OutputMode = OutputMode.FULL_JSON
    train_size: int | None = None
    label_filter: LabelFilter = LabelFilter.RAW
    code_switch_share: float = 1.0
    """Fraction of code-switched calls to keep. Used only by ablation F."""
    eval_every_steps: int = 200
    eval_calls: int = 100
    max_new_tokens: int = 700
    early_stopping_patience: int = 4
    gpu_usd_per_hour: float = 0.0
    report_to: str = "wandb"
    wandb_project: str = "scorecard"
    gradient_checkpointing: bool = True
    disable_thinking: bool = True

    @property
    def alpha(self) -> int:
        return self.lora_alpha if self.lora_alpha is not None else 2 * self.lora_rank

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.name


def load_train_config(path: Path) -> TrainConfig:
    return TrainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def ablation_plan(seeds: tuple[int, ...] = (42,)) -> list[dict[str, Any]]:
    """One factor at a time, from the baseline. Selection never touches the test set."""
    factors: list[tuple[str, str, tuple[Any, ...]]] = [
        ("A_size", "train_size", (250, 500, 1000, 2000, 4000)),
        ("B_rank", "lora_rank", (4, 8, 16, 32, 64)),
        ("C_output", "output_mode", tuple(OutputMode)),
        ("D_quant", "quantization", tuple(Quantization)),
        ("E_labels", "label_filter", tuple(LabelFilter)),
        ("F_codeswitch", "code_switch_share", (0.25, 1.0)),
    ]
    runs: list[dict[str, Any]] = []
    for factor, key, levels in factors:
        for level in levels:
            for seed in seeds:
                value = level.value if isinstance(level, StrEnum) else level
                runs.append(
                    {
                        "name": f"{factor}-{value}-s{seed}",
                        "seed": seed,
                        key: value,
                        "factor": factor,
                    }
                )
    return runs


def write_sweep(base: TrainConfig, directory: Path, seeds: tuple[int, ...] = (42,)) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for run in ablation_plan(seeds):
        overrides = {k: v for k, v in run.items() if k != "factor"}
        config = TrainConfig.model_validate({**base.model_dump(mode="json"), **overrides})
        path = directory / f"{run['name']}.yaml"
        path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
        paths.append(path)
    return paths

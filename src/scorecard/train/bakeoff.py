"""Student bake-off, run comparison and freezing. Needs no ML libraries."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from scorecard.data.codeswitch import tokenize_words
from scorecard.data.models import Transcript
from scorecard.train.config import TrainConfig
from scorecard.train.format import ChatTokenizer, full_messages, render_prompt


class LicenceNotVerified(ValueError):
    """A candidate's licence hasn't been read from its model card."""


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    slug: str
    base_model: str
    revision: str | None = None
    licence: str
    licence_verified: bool = False
    notes: str = ""


class BakeoffPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train_size: int = 500
    candidates: list[Candidate]


def load_bakeoff(path: Path) -> BakeoffPlan:
    return BakeoffPlan.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def write_bakeoff_configs(plan: BakeoffPlan, base: TrainConfig, directory: Path) -> list[Path]:
    """One short run per candidate. Refuses any candidate whose licence hasn't been checked."""
    unverified = [c.slug for c in plan.candidates if not c.licence_verified]
    if unverified:
        raise LicenceNotVerified(
            f"check the model card, then set licence_verified: {', '.join(unverified)}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for candidate in plan.candidates:
        config = base.model_copy(
            update={
                "name": f"bakeoff-{candidate.slug}",
                "base_model": candidate.base_model,
                "revision": candidate.revision,
                "train_size": plan.train_size,
            }
        )
        path = directory / f"{candidate.slug}.yaml"
        path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
        paths.append(path)
    return paths


def token_report(
    tokenizer: ChatTokenizer, rows: Iterable[Mapping[str, Any]], max_seq_length: int
) -> dict[str, dict[str, float]]:
    """Tokens per word by language mix. isiZulu words may split into many tokens, using context."""
    per_word: dict[str, list[float]] = defaultdict(list)
    prompt_lengths: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        transcript = Transcript.model_validate(row["transcript"])
        mix = str(row["language_mix"])
        text = " ".join(turn.text for turn in transcript.turns)
        words = len(tokenize_words(text))
        tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if words:
            per_word[mix].append(tokens / words)
        prompt = render_prompt(tokenizer, full_messages(transcript))
        prompt_lengths[mix].append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    report: dict[str, dict[str, float]] = {}
    for mix, lengths in sorted(prompt_lengths.items()):
        values = np.array(lengths)
        report[mix] = {
            "calls": float(len(lengths)),
            "tokens_per_word": float(np.mean(per_word[mix])) if per_word[mix] else float("nan"),
            "prompt_tokens_p50": float(np.percentile(values, 50)),
            "prompt_tokens_p95": float(np.percentile(values, 95)),
            "prompt_tokens_max": float(values.max()),
            "share_over_limit": float((values > max_seq_length).mean()),
        }
    return report


def load_run_summaries(runs_dir: Path, prefix: str = "") -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob(f"{prefix}*/run_summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def _score(summary: Mapping[str, Any]) -> float:
    final = summary.get("final_metrics") or {}
    best = summary.get("best_score")
    candidates = [v for v in (best, final.get("val/score")) if v is not None]
    return max(candidates) if candidates else float("-inf")


def compare_runs(summaries: Iterable[Mapping[str, Any]]) -> str:
    """Rank runs by validation score. This table selects configs, so test data never feeds it."""
    ordered = sorted(summaries, key=_score, reverse=True)
    lines = [
        "| Run | Validation score | Validity | Steps | Hours | GPU cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in ordered:
        final = summary.get("final_metrics") or {}
        score = _score(summary)
        score_text = f"{score:.3f}" if score != float("-inf") else "n/a"
        validity = final.get("val/validity")
        validity_text = f"{validity:.3f}" if validity is not None else "n/a"
        steps = summary.get("global_step", summary.get("steps", "n/a"))
        lines.append(
            f"| {summary['name']} | {score_text} | {validity_text} | {steps} | "
            f"{summary.get('wall_hours', 0):.2f} | ${summary.get('gpu_cost_usd', 0):.2f} |"
        )
    return "\n".join(lines) + "\n"


def _hash_directory(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def freeze_run(run_dir: Path, output: Path, adapter: str = "final_adapter") -> dict[str, Any]:
    """Record the chosen run before the test set opens. Commit the file this writes."""
    if output.exists():
        raise FileExistsError(f"{output} exists; a config should only be frozen once")
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unknown"
    frozen = {
        "run": summary["name"],
        "frozen_at": time.time(),
        "git_sha": git_sha,
        "adapter_sha256": _hash_directory(run_dir / adapter),
        "config": summary.get("config"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(frozen), encoding="utf-8")
    return frozen

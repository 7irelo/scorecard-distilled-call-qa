"""Stratified gold sampling and the one-time lock on the test set."""

from __future__ import annotations

import json
import random
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from scorecard.data.models import ScenarioSpec
from scorecard.io import sha256_file, write_jsonl
from scorecard.schema import ComplianceStatus


class TestSetSealed(RuntimeError):
    """The sealed gold test set was accessed outside the permitted one-time opening."""

    __test__ = False


def has_failure(spec: ScenarioSpec) -> bool:
    return spec.escalation.flag or ComplianceStatus.FAIL in spec.compliance.values()


def stratum(spec: ScenarioSpec) -> str:
    return (
        f"{spec.call_type.value}|{spec.language_mix.value}|{'fail' if has_failure(spec) else 'ok'}"
    )


@dataclass(frozen=True)
class GoldItem:
    scenario_id: str
    split: str
    stratum: str
    design_weight: float
    """Population represented by this item: stratum size over number sampled, times spec weight."""


def stratified_gold_sample(
    specs: Iterable[ScenarioSpec], n_dev: int = 50, n_test: int = 200, seed: int = 0
) -> list[GoldItem]:
    """Equal allocation across strata, topped up from what remains, then split into dev and test."""
    rng = random.Random(seed)
    groups: dict[str, list[ScenarioSpec]] = defaultdict(list)
    for spec in sorted(specs, key=lambda s: s.scenario_id):
        groups[stratum(spec)].append(spec)
    for members in groups.values():
        rng.shuffle(members)

    wanted = n_dev + n_test
    available = sum(len(members) for members in groups.values())
    if available < wanted:
        raise ValueError(f"gold pool has {available} items, need {wanted}")

    allocation = dict.fromkeys(groups, 0)
    while sum(allocation.values()) < wanted:
        for name in sorted(groups, key=lambda g: allocation[g]):
            if allocation[name] < len(groups[name]):
                allocation[name] += 1
                break

    items: list[tuple[ScenarioSpec, float]] = []
    for name, count in allocation.items():
        members = groups[name]
        if count == 0:
            continue
        for spec in members[:count]:
            items.append((spec, len(members) / count * spec.weight))
    rng.shuffle(items)
    # Dev takes every k-th item of a stratum-sorted list, so dev mirrors the test strata.
    items.sort(key=lambda item: stratum(item[0]))
    step = len(items) / n_dev if n_dev else 0
    dev_positions = {int(i * step) for i in range(n_dev)}
    return [
        GoldItem(
            scenario_id=spec.scenario_id,
            split="dev" if position in dev_positions else "test",
            stratum=stratum(spec),
            design_weight=weight,
        )
        for position, (spec, weight) in enumerate(items)
    ]


def write_gold_items(path: Path, items: Iterable[GoldItem]) -> None:
    write_jsonl(path, (vars(item) for item in items))


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def lock_test_set(labels_path: Path, lock_path: Path) -> str:
    """Record the hash of the completed test labels. Only this hash is committed to git."""
    if lock_path.exists():
        raise TestSetSealed(f"{lock_path} already exists")
    digest = sha256_file(labels_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    body = {"sha256": digest, "locked_at": time.time(), "git_sha": _git_sha(), "openings": []}
    lock_path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return digest


def open_test_set(labels_path: Path, lock_path: Path, post_hoc_reason: str | None = None) -> Path:
    """Allow one scored opening. Any later opening needs a reason, which the report must show."""
    if not lock_path.exists():
        raise TestSetSealed("the test set has not been locked")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256_file(labels_path) != lock["sha256"]:
        raise TestSetSealed("test labels changed after locking")
    if lock["openings"] and not post_hoc_reason:
        raise TestSetSealed("already opened; pass a post-hoc reason to open again and disclose it")
    lock["openings"].append(
        {"at": time.time(), "git_sha": _git_sha(), "post_hoc_reason": post_hoc_reason}
    )
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return labels_path


def post_hoc_openings(lock_path: Path) -> list[str]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    return [o["post_hoc_reason"] for o in lock["openings"] if o["post_hoc_reason"]]

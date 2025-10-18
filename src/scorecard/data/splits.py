"""Group-aware splits and content-hash manifests."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from scorecard.data.models import ScenarioSpec
from scorecard.io import sha256_file


class LeakageError(ValueError):
    """The same scenario group appeared in more than one split."""


def assign_splits(
    specs: Iterable[ScenarioSpec], fractions: Mapping[str, float], seed: int = 0
) -> dict[str, str]:
    """Assign whole groups to splits. Largest groups go first, each to the neediest split."""
    groups: dict[str, list[ScenarioSpec]] = defaultdict(list)
    for spec in specs:
        groups[spec.group_id].append(spec)
    total = sum(len(members) for members in groups.values())
    scale = sum(fractions.values())
    targets = {name: fraction / scale * total for name, fraction in fractions.items()}
    filled: Counter[str] = Counter()
    # Tracks each call type per split so that rare types are spread across splits.
    type_filled: Counter[tuple[str, str]] = Counter()
    type_total: Counter[str] = Counter(
        s.call_type.value for members in groups.values() for s in members
    )

    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda group: len(groups[group]), reverse=True)

    assignment: dict[str, str] = {}
    for group in ordered:
        members = groups[group]
        call_type = members[0].call_type.value

        def deficit(split: str, call_type: str = call_type) -> float:
            fraction = targets[split] / total
            overall = (targets[split] - filled[split]) / targets[split]
            typed = (fraction * type_total[call_type] - type_filled[(split, call_type)]) / max(
                1.0, fraction * type_total[call_type]
            )
            return overall + typed

        split = max(targets, key=deficit)
        filled[split] += len(members)
        type_filled[(split, call_type)] += len(members)
        for member in members:
            assignment[member.scenario_id] = split
    return assignment


def assert_no_group_leakage(specs: Iterable[ScenarioSpec], assignment: Mapping[str, str]) -> None:
    seen: dict[str, str] = {}
    for spec in specs:
        split = assignment[spec.scenario_id]
        if seen.setdefault(spec.group_id, split) != split:
            raise LeakageError(f"group {spec.group_id} is in {seen[spec.group_id]} and {split}")


def write_manifest(
    path: Path, files: Iterable[Path], extra: dict[str, object] | None = None
) -> None:
    manifest = {
        "files": {file.name: sha256_file(file) for file in files},
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_manifest(path: Path, directory: Path) -> list[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return [
        name
        for name, digest in manifest["files"].items()
        if not (directory / name).exists() or sha256_file(directory / name) != digest
    ]

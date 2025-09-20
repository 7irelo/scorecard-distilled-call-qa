"""Statistics for the final evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def format(self, digits: int = 3) -> str:
        return f"{self.point:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def wilson(successes: float, total: float, z: float = 1.959964) -> Interval:
    if total <= 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return Interval(p, max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap(
    statistic: Callable[[np.ndarray], float],
    size: int,
    resamples: int = 10_000,
    seed: int = 0,
    strata: Sequence[str] | None = None,
) -> Interval:
    """Percentile bootstrap over call indices, optionally stratified.

    `statistic` takes an index array, so paired comparisons resample both systems identically.
    """
    rng = np.random.default_rng(seed)
    point = statistic(np.arange(size))
    groups = (
        [np.arange(size)]
        if strata is None
        else [np.flatnonzero(np.asarray(strata) == s) for s in sorted(set(strata))]
    )
    values = np.empty(resamples)
    for i in range(resamples):
        picks = [rng.choice(group, size=len(group), replace=True) for group in groups]
        values[i] = statistic(np.concatenate(picks))
    finite = values[np.isfinite(values)]
    if not len(finite):
        return Interval(point, float("nan"), float("nan"))
    low, high = np.percentile(finite, [2.5, 97.5])
    return Interval(point, float(low), float(high))


def mcnemar_exact(only_first_correct: int, only_second_correct: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = only_first_correct + only_second_correct
    if n == 0:
        return 1.0
    k = min(only_first_correct, only_second_correct)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down correction. True means the comparison remains significant."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    result = dict.fromkeys(p_values, False)
    for rank, (name, p) in enumerate(ordered):
        if p > alpha / (len(ordered) - rank):
            break
        result[name] = True
    return result

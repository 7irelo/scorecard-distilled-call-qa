"""Dataset checks from Phase 2: near-duplicates, distributions, native review and noise accuracy."""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from scorecard.data.asr_noise import NoiseProfile, apply_noise, word_error_rate
from scorecard.data.codeswitch import switch_stats
from scorecard.data.models import LanguageMix, ScenarioSpec, Transcript

NATURALNESS_GATE = 3.5
NOISE_TOLERANCE = 0.05
REVIEW_FIELDS = ("scenario_id", "language_mix", "transcript", "naturalness_1_to_5", "notes")


def _text(transcript: Transcript) -> str:
    return " ".join(turn.text for turn in transcript.turns)


def find_near_duplicates(
    transcripts: Sequence[Transcript], threshold: float = 0.9
) -> list[tuple[str, str, float]]:
    """Pairs with TF-IDF cosine similarity at the threshold. Sparse, so 5k calls stay cheap."""
    if len(transcripts) < 2:
        return []
    matrix = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(
        _text(t) for t in transcripts
    )
    similarity = (matrix @ matrix.T).tocoo()
    pairs = [
        (transcripts[int(i)].scenario_id, transcripts[int(j)].scenario_id, float(value))
        for i, j, value in zip(similarity.row, similarity.col, similarity.data, strict=True)
        if i < j and value >= threshold
    ]
    return sorted(pairs, key=lambda pair: pair[2], reverse=True)


def drop_near_duplicates(
    transcripts: Sequence[Transcript], pairs: Iterable[tuple[str, str, float]]
) -> list[Transcript]:
    """Keep the first call of each duplicate pair."""
    dropped: set[str] = set()
    for first, second, _ in pairs:
        if first not in dropped:
            dropped.add(second)
    return [t for t in transcripts if t.scenario_id not in dropped]


def dataset_report(
    specs: Mapping[str, ScenarioSpec], transcripts: Sequence[Transcript]
) -> dict[str, object]:
    turns = np.array([len(t.turns) for t in transcripts]) if transcripts else np.array([0])
    words = np.array([len(_text(t).split()) for t in transcripts]) if transcripts else np.array([0])
    by_mix: dict[str, list[Transcript]] = defaultdict(list)
    for transcript in transcripts:
        by_mix[specs[transcript.scenario_id].language_mix.value].append(transcript)
    lexicon_rates: dict[str, dict[str, float]] = {}
    for mix, members in sorted(by_mix.items()):
        stats = [switch_stats(_text(t)) for t in members]
        lexicon_rates[mix] = {
            "calls": float(len(members)),
            "zulu_lexicon_share": float(np.mean([s.zulu_share for s in stats])),
            "afrikaans_lexicon_share": float(np.mean([s.afrikaans_share for s in stats])),
        }
    return {
        "calls": len(transcripts),
        "call_types": dict(
            Counter(specs[t.scenario_id].call_type.value for t in transcripts).most_common()
        ),
        "turns": _percentiles(turns),
        "words": _percentiles(words),
        "language_mix": lexicon_rates,
    }


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def write_review_sheet(
    path: Path,
    specs: Mapping[str, ScenarioSpec],
    transcripts: Sequence[Transcript],
    language: str,
    n: int = 50,
    seed: int = 0,
) -> int:
    """A sheet for one native reviewer. `language` is `zu` or `af`. Labels are left out."""
    wanted = {
        "zu": {LanguageMix.EN_ZU, LanguageMix.EN_ZU_AF},
        "af": {LanguageMix.EN_AF, LanguageMix.EN_ZU_AF},
    }[language]
    pool = [t for t in transcripts if specs[t.scenario_id].language_mix in wanted]
    random.Random(seed).shuffle(pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for transcript in pool[:n]:
            writer.writerow(
                {
                    "scenario_id": transcript.scenario_id,
                    "language_mix": specs[transcript.scenario_id].language_mix.value,
                    "transcript": transcript.render(),
                    "naturalness_1_to_5": "",
                    "notes": "",
                }
            )
    return len(pool[:n])


@dataclass(frozen=True)
class ReviewResult:
    reviewed: int
    blank: int
    mean: float
    passed: bool


def score_review_sheet(path: Path, gate: float = NATURALNESS_GATE) -> ReviewResult:
    """Unrated rows don't count, and an invalid rating is an error, never a guess."""
    scores: list[int] = []
    blank = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value = (row.get("naturalness_1_to_5") or "").strip()
            if not value:
                blank += 1
                continue
            score = int(value)
            if not 1 <= score <= 5:
                raise ValueError(f"{row['scenario_id']}: rating must be 1 to 5, got {score}")
            scores.append(score)
    mean = float(np.mean(scores)) if scores else float("nan")
    return ReviewResult(len(scores), blank, mean, bool(scores) and mean >= gate)


@dataclass(frozen=True)
class NoiseCheck:
    achieved_wer: float
    target_wer: float
    within_tolerance: bool


def check_noise_accuracy(
    transcripts: Sequence[Transcript],
    profile: NoiseProfile,
    target_wer: float | None = None,
    seed: int = 0,
    tolerance: float = NOISE_TOLERANCE,
) -> NoiseCheck:
    """Compare the injector's real WER with the measured one. Phase 2's gate is a few points."""
    target = (
        target_wer
        if target_wer is not None
        else profile.substitution + profile.deletion + profile.insertion
    )
    rates = [
        word_error_rate(_text(t), _text(apply_noise(t, profile, seed)))
        for t in transcripts
        if _text(t)
    ]
    achieved = float(np.mean(rates)) if rates else float("nan")
    return NoiseCheck(achieved, target, abs(achieved - target) <= tolerance)

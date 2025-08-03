"""Deterministic ASR noise injection, calibrated from a real speech-to-text round trip.

Turn IDs stay stable, so evidence pointers keep their meaning across noise levels.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Iterable

import jiwer
from pydantic import BaseModel, ConfigDict, Field

from scorecard.data.codeswitch import is_code_switched_word
from scorecard.data.models import Speaker, Transcript, Turn

_FILLERS = ("um", "uh", "ja", "eh", "mm")
_NUMBER = re.compile(r"\d+")


class NoiseProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = "default"
    substitution: float = 0.08
    deletion: float = 0.03
    insertion: float = 0.01
    code_switch_multiplier: float = Field(default=2.0, ge=1.0)
    """How much more often code-switched words go wrong, relative to the rest."""
    diarization_swap: float = 0.02
    interruption: float = 0.05
    filler: float = 0.03
    number_garble: float = 0.2
    strip_punctuation: bool = True
    lowercase: bool = True
    confusions: dict[str, list[str]] = Field(default_factory=dict)

    def scaled(self, factor: float, name: str) -> NoiseProfile:
        return self.model_copy(
            update={
                "name": name,
                "substitution": min(1.0, self.substitution * factor),
                "deletion": min(1.0, self.deletion * factor),
                "insertion": min(1.0, self.insertion * factor),
                "diarization_swap": min(1.0, self.diarization_swap * factor),
                "interruption": min(1.0, self.interruption * factor),
                "filler": min(1.0, self.filler * factor),
                "number_garble": min(1.0, self.number_garble * factor),
            }
        )


def _perturb(word: str, rng: random.Random, profile: NoiseProfile) -> str:
    options = profile.confusions.get(word)
    if options:
        return rng.choice(options)
    if len(word) <= 2:
        return word + rng.choice("aeiou")
    position = rng.randrange(1, len(word))
    return word[:position] + rng.choice("aeioumnst") + word[position + 1 :]


def _garble_number(match: re.Match[str], rng: random.Random, chance: float) -> str:
    digits = match.group(0)
    if rng.random() >= chance:
        return digits
    position = rng.randrange(len(digits))
    return digits[:position] + str(rng.randrange(10)) + digits[position + 1 :]


def corrupt_text(text: str, rng: random.Random, profile: NoiseProfile) -> str:
    text = _NUMBER.sub(lambda m: _garble_number(m, rng, profile.number_garble), text)
    if profile.lowercase:
        text = text.lower()
    if profile.strip_punctuation:
        text = re.sub(r"[.,!?;:\"]", "", text)
    output: list[str] = []
    for word in text.split():
        scale = profile.code_switch_multiplier if is_code_switched_word(word.lower()) else 1.0
        roll = rng.random()
        if roll < min(1.0, profile.deletion * scale):
            continue
        if roll < min(1.0, (profile.deletion + profile.substitution) * scale):
            word = _perturb(word, rng, profile)
        if rng.random() < profile.filler:
            output.append(rng.choice(_FILLERS))
        output.append(word)
        if rng.random() < profile.insertion:
            output.append(rng.choice(output))
    return " ".join(output)


def apply_noise(transcript: Transcript, profile: NoiseProfile, seed: int) -> Transcript:
    rng = random.Random(f"{seed}:{transcript.scenario_id}:{profile.name}")
    turns: list[Turn] = []
    for turn in transcript.turns:
        text = corrupt_text(turn.text, rng, profile)
        words = text.split()
        if len(words) > 4 and rng.random() < profile.interruption:
            text = " ".join(words[: rng.randrange(2, len(words))]) + " --"
        speaker = turn.speaker
        if rng.random() < profile.diarization_swap:
            speaker = Speaker.CUSTOMER if speaker is Speaker.AGENT else Speaker.AGENT
        turns.append(Turn(id=turn.id, speaker=speaker, text=text or "[inaudible]"))
    return Transcript(scenario_id=transcript.scenario_id, turns=tuple(turns), noise=profile.name)


def _normalize(text: str) -> str:
    return re.sub(r"[.,!?;:\"]", "", text.lower())


def calibrate(pairs: Iterable[tuple[str, str]], name: str = "calibrated") -> NoiseProfile:
    """Fit word-level rates from (reference, STT hypothesis) text pairs."""
    hits = subs = deletions = insertions = 0
    cs_total = cs_errors = other_total = other_errors = 0
    confusions: Counter[tuple[str, str]] = Counter()
    for reference, hypothesis in pairs:
        ref_text, hyp_text = _normalize(reference), _normalize(hypothesis)
        if not ref_text.strip():
            continue
        if not hyp_text.strip():
            ref_words = ref_text.split()
            deletions += len(ref_words)
            cs_count = sum(is_code_switched_word(w) for w in ref_words)
            cs_total += cs_count
            cs_errors += cs_count
            other_total += len(ref_words) - cs_count
            other_errors += len(ref_words) - cs_count
            continue
        result = jiwer.process_words(ref_text, hyp_text)
        ref_words, hyp_words = result.references[0], result.hypotheses[0]
        for chunk in result.alignments[0]:
            if chunk.type == "insert":
                insertions += chunk.hyp_end_idx - chunk.hyp_start_idx
                continue
            span = ref_words[chunk.ref_start_idx : chunk.ref_end_idx]
            failed = chunk.type != "equal"
            if chunk.type == "equal":
                hits += len(span)
            elif chunk.type == "delete":
                deletions += len(span)
            else:
                subs += len(span)
                replaced = hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx]
                confusions.update(zip(span, replaced, strict=False))
            for word in span:
                if is_code_switched_word(word):
                    cs_total += 1
                    cs_errors += failed
                else:
                    other_total += 1
                    other_errors += failed
    total = hits + subs + deletions
    if total == 0:
        raise ValueError("no reference words to calibrate from")
    # Add-one smoothing keeps the ratio finite when one group has no errors in a small sample.
    cs_rate = (cs_errors + 1) / (cs_total + 2)
    other_rate = (other_errors + 1) / (other_total + 2)
    multiplier = max(1.0, cs_rate / other_rate) if cs_total else 1.0
    table: dict[str, list[str]] = {}
    for (source, target), _count in confusions.most_common(500):
        table.setdefault(source, []).append(target)
    return NoiseProfile(
        name=name,
        substitution=subs / total,
        deletion=deletions / total,
        insertion=insertions / total,
        code_switch_multiplier=multiplier,
        confusions=table,
    )


def word_error_rate(reference: str, hypothesis: str) -> float:
    return float(jiwer.wer(_normalize(reference), _normalize(hypothesis)))

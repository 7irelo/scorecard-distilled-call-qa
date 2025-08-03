"""Code-switching guidance and a lexicon-based measure of it.

The lexicon is a coarse aid, not a language identifier: it undercounts inflected isiZulu forms and
cannot judge naturalness. Native-speaker review is the actual gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scorecard.data.models import LanguageMix, SwitchStyle

ZULU_LEXICON = frozenset(
    [
        "sawubona",
        "yebo",
        "cha",
        "ngiyabonga",
        "ngicela",
        "kodwa",
        "manje",
        "hhayi",
        "angazi",
        "uxolo",
        "kahle",
        "imali",
        "ugesi",
        "inkinga",
        "ngiyaxolisa",
        "ngiyaqonda",
        "usizo",
        "sicela",
        "ngifuna",
        "akukho",
        "kungani",
        "namhlanje",
        "izolo",
        "kusasa",
        "hawu",
    ]
)
AFRIKAANS_LEXICON = frozenset(
    [
        "ja",
        "nee",
        "asseblief",
        "dankie",
        "maar",
        "nou",
        "baie",
        "ek",
        "verstaan",
        "rekening",
        "lekker",
        "hoekom",
        "geld",
        "goed",
        "môre",
        "vandag",
        "gister",
        "betaal",
        "kanselleer",
        "kan",
        "nie",
        "jy",
        "julle",
        "ons",
        "wat",
    ]
)
SA_ENGLISH_LEXICON = frozenset({"eish", "shame", "hayibo", "now-now", "yoh", "sharp", "aikona"})

STYLE_GUIDANCE = {
    SwitchStyle.NONE: "Use South African English only. Local expressions like 'eish' are fine.",
    SwitchStyle.SINGLE_INSERTIONS: "Mostly English, with single words or greetings in the "
    "other language(s).",
    SwitchStyle.INTRA_SENTENCE: "Switch languages mid-sentence, the way bilingual speakers do "
    "when a phrase comes more naturally in the other language.",
    SwitchStyle.WHOLE_TURN: "Some whole turns are in the other language, while the agent keeps "
    "replying mostly in English.",
    SwitchStyle.WHEN_UPSET: "The customer starts in English and moves into the other language "
    "as their frustration rises.",
}
MIX_NAMES = {
    LanguageMix.EN: "South African English",
    LanguageMix.EN_ZU: "South African English and isiZulu",
    LanguageMix.EN_AF: "South African English and Afrikaans",
    LanguageMix.EN_ZU_AF: "South African English, isiZulu and Afrikaans",
}
_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)?", re.UNICODE)


@dataclass(frozen=True)
class SwitchStats:
    words: int
    zulu: int
    afrikaans: int

    @property
    def zulu_share(self) -> float:
        return self.zulu / self.words if self.words else 0.0

    @property
    def afrikaans_share(self) -> float:
        return self.afrikaans / self.words if self.words else 0.0


def tokenize_words(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(text)]


def is_code_switched_word(word: str) -> bool:
    return word in ZULU_LEXICON or word in AFRIKAANS_LEXICON


def switch_stats(text: str) -> SwitchStats:
    words = tokenize_words(text)
    return SwitchStats(
        words=len(words),
        zulu=sum(word in ZULU_LEXICON for word in words),
        afrikaans=sum(word in AFRIKAANS_LEXICON for word in words),
    )


def mix_is_present(text: str, mix: LanguageMix) -> bool:
    """True if the requested languages are detectably present. Coarse by design."""
    stats = switch_stats(text)
    wants_zulu = mix in {LanguageMix.EN_ZU, LanguageMix.EN_ZU_AF}
    wants_afrikaans = mix in {LanguageMix.EN_AF, LanguageMix.EN_ZU_AF}
    return (not wants_zulu or stats.zulu > 0) and (not wants_afrikaans or stats.afrikaans > 0)

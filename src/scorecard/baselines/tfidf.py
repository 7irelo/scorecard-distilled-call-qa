"""TF-IDF and logistic regression per field: the cheap ML floor, trained on the same labels."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from scorecard.data.models import Speaker, Transcript
from scorecard.schema import COMPLIANCE_CHECKS, Scorecard, flatten_labels


def _document(transcript: Transcript) -> str:
    return "\n".join(f"{t.speaker.value} {t.text}" for t in transcript.turns)


class _Constant:
    def __init__(self, value: str) -> None:
        self.value = value

    def predict(self, documents: Sequence[str]) -> list[str]:
        return [self.value] * len(documents)


class TfidfBaseline:
    def __init__(self) -> None:
        self._models: dict[str, Pipeline | _Constant] = {}

    def fit(self, transcripts: Sequence[Transcript], cards: Sequence[Scorecard]) -> TfidfBaseline:
        documents = [_document(t) for t in transcripts]
        rows = [flatten_labels(card) for card in cards]
        for name in rows[0]:
            labels = [row[name] for row in rows]
            if len(set(labels)) < 2:
                self._models[name] = _Constant(labels[0])
                continue
            model = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            )
            model.fit(documents, labels)
            self._models[name] = model
        return self

    def predict(self, transcript: Transcript) -> Scorecard:
        document = [_document(transcript)]
        flat = {name: str(model.predict(document)[0]) for name, model in self._models.items()}
        customer = [t.text for t in transcript.turns if t.speaker is Speaker.CUSTOMER]
        reason = flat["escalation.reason"]
        return Scorecard.model_validate(
            {
                # Classifiers cannot summarize or cite, so these fields are stand-ins by design.
                "summary": " ".join(" ".join(customer[:2]).split()[:60]) or "No customer speech.",
                "call_type": flat["call_type"],
                "sentiment": {
                    "opening": flat["sentiment.opening"],
                    "middle": flat["sentiment.middle"],
                    "closing": flat["sentiment.closing"],
                    "trajectory": flat["sentiment.trajectory"],
                },
                "compliance": {
                    check: {"status": flat[f"compliance.{check}"], "evidence_turns": []}
                    for check in COMPLIANCE_CHECKS
                },
                "escalation": {"flag": flat["escalation.flag"] == "true", "reason": reason},
            }
        )

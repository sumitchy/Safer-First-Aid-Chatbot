"""Baseline B: intent-classifier FAQ bot.

Represents the current rule-based state of the art: a classifier maps the user
query to one of a fixed set of first-aid intents, and a canned, human-written
answer for that intent is returned. Safe but brittle — it cannot handle anything
outside its training intents.

Implementation:
    TF-IDF features + Logistic Regression. This is deliberately simple, fast, fully
    reproducible, and needs no GPU. It is trained on a first-aid intents dataset
    (Kaggle elvisblitti/first-aid or a JSON of intent->examples). For queries whose
    top predicted probability is below a threshold, it abstains and refers the user
    on — modelling the brittleness of FAQ bots honestly.

For the thesis this baseline shows that guideline-grounded RAG can match the SAFETY
of a canned FAQ bot while far exceeding its COVERAGE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..pipeline.safety import SafetyDecision, SafetyLayer


@dataclass
class IntentResponse:
    query: str
    answer: str
    raw_answer: str
    intent: str | None = None
    confidence: float = 0.0
    safety: SafetyDecision | None = None
    backend_name: str = "intent-classifier"
    retrieved: list = field(default_factory=list)

    def provenance(self) -> list[str]:
        return [f"intent:{self.intent}"] if self.intent else []


class IntentClassifierBaseline:
    """TF-IDF + Logistic Regression FAQ bot."""

    def __init__(
        self,
        safety: SafetyLayer | None = None,
        confidence_threshold: float = 0.35,
    ) -> None:
        self.safety = safety or SafetyLayer()
        self.confidence_threshold = confidence_threshold
        self._vectorizer = None
        self._clf = None
        self._answers: dict[str, str] = {}

    # -- training ---------------------------------------------------------- #
    def fit(self, intents: dict[str, dict]) -> "IntentClassifierBaseline":
        """Train on an intents dict.

        Format:
            {
              "choking_adult": {
                  "examples": ["someone is choking", "food stuck in throat", ...],
                  "answer": "1. Encourage them to cough ..."
              },
              ...
            }
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        texts: list[str] = []
        labels: list[str] = []
        for intent, payload in intents.items():
            self._answers[intent] = payload["answer"]
            for ex in payload["examples"]:
                texts.append(ex)
                labels.append(intent)

        if len(set(labels)) < 2:
            raise ValueError("Need at least two intents to train the classifier.")

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = self._vectorizer.fit_transform(texts)
        self._clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._clf.fit(X, labels)
        return self

    @classmethod
    def from_json(cls, path: Path, **kwargs) -> "IntentClassifierBaseline":
        intents = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**kwargs).fit(intents)

    # -- inference --------------------------------------------------------- #
    def _predict(self, query: str) -> tuple[str | None, float]:
        if self._clf is None:
            raise RuntimeError("Classifier not trained. Call fit() or from_json().")
        X = self._vectorizer.transform([query])
        probs = self._clf.predict_proba(X)[0]
        classes = self._clf.classes_
        best = probs.argmax()
        return classes[best], float(probs[best])

    def answer(self, query: str) -> IntentResponse:
        intent, conf = self._predict(query)
        if conf < self.confidence_threshold:
            raw = (
                "I'm not sure I have specific first-aid guidance for that. Please "
                "contact a medical professional or call the emergency number if urgent."
            )
            intent = None
        else:
            raw = self._answers.get(intent, "")

        final, decision = self.safety.apply(query, raw)
        return IntentResponse(
            query=query,
            answer=final,
            raw_answer=raw,
            intent=intent,
            confidence=conf,
            safety=decision,
        )

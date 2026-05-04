from __future__ import annotations

import time

from src.config import LABELS
from src.preprocessing import clean_text
from src.schema import ModerationResult, explanation_for


class KeywordFallbackModerator:
    """Deterministic fallback for demos before a trained artifact exists."""

    toxic_terms = {"idiot", "trash", "garbage", "fool", "dumbest", "useless", "pathetic"}
    abusive_terms = {"harass", "threaten", "attack", "targeting", "humiliating", "pile on"}
    policy_terms = {
        "stolen",
        "password",
        "credit card",
        "malware",
        "api key",
        "private data",
        "fake account",
        "bypass",
        "leaked",
    }

    def predict(self, text: str) -> ModerationResult:
        start = time.perf_counter()
        cleaned = clean_text(text).lower()
        scores = {label: 0.05 for label in LABELS}
        scores["safe"] = 0.35

        for term in self.toxic_terms:
            if term in cleaned:
                scores["toxic"] += 0.35
                scores["safe"] -= 0.08
        for term in self.abusive_terms:
            if term in cleaned:
                scores["abusive"] += 0.45
                scores["safe"] -= 0.1
        for term in self.policy_terms:
            if term in cleaned:
                scores["policy_violation"] += 0.45
                scores["safe"] -= 0.1

        scores = {label: max(value, 0.01) for label, value in scores.items()}
        total = sum(scores.values())
        probabilities = {label: value / total for label, value in scores.items()}
        predicted_label = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_label]
        latency_ms = (time.perf_counter() - start) * 1000
        explanation = explanation_for(predicted_label, confidence)
        if predicted_label == "safe":
            explanation += " This is a keyword fallback because no trained model artifact was found."
        return ModerationResult(
            text=text,
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=latency_ms,
            model_name="keyword_demo_fallback",
            explanation=explanation,
        )


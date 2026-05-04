from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModerationResult:
    text: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    model_name: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "predicted_label": self.predicted_label,
            "confidence": round(float(self.confidence), 6),
            "probabilities": {
                label: round(float(probability), 6)
                for label, probability in self.probabilities.items()
            },
            "latency_ms": round(float(self.latency_ms), 3),
            "model_name": self.model_name,
            "explanation": self.explanation,
        }


def explanation_for(label: str, confidence: float) -> str:
    if label == "safe":
        return "No strong toxicity, harassment, or policy-violation signal was detected."
    if label == "toxic":
        return "The text contains insulting or hostile language that may degrade the conversation."
    if label == "abusive":
        return "The text appears to target a person or group with harassment or intimidation."
    if label == "policy_violation":
        return "The text appears to request, offer, or encourage behavior that violates platform policy."
    return f"The model selected {label} with confidence {confidence:.2%}."


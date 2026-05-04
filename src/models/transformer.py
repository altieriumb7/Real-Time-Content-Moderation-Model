from __future__ import annotations

import time
from pathlib import Path

from src.config import LABELS
from src.preprocessing import clean_text
from src.schema import ModerationResult, explanation_for


class TransformerModerator:
    def __init__(self, model_dir: Path) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Transformer inference requires torch and transformers. Install requirements.txt first."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        id2label = getattr(self.model.config, "id2label", None) or {}
        self.labels = [
            str(id2label.get(idx, LABELS[idx] if idx < len(LABELS) else idx))
            for idx in range(self.model.config.num_labels)
        ]
        self.model_name = f"transformer:{model_dir.name}"

    @classmethod
    def load(cls, model_dir: str | Path) -> "TransformerModerator":
        return cls(Path(model_dir))

    def predict(self, text: str) -> ModerationResult:
        cleaned = clean_text(text)
        inputs = self.tokenizer(
            cleaned,
            truncation=True,
            padding=True,
            max_length=160,
            return_tensors="pt",
        )
        start = time.perf_counter()
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities_tensor = self.torch.softmax(logits, dim=-1)[0].detach().cpu()
        latency_ms = (time.perf_counter() - start) * 1000
        probabilities = {
            label: float(probabilities_tensor[idx])
            for idx, label in enumerate(self.labels)
        }
        predicted_label = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_label]
        return ModerationResult(
            text=text,
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=latency_ms,
            model_name=self.model_name,
            explanation=explanation_for(predicted_label, confidence),
        )


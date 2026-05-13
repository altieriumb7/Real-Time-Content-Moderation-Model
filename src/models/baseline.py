from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.config import BASELINE_MODEL_PATH, LABELS
from src.models.numpy_tfidf import NumpyTfidfLogReg
from src.preprocessing import clean_text
from src.schema import ModerationResult, explanation_for


def train_numpy_baseline(
    texts: list[str],
    labels: list[str],
    epochs: int = 700,
    max_features: int = 5000,
) -> NumpyTfidfLogReg:
    model = NumpyTfidfLogReg(epochs=epochs, max_features=max_features)
    return model.fit(texts, labels)


def train_sklearn_baseline(texts: list[str], labels: list[str]) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=20000,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=None,
                ),
            ),
        ]
    )
    return pipeline.fit(texts, labels)


def train_baseline_model(
    texts: list[str],
    labels: list[str],
    prefer_sklearn: bool = True,
    numpy_epochs: int = 700,
) -> tuple[Any, str]:
    if prefer_sklearn:
        try:
            return train_sklearn_baseline(texts, labels), "sklearn_tfidf_logreg"
        except Exception as exc:
            print(f"scikit-learn baseline unavailable ({exc}); using NumPy fallback.")
    return train_numpy_baseline(texts, labels, epochs=numpy_epochs), "numpy_tfidf_logreg"


def infer_label_order(model: Any) -> list[str]:
    if hasattr(model, "labels"):
        return list(model.labels)
    if hasattr(model, "classes_"):
        return [str(label) for label in model.classes_]
    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        if classifier is not None and hasattr(classifier, "classes_"):
            return [str(label) for label in classifier.classes_]
    return list(LABELS)


def save_baseline(model: Any, model_type: str, path: Path = BASELINE_MODEL_PATH) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "model_type": model_type, "labels": infer_label_order(model)}, path)


class BaselineModerator:
    def __init__(
        self,
        model: Any,
        model_type: str = "baseline_tfidf_logreg",
        labels: list[str] | None = None,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.labels = labels or infer_label_order(model)

    @classmethod
    def load(cls, path: Path = BASELINE_MODEL_PATH) -> "BaselineModerator":
        import joblib

        artifact = joblib.load(path)
        if isinstance(artifact, dict):
            return cls(
                artifact["model"],
                artifact.get("model_type", "baseline_tfidf_logreg"),
                artifact.get("labels"),
            )
        return cls(artifact)

    def predict(self, text: str) -> ModerationResult:
        start = time.perf_counter()
        cleaned = clean_text(text)
        raw_proba = self.model.predict_proba([cleaned])[0]
        probabilities = {
            label: float(raw_proba[idx])
            for idx, label in enumerate(self.labels)
        }
        predicted_label = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_label]
        latency_ms = (time.perf_counter() - start) * 1000
        return ModerationResult(
            text=text,
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=latency_ms,
            model_name=self.model_type,
            explanation=explanation_for(predicted_label, confidence),
        )

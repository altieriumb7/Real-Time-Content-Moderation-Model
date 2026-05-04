from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from src.config import LABELS

TOKEN_RE = re.compile(r"[a-z0-9_']+", flags=re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class NumpyTfidfLogReg:
    """Small dependency-light TF-IDF + multinomial logistic regression baseline."""

    def __init__(
        self,
        labels: list[str] | None = None,
        max_features: int = 5000,
        min_df: int = 1,
        learning_rate: float = 0.8,
        epochs: int = 700,
        l2: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.labels = labels or list(LABELS)
        self.max_features = max_features
        self.min_df = min_df
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None

    def fit(self, texts: list[str], labels: list[str]) -> "NumpyTfidfLogReg":
        self._fit_vectorizer(texts)
        x = self.transform(texts)
        y = np.array([self.labels.index(label) for label in labels], dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = x.shape
        n_classes = len(self.labels)
        self.weights_ = rng.normal(0.0, 0.01, size=(n_features, n_classes))
        self.bias_ = np.zeros(n_classes)
        y_onehot = np.eye(n_classes)[y]

        for _ in range(self.epochs):
            logits = x @ self.weights_ + self.bias_
            probabilities = _softmax(logits)
            error = probabilities - y_onehot
            grad_w = (x.T @ error) / n_samples + self.l2 * self.weights_
            grad_b = error.mean(axis=0)
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
        return self

    def _fit_vectorizer(self, texts: list[str]) -> None:
        df: Counter[str] = Counter()
        tf: Counter[str] = Counter()
        for text in texts:
            tokens = tokenize(text)
            tf.update(tokens)
            df.update(set(tokens))
        terms = [
            term
            for term, _ in tf.most_common()
            if df[term] >= self.min_df
        ][: self.max_features]
        self.vocabulary_ = {term: idx for idx, term in enumerate(terms)}
        n_docs = len(texts)
        self.idf_ = np.ones(len(self.vocabulary_), dtype=np.float64)
        for term, idx in self.vocabulary_.items():
            self.idf_[idx] = math.log((1 + n_docs) / (1 + df[term])) + 1

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.idf_ is None:
            raise ValueError("Vectorizer is not fitted.")
        x = np.zeros((len(texts), len(self.vocabulary_)), dtype=np.float64)
        for row_idx, text in enumerate(texts):
            counts = Counter(tokenize(text))
            total = sum(counts.values()) or 1
            for token, count in counts.items():
                col_idx = self.vocabulary_.get(token)
                if col_idx is not None:
                    x[row_idx, col_idx] = (count / total) * self.idf_[col_idx]
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model is not fitted.")
        x = self.transform(texts)
        return _softmax(x @ self.weights_ + self.bias_)

    def predict(self, texts: list[str]) -> list[str]:
        probabilities = self.predict_proba(texts)
        indices = probabilities.argmax(axis=1)
        return [self.labels[int(index)] for index in indices]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


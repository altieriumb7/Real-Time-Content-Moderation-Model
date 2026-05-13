from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Sequence

from src.config import CONFUSION_MATRIX_CSV, LABELS


def _empty_label_metrics() -> dict[str, float | int]:
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = LABELS,
) -> list[list[int]]:
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for actual, predicted in zip(y_true, y_pred):
        if actual in index and predicted in index:
            matrix[index[actual]][index[predicted]] += 1
    return matrix


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = LABELS,
) -> dict[str, object]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    matrix = confusion_matrix(y_true, y_pred, labels)
    per_label: dict[str, dict[str, float | int]] = {}
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(labels)))

    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[row][i] for row in range(len(labels)) if row != i)
        fn = sum(matrix[i][col] for col in range(len(labels)) if col != i)
        support = sum(matrix[i])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    supports = [int(per_label[label]["support"]) for label in labels]
    macro = {
        key: mean(float(per_label[label][key]) for label in labels)
        for key in ["precision", "recall", "f1"]
    }
    weighted = {}
    for key in ["precision", "recall", "f1"]:
        weighted[key] = (
            sum(float(per_label[label][key]) * int(per_label[label]["support"]) for label in labels)
            / total
            if total
            else 0.0
        )

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_avg": macro,
        "weighted_avg": weighted,
        "per_label": per_label,
        "support": dict(zip(labels, supports)),
        "confusion_matrix": matrix,
    }


def try_roc_auc(
    y_true: Sequence[str],
    probabilities: Sequence[Sequence[float]] | None,
    labels: Sequence[str] = LABELS,
) -> float | None:
    if probabilities is None:
        return None
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None
    if len(set(y_true)) < 2:
        return None
    y_indices = [labels.index(label) for label in y_true]
    try:
        return float(roc_auc_score(y_indices, probabilities, multi_class="ovr", labels=list(range(len(labels)))))
    except Exception:
        return None


def latency_summary(latencies_ms: Sequence[float] | None) -> dict[str, float] | None:
    if not latencies_ms:
        return None
    sorted_values = sorted(float(value) for value in latencies_ms)
    p95_index = min(len(sorted_values) - 1, math.ceil(0.95 * len(sorted_values)) - 1)
    return {
        "mean_ms": mean(sorted_values),
        "median_ms": median(sorted_values),
        "p95_ms": sorted_values[p95_index],
        "min_ms": min(sorted_values),
        "max_ms": max(sorted_values),
    }


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probabilities: Sequence[Sequence[float]] | None = None,
    latencies_ms: Sequence[float] | None = None,
    labels: Sequence[str] = LABELS,
) -> dict[str, object]:
    metrics = classification_metrics(y_true, y_pred, labels)
    metrics["roc_auc_ovr"] = try_roc_auc(y_true, probabilities, labels)
    metrics["latency"] = latency_summary(latencies_ms)
    return metrics


def write_json_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def write_confusion_matrix_csv(
    matrix: list[list[int]],
    labels: Sequence[str] = LABELS,
    path: Path = CONFUSION_MATRIX_CSV,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def plot_confusion_matrix(
    matrix: list[list[int]],
    labels: Sequence[str],
    path: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion Matrix")
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color="black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


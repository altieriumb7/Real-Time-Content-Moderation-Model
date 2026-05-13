from __future__ import annotations

import argparse
import json
import time

from src.comparison import update_model_comparison
from src.config import (
    BASELINE_METRICS_PATH,
    BASELINE_MODEL_PATH,
    CONFUSION_MATRIX_CSV,
    CONFUSION_MATRIX_PNG,
    LABELS,
)
from src.evaluation import (
    evaluate_predictions,
    plot_confusion_matrix,
    write_confusion_matrix_csv,
    write_json_report,
)
from src.models.baseline import BaselineModerator, save_baseline, train_baseline_model
from src.preprocessing import load_records, split_texts_labels, stratified_split, summarize_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TF-IDF + Logistic Regression baseline.")
    parser.add_argument("--dataset", default=None, help="Path to labeled CSV. Defaults to Jigsaw or demo data.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-sklearn", action="store_true", help="Force the NumPy fallback baseline.")
    parser.add_argument("--numpy-epochs", type=int, default=700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.dataset)
    train_records, val_records, test_records = stratified_split(records, seed=args.seed)
    train_texts, train_labels = split_texts_labels(train_records)
    test_texts, test_labels = split_texts_labels(test_records)

    model, model_type = train_baseline_model(
        train_texts,
        train_labels,
        prefer_sklearn=not args.no_sklearn,
        numpy_epochs=args.numpy_epochs,
    )
    save_baseline(model, model_type, BASELINE_MODEL_PATH)
    moderator = BaselineModerator.load(BASELINE_MODEL_PATH)

    predictions = []
    probabilities = []
    latencies = []
    for text in test_texts:
        result = moderator.predict(text)
        predictions.append(result.predicted_label)
        probabilities.append([result.probabilities.get(label, 0.0) for label in LABELS])
        latencies.append(result.latency_ms)

    metrics = evaluate_predictions(
        test_labels,
        predictions,
        probabilities=probabilities,
        latencies_ms=latencies,
        labels=LABELS,
    )
    metrics["model_name"] = model_type
    metrics["model_path"] = str(BASELINE_MODEL_PATH)
    metrics["latencies_ms"] = [round(value, 6) for value in latencies]
    metrics["dataset"] = summarize_records(records)
    metrics["split_sizes"] = {
        "train": len(train_records),
        "validation": len(val_records),
        "test": len(test_records),
    }
    metrics["generated_at_unix"] = time.time()

    write_json_report(metrics, BASELINE_METRICS_PATH)
    write_confusion_matrix_csv(metrics["confusion_matrix"], LABELS, CONFUSION_MATRIX_CSV)
    plot_confusion_matrix(metrics["confusion_matrix"], LABELS, CONFUSION_MATRIX_PNG)
    update_model_comparison()

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import BASELINE_METRICS_PATH, LABELS
from src.evaluation import evaluate_predictions, write_json_report
from src.inference import load_moderator
from src.preprocessing import load_records, split_texts_labels, stratified_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained moderation model.")
    parser.add_argument("--dataset", default=None, help="Path to labeled CSV. Defaults to Jigsaw or demo data.")
    parser.add_argument("--model-path", default=None, help="Model artifact path.")
    parser.add_argument("--output", default=str(BASELINE_METRICS_PATH), help="Metrics JSON output path.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.dataset)
    _, _, test_records = stratified_split(records, seed=args.seed)
    texts, labels = split_texts_labels(test_records)
    moderator = load_moderator(args.model_path, allow_fallback=False)

    predictions = []
    probabilities = []
    latencies = []
    for text in texts:
        result = moderator.predict(text)
        predictions.append(result.predicted_label)
        probabilities.append([result.probabilities[label] for label in LABELS])
        latencies.append(result.latency_ms)

    metrics = evaluate_predictions(labels, predictions, probabilities, latencies, LABELS)
    metrics["latencies_ms"] = [round(value, 6) for value in latencies]
    write_json_report(metrics, Path(args.output))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

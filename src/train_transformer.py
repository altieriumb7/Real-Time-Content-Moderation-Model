from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.comparison import update_model_comparison
from src.config import (
    DEFAULT_TRANSFORMER_MODEL,
    LABELS,
    MODEL_DIR,
    TRANSFORMER_METRICS_PATH,
)
from src.evaluation import evaluate_predictions, write_json_report
from src.preprocessing import load_records, split_texts_labels, stratified_split, summarize_records


class ModerationTorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels: list[str]) -> None:
        self.encodings = encodings
        self.labels = [LABELS.index(label) for label in labels]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a lightweight transformer moderator.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model-name", default=DEFAULT_TRANSFORMER_MODEL)
    parser.add_argument("--output-dir", default=str(MODEL_DIR / "transformer_distilbert"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick experiments.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError(
            "Transformer training requires `transformers`, `accelerate`, and their dependencies. "
            "Install requirements.txt and retry."
        ) from exc

    args = parse_args()
    records = load_records(args.dataset)
    if args.max_samples:
        records = records[: args.max_samples]
    train_records, val_records, test_records = stratified_split(records, seed=args.seed)
    train_texts, train_labels = split_texts_labels(train_records)
    val_texts, val_labels = split_texts_labels(val_records)
    test_texts, test_labels = split_texts_labels(test_records)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label={idx: label for idx, label in enumerate(LABELS)},
        label2id={label: idx for idx, label in enumerate(LABELS)},
    )
    train_dataset = ModerationTorchDataset(
        tokenizer(train_texts, truncation=True, padding=True, max_length=args.max_length),
        train_labels,
    )
    val_dataset = ModerationTorchDataset(
        tokenizer(val_texts, truncation=True, padding=True, max_length=args.max_length),
        val_labels,
    )

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        probabilities = softmax_np(logits)
        predicted_ids = probabilities.argmax(axis=1)
        y_true = [LABELS[int(idx)] for idx in label_ids]
        y_pred = [LABELS[int(idx)] for idx in predicted_ids]
        metrics = evaluate_predictions(y_true, y_pred, probabilities.tolist(), labels=LABELS)
        return {
            "macro_f1": metrics["macro_avg"]["f1"],
            "weighted_f1": metrics["weighted_avg"]["f1"],
            "accuracy": metrics["accuracy"],
        }

    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        seed=args.seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
    latencies = []
    probabilities = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for idx, text in enumerate(test_texts):
            single = tokenizer(text, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
            start = time.perf_counter()
            logits = model(**single).logits.detach().cpu().numpy()
            latencies.append((time.perf_counter() - start) * 1000)
            proba = softmax_np(logits)[0]
            probabilities.append(proba.tolist())
            predictions.append(LABELS[int(proba.argmax())])

    metrics = evaluate_predictions(test_labels, predictions, probabilities, latencies, LABELS)
    metrics["model_name"] = args.model_name
    metrics["model_path"] = str(output_dir)
    metrics["latencies_ms"] = [round(value, 6) for value in latencies]
    metrics["dataset"] = summarize_records(records)
    metrics["split_sizes"] = {
        "train": len(train_records),
        "validation": len(val_records),
        "test": len(test_records),
    }
    write_json_report(metrics, TRANSFORMER_METRICS_PATH)
    update_model_comparison()
    print(json.dumps(metrics, indent=2, sort_keys=True))


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()

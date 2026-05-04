from __future__ import annotations

import json
from pathlib import Path

from src.config import BASELINE_METRICS_PATH, COMPARISON_PATH, TRANSFORMER_METRICS_PATH
from src.evaluation import write_json_report


def update_model_comparison(
    baseline_path: Path = BASELINE_METRICS_PATH,
    transformer_path: Path = TRANSFORMER_METRICS_PATH,
    output_path: Path = COMPARISON_PATH,
) -> dict[str, object]:
    comparison = {
        "baseline": _summarize_metric_file(
            baseline_path,
            missing_reason="Run `python -m src.train_baseline` to compute baseline metrics.",
        ),
        "transformer": _summarize_metric_file(
            transformer_path,
            missing_reason="Run `python -m src.train_transformer` after installing transformer dependencies.",
        ),
    }
    write_json_report(comparison, output_path)
    return comparison


def _summarize_metric_file(path: Path, missing_reason: str) -> dict[str, object]:
    if not path.exists():
        return {"status": "not_run", "reason": missing_reason}
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return {
        "status": "computed",
        "model_name": metrics.get("model_name"),
        "macro_f1": metrics.get("macro_avg", {}).get("f1"),
        "weighted_f1": metrics.get("weighted_avg", {}).get("f1"),
        "macro_precision": metrics.get("macro_avg", {}).get("precision"),
        "macro_recall": metrics.get("macro_avg", {}).get("recall"),
        "roc_auc_ovr": metrics.get("roc_auc_ovr"),
        "latency": metrics.get("latency"),
    }


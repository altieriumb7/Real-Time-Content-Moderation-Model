from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.config import BASELINE_MODEL_PATH, MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export moderation models to ONNX when dependencies support it.")
    parser.add_argument("--baseline-path", default=str(BASELINE_MODEL_PATH))
    parser.add_argument("--transformer-dir", default=str(MODEL_DIR / "transformer_distilbert"))
    parser.add_argument("--output", default=str(MODEL_DIR / "moderation_model.onnx"))
    parser.add_argument("--compare-text", default="Please review this comment before posting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transformer_dir = Path(args.transformer_dir)
    if transformer_dir.exists():
        report = export_transformer_onnx(transformer_dir, Path(args.output), args.compare_text)
    else:
        report = export_sklearn_baseline(Path(args.baseline_path), Path(args.output), args.compare_text)
    print(json.dumps(report, indent=2, sort_keys=True))


def export_sklearn_baseline(model_path: Path, output_path: Path, compare_text: str) -> dict[str, object]:
    try:
        import joblib
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import StringTensorType
    except Exception as exc:
        return {
            "status": "not_exported",
            "reason": f"Baseline ONNX export requires joblib and skl2onnx: {exc}",
        }
    if not model_path.exists():
        return {"status": "not_exported", "reason": f"No baseline artifact found at {model_path}."}
    artifact = joblib.load(model_path)
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    if not hasattr(model, "named_steps"):
        return {
            "status": "not_exported",
            "reason": "The NumPy fallback model is not ONNX-exportable. Train with scikit-learn first.",
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_model = convert_sklearn(model, initial_types=[("text", StringTensorType([None, 1]))])
    output_path.write_bytes(onnx_model.SerializeToString())
    return {"status": "exported", "onnx_path": str(output_path), "latency_comparison": None}


def export_transformer_onnx(transformer_dir: Path, output_path: Path, compare_text: str) -> dict[str, object]:
    try:
        import onnxruntime as ort
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        return {
            "status": "not_exported",
            "reason": f"Transformer ONNX export requires torch, transformers, and onnxruntime: {exc}",
        }
    tokenizer = AutoTokenizer.from_pretrained(str(transformer_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(transformer_dir))
    model.eval()
    inputs = tokenizer(compare_text, return_tensors="pt", truncation=True, padding=True, max_length=160)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )
    pytorch_ms = _time_pytorch(model, inputs)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }
    onnx_ms = _time_onnx(session, onnx_inputs)
    return {
        "status": "exported",
        "onnx_path": str(output_path),
        "latency_comparison": {"pytorch_ms": pytorch_ms, "onnx_ms": onnx_ms},
    }


def _time_pytorch(model, inputs, runs: int = 20) -> float:
    import torch

    with torch.no_grad():
        for _ in range(3):
            model(**inputs)
        start = time.perf_counter()
        for _ in range(runs):
            model(**inputs)
    return (time.perf_counter() - start) * 1000 / runs


def _time_onnx(session, inputs, runs: int = 20) -> float:
    for _ in range(3):
        session.run(None, inputs)
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, inputs)
    return (time.perf_counter() - start) * 1000 / runs


if __name__ == "__main__":
    main()


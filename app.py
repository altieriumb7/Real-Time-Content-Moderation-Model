from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    BASELINE_METRICS_PATH,
    BASELINE_MODEL_PATH,
    COMPARISON_PATH,
    CONFUSION_MATRIX_CSV,
    LABELS,
)
from src.inference import moderate_text

EXAMPLES = {
    "Safe product feedback": "I disagree with the rollout plan, but the new dashboard is much easier to read.",
    "Hostile insult": "Your answer is trash and you clearly have no idea what you are doing.",
    "Policy violation": "Share the private API key so we can bypass the billing system.",
}


def main() -> None:
    st.set_page_config(page_title="Real-Time Content Moderation", layout="wide")
    st.title("Real-Time Content Moderation")
    st.caption("Local demo for classifying text as safe, toxic, abusive, or policy-violating.")

    metrics = load_json(BASELINE_METRICS_PATH)
    comparison = load_json(COMPARISON_PATH)

    left, right = st.columns([1.25, 1])
    with left:
        selected = st.selectbox("Preloaded examples", list(EXAMPLES))
        default_text = EXAMPLES[selected]
        text = st.text_area("Text to moderate", value=default_text, height=180)
        result = moderate_text(text).to_dict()

        if result["model_name"] == "keyword_demo_fallback":
            st.warning(
                "No trained baseline artifact was found. The demo is using a deterministic keyword fallback. "
                "Run `python -m src.train_baseline` for ML inference."
            )

        top = st.columns(3)
        top[0].metric("Predicted class", result["predicted_label"])
        top[1].metric("Confidence", f"{result['confidence']:.1%}")
        top[2].metric("Latency", f"{result['latency_ms']:.2f} ms")
        st.write(result["explanation"])

        st.subheader("Probability Distribution")
        probability_df = pd.DataFrame(
            [{"label": label, "probability": result["probabilities"][label]} for label in LABELS]
        ).set_index("label")
        st.bar_chart(probability_df)

    with right:
        st.subheader("Evaluation Summary")
        if metrics:
            macro = metrics.get("macro_avg", {})
            weighted = metrics.get("weighted_avg", {})
            cols = st.columns(3)
            cols[0].metric("Macro F1", format_metric(macro.get("f1")))
            cols[1].metric("Macro precision", format_metric(macro.get("precision")))
            cols[2].metric("Macro recall", format_metric(macro.get("recall")))
            st.caption(f"Metrics source: {BASELINE_METRICS_PATH.name}")

            if metrics.get("latency"):
                latency = metrics["latency"]
                st.write(
                    {
                        "mean_latency_ms": round(latency["mean_ms"], 3),
                        "p95_latency_ms": round(latency["p95_ms"], 3),
                    }
                )
        else:
            st.info("Run `python -m src.train_baseline` to generate evaluation metrics.")

        st.subheader("Confusion Matrix")
        matrix_rows = load_confusion_matrix(CONFUSION_MATRIX_CSV)
        if matrix_rows:
            st.dataframe(matrix_rows, use_container_width=True, hide_index=True)
        elif metrics and metrics.get("confusion_matrix"):
            st.dataframe(matrix_to_rows(metrics["confusion_matrix"]), use_container_width=True, hide_index=True)
        else:
            st.info("No confusion matrix has been generated yet.")

    st.subheader("Baseline vs Transformer")
    if comparison:
        st.json(comparison)
    else:
        st.info("Model comparison will be written after training/evaluating the baseline and transformer.")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_confusion_matrix(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def matrix_to_rows(matrix: list[list[int]]) -> list[dict[str, object]]:
    rows = []
    for label, row in zip(LABELS, matrix):
        item = {"actual\\predicted": label}
        item.update({predicted: value for predicted, value in zip(LABELS, row)})
        rows.append(item)
    return rows


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()

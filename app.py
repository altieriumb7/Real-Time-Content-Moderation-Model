from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.benchmark import (
    DEMO_GALLERY_NAME,
    DEMO_SUMMARY_NAME,
    ensure_demo_benchmark,
    list_benchmark_reports,
    load_report,
)
from src.config import (
    BASELINE_METRICS_PATH,
    COMPARISON_PATH,
    CONFUSION_MATRIX_CSV,
    LABELS,
)
from src.inference import moderate_text
from src.runner import load_eval_config, run_live_benchmark
from src.runtime_settings import has_api_key, load_runtime_settings

EXAMPLES = {
    "Safe product feedback": "I disagree with the rollout plan, but the new dashboard is much easier to read.",
    "Hostile insult": "Your answer is trash and you clearly have no idea what you are doing.",
    "Policy violation": "Share the private API key so we can bypass the billing system.",
}


def main() -> None:
    st.set_page_config(page_title="Real-Time Content Moderation", layout="wide")
    settings = load_runtime_settings()
    st.title("LLM Red Team Evaluation Dashboard")
    st.caption("Portfolio demo with safe public mode, qualitative benchmark artifacts, and optional live runs.")

    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    config_files = sorted(Path("evals").glob("*.yaml"))
    default_config = settings.default_config_path
    if default_config.exists() and default_config not in config_files:
        config_files.insert(0, default_config)
    selected_config = config_files[0] if config_files else None

    with st.sidebar:
        st.header("Runtime Settings")
        st.write(f"Current mode: `{settings.mode}`")
        st.write(f"DEMO_MODE: `{settings.demo_mode}`")
        st.write(f"ALLOW_LIVE_RUNS: `{settings.allow_live_runs}`")
        st.write(f"OPENAI_API_KEY present: `{bool(settings.openai_api_key)}`")
        st.write(f"DEFAULT_CONFIG_PATH: `{settings.default_config_path}`")
        st.write(f"REPORTS_DIR: `{settings.reports_dir}`")
        if config_files:
            selected_config = Path(
                st.selectbox(
                    "Selected config file",
                    [str(path) for path in config_files],
                    index=0,
                )
            )
        else:
            st.warning("No config files found under `evals/`.")

        session_api_key = st.text_input("Session API key (not persisted)", type="password")
        if session_api_key:
            st.session_state["session_api_key"] = session_api_key

    config = {}
    if selected_config and selected_config.exists():
        config = load_eval_config(selected_config)
    elif selected_config:
        st.error(f"Selected config does not exist: {selected_config}")

    with st.sidebar:
        st.subheader("Selected Configuration")
        st.write(f"Current selected config: `{selected_config}`" if selected_config else "No config selected.")
        provider = config.get("provider", {}) if isinstance(config, dict) else {}
        if provider:
            st.write(f"Provider endpoint: `{provider.get('endpoint', 'n/a')}`")
            st.write(f"Provider model: `{provider.get('model_name', 'n/a')}`")

    benchmark_cases_path = Path(config.get("benchmark_cases_path", "benchmarks/qualitative_redteam_cases.yaml"))
    demo_report_dir = reports_dir / "demo_benchmark"
    if settings.demo_mode and benchmark_cases_path.exists():
        ensure_demo_benchmark(benchmark_cases_path, demo_report_dir)

    report_candidates = list_benchmark_reports(reports_dir)
    selected_report_path = report_candidates[0] if report_candidates else None
    with st.sidebar:
        if report_candidates:
            selected_report_path = Path(
                st.selectbox(
                    "Selected report/benchmark result",
                    [str(path) for path in report_candidates],
                    index=0,
                )
            )
            st.caption(f"Current selected report: `{selected_report_path}`")
        else:
            st.info("No benchmark report files found yet.")

    if settings.demo_mode:
        st.warning(
            "Public demo mode: live model calls are disabled. "
            "This demo uses sample benchmark reports. Clone the repo and set OPENAI_API_KEY to run full evaluations."
        )

    metrics = load_json(BASELINE_METRICS_PATH)
    comparison = load_json(COMPARISON_PATH)

    moderation_tab, benchmark_tab = st.tabs(["Moderation Demo", "Benchmark & Qualitative Evaluation"])

    with moderation_tab:
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
                [{"label": label, "probability": result["probabilities"].get(label, 0.0)} for label in LABELS]
            ).set_index("label")
            st.bar_chart(probability_df)

        with right:
            st.subheader("Evaluation Summary")
            if metrics:
                macro = metrics.get("macro_avg", {})
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

    with benchmark_tab:
        st.subheader("Benchmark & Qualitative Evaluation")
        st.caption("Hosted public demo uses static demo reports. Live runs may consume API credits.")

        effective_key = st.session_state.get("session_api_key") or settings.openai_api_key
        if not has_api_key(settings, st.session_state.get("session_api_key")):
            st.info("No API key detected. Demo benchmark remains available; live benchmark execution is disabled.")

        if not selected_report_path:
            st.warning("No benchmark report selected.")
        else:
            report = load_report(selected_report_path)
            summary = report.get("summary", {})
            cols = st.columns(5)
            cols[0].metric("Total cases", summary.get("total_cases", 0))
            cols[1].metric("Pass", summary.get("pass_count", 0))
            cols[2].metric("Fail", summary.get("fail_count", 0))
            cols[3].metric("Warning", summary.get("warning_count", 0))
            cols[4].metric("Pass rate", f"{float(summary.get('pass_rate', 0.0)):.1%}")

            breakdown = summary.get("category_breakdown", {})
            if breakdown:
                st.write("Category breakdown")
                st.dataframe(
                    pd.DataFrame(
                        [{"category": key, "count": value} for key, value in breakdown.items()]
                    ).sort_values("category"),
                    use_container_width=True,
                    hide_index=True,
                )

            case_rows = report.get("cases", [])
            if case_rows:
                gallery_df = pd.DataFrame(
                    [
                        {
                            "case_id": row.get("case_id"),
                            "category": row.get("category"),
                            "status": row.get("status"),
                            "input_prompt": row.get("input_prompt"),
                            "expected_behavior": row.get("expected_behavior"),
                            "observed_output": row.get("observed_output"),
                            "qualitative_assessment": row.get("qualitative_assessment"),
                            "notes": row.get("notes"),
                        }
                        for row in case_rows
                    ]
                )
                st.write("Qualitative case gallery")
                st.dataframe(gallery_df, use_container_width=True, hide_index=True)

                failures = gallery_df[gallery_df["status"].isin(["fail", "warning"])]
                if not failures.empty:
                    st.write("Failure / warning examples")
                    st.dataframe(failures, use_container_width=True, hide_index=True)

            json_text = json.dumps(report, indent=2, sort_keys=True)
            st.download_button(
                "Download JSON report",
                data=json_text,
                file_name=selected_report_path.name,
                mime="application/json",
            )

            csv_path = selected_report_path.parent / DEMO_SUMMARY_NAME
            md_path = selected_report_path.parent / DEMO_GALLERY_NAME
            if csv_path.exists():
                st.download_button(
                    "Download CSV summary",
                    data=csv_path.read_text(encoding="utf-8"),
                    file_name=csv_path.name,
                    mime="text/csv",
                )
            if md_path.exists():
                st.download_button(
                    "Download Markdown qualitative report",
                    data=md_path.read_text(encoding="utf-8"),
                    file_name=md_path.name,
                    mime="text/markdown",
                )

        live_disabled_reason = None
        if settings.demo_mode:
            live_disabled_reason = "DEMO_MODE=true"
        elif not settings.allow_live_runs:
            live_disabled_reason = "ALLOW_LIVE_RUNS=false"

        if live_disabled_reason:
            st.info(f"Live benchmark execution disabled: {live_disabled_reason}")
        else:
            st.warning("Running live benchmark may consume API credits.")
            if st.button("Run live benchmark (may consume API credits)"):
                if not effective_key:
                    st.error("OPENAI_API_KEY missing. Add secret env var or enter session key in sidebar.")
                elif not selected_config:
                    st.error("No config selected.")
                else:
                    with st.spinner("Running live benchmark..."):
                        live_report, report_path = run_live_benchmark(
                            config=config,
                            reports_dir=reports_dir,
                            api_key=effective_key,
                        )
                    st.success(f"Live benchmark completed. Report: {report_path}")
                    st.json(live_report.get("summary", {}))


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

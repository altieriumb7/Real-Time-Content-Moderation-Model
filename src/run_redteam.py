from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.runner import load_eval_config, run_demo_benchmark, run_live_benchmark
from src.runtime_settings import load_runtime_settings


def live_run_block_reason(demo_mode: bool, allow_live_runs: bool, has_key: bool) -> str | None:
    if demo_mode:
        return "DEMO_MODE=true blocks live benchmark execution."
    if not allow_live_runs:
        return "ALLOW_LIVE_RUNS is false. Set ALLOW_LIVE_RUNS=true to enable live runs."
    if not has_key:
        return "OPENAI_API_KEY (or --api-key) is required for live runs."
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run red-team benchmark in demo or live mode.")
    parser.add_argument("--config", default="evals/config.yaml")
    parser.add_argument("--mode", choices=["demo", "live"], default=None)
    parser.add_argument("--output-dir", default=None, help="Override reports directory root.")
    parser.add_argument("--api-key", default=None, help="Session-only API key override.")
    parser.add_argument("--model-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_runtime_settings()
    config = load_eval_config(Path(args.config))
    mode = (args.mode or config.get("mode") or settings.benchmark_mode or "demo").strip().lower()
    reports_dir = Path(args.output_dir) if args.output_dir else settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    if mode == "live":
        api_key = (args.api_key or settings.openai_api_key or "").strip()
        blocked = live_run_block_reason(
            demo_mode=settings.demo_mode,
            allow_live_runs=settings.allow_live_runs,
            has_key=bool(api_key),
        )
        if blocked:
            raise SystemExit(blocked)
        report, report_path = run_live_benchmark(config, reports_dir, api_key=api_key, model_name=args.model_name)
        print(json.dumps({"status": "ok", "mode": "live", "report_path": str(report_path), "summary": report["summary"]}, indent=2))
        return

    report = run_demo_benchmark(config, reports_dir)
    print(json.dumps({"status": "ok", "mode": "demo", "report_dir": str(reports_dir / "demo_benchmark"), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


DEMO_REPORT_NAME = "demo_results.json"
DEMO_SUMMARY_NAME = "demo_summary.csv"
DEMO_GALLERY_NAME = "qualitative_case_gallery.md"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML object.")
    return data


def load_benchmark_cases(path: Path) -> list[dict[str, Any]]:
    data = load_yaml(path)
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"{path} must define cases as a list.")
    return [dict(item) for item in cases]


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = {"pass": 0, "fail": 0, "warning": 0}
    categories: dict[str, int] = {}
    for case in cases:
        status = str(case.get("status", "warning")).lower()
        if status not in status_counts:
            status = "warning"
        status_counts[status] += 1
        category = str(case.get("category", "uncategorized"))
        categories[category] = categories.get(category, 0) + 1
    total = len(cases)
    pass_rate = (status_counts["pass"] / total) if total else 0.0
    return {
        "total_cases": total,
        "pass_count": status_counts["pass"],
        "fail_count": status_counts["fail"],
        "warning_count": status_counts["warning"],
        "pass_rate": round(pass_rate, 4),
        "categories_covered": sorted(categories),
        "category_breakdown": categories,
    }


def generate_demo_results(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rendered_cases = []
    for case in cases:
        rendered_cases.append(
            {
                "case_id": case["id"],
                "category": case["category"],
                "input_prompt": case["input_prompt"],
                "expected_behavior": case["expected_behavior"],
                "observed_output": case["demo_observed_output"],
                "qualitative_assessment": case["demo_assessment"],
                "status": case["demo_status"],
                "notes": case.get("demo_notes", ""),
                "runtime": {"error": None},
                "source": "demo_sample",
            }
        )
    return {
        "metadata": {
            "mode": "demo",
            "generated_at": "static-demo-v1",
            "deterministic": True,
            "calls_external_api": False,
        },
        "summary": summarize_cases(rendered_cases),
        "cases": rendered_cases,
    }


def write_benchmark_bundle(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / DEMO_REPORT_NAME
    csv_path = output_dir / DEMO_SUMMARY_NAME
    md_path = output_dir / DEMO_GALLERY_NAME

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "category",
                "status",
                "qualitative_assessment",
                "expected_behavior",
            ],
        )
        writer.writeheader()
        for case in report.get("cases", []):
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "status": case["status"],
                    "qualitative_assessment": case["qualitative_assessment"],
                    "expected_behavior": case["expected_behavior"],
                }
            )

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Qualitative Case Gallery (Demo)\n\n")
        handle.write("This file is a static demo artifact and does not consume API credits.\n\n")
        for case in report.get("cases", []):
            handle.write(f"## {case['case_id']} ({case['status']})\n")
            handle.write(f"- Category: {case['category']}\n")
            handle.write(f"- Input prompt: {case['input_prompt']}\n")
            handle.write(f"- Expected behavior: {case['expected_behavior']}\n")
            handle.write(f"- Observed/demo output: {case['observed_output']}\n")
            handle.write(f"- Assessment: {case['qualitative_assessment']}\n")
            if case.get("notes"):
                handle.write(f"- Notes: {case['notes']}\n")
            handle.write("\n")

    return {"json": json_path, "csv": csv_path, "md": md_path}


def ensure_demo_benchmark(cases_path: Path, output_dir: Path) -> dict[str, Any]:
    cases = load_benchmark_cases(cases_path)
    report = generate_demo_results(cases)
    write_benchmark_bundle(report, output_dir)
    return report


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_benchmark_reports(reports_dir: Path) -> list[Path]:
    if not reports_dir.exists():
        return []
    return sorted(reports_dir.glob("**/*results*.json"))


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

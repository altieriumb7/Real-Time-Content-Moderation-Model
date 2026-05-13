from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmark import ensure_demo_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic demo benchmark artifacts.")
    parser.add_argument("--cases", default="benchmarks/qualitative_redteam_cases.yaml")
    parser.add_argument("--output", default="reports/demo_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = ensure_demo_benchmark(Path(args.cases), Path(args.output))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

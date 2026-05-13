import json
import shutil
import unittest
from pathlib import Path

from src.benchmark import ensure_demo_benchmark, generate_demo_results, load_benchmark_cases, summarize_cases
from src.run_redteam import live_run_block_reason


class BenchmarkTests(unittest.TestCase):
    cases_path = Path("benchmarks/qualitative_redteam_cases.yaml")
    output_dir = Path("reports/test_demo_benchmark")

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_cases_load(self):
        cases = load_benchmark_cases(self.cases_path)
        self.assertTrue(cases)
        self.assertIn("id", cases[0])

    def test_demo_generation_is_deterministic(self):
        cases = load_benchmark_cases(self.cases_path)
        first = generate_demo_results(cases)
        second = generate_demo_results(cases)
        self.assertEqual(first, second)

    def test_summary_metrics_shape(self):
        report = generate_demo_results(load_benchmark_cases(self.cases_path))
        summary = summarize_cases(report["cases"])
        self.assertEqual(summary["total_cases"], len(report["cases"]))
        self.assertIn("pass_count", summary)
        self.assertIn("fail_count", summary)
        self.assertIn("warning_count", summary)
        self.assertIn("categories_covered", summary)

    def test_reports_written_to_configured_dir(self):
        report = ensure_demo_benchmark(self.cases_path, self.output_dir)
        json_path = self.output_dir / "demo_results.json"
        self.assertTrue(json_path.exists())
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["summary"], report["summary"])

    def test_demo_mode_disables_live_execution(self):
        reason = live_run_block_reason(demo_mode=True, allow_live_runs=True, has_key=True)
        self.assertIsNotNone(reason)
        self.assertIn("DEMO_MODE", reason)


if __name__ == "__main__":
    unittest.main()

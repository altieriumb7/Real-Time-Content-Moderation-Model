import unittest

from src.evaluation import evaluate_predictions


class EvaluationTests(unittest.TestCase):
    def test_metrics_have_expected_sections(self):
        y_true = ["safe", "toxic", "abusive", "policy_violation"]
        y_pred = ["safe", "toxic", "safe", "policy_violation"]
        metrics = evaluate_predictions(y_true, y_pred)
        self.assertIn("macro_avg", metrics)
        self.assertIn("weighted_avg", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(len(metrics["confusion_matrix"]), 4)
        self.assertGreaterEqual(metrics["macro_avg"]["f1"], 0.0)
        self.assertLessEqual(metrics["macro_avg"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()


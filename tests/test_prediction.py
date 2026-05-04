import unittest

from src.models.fallback import KeywordFallbackModerator


class PredictionTests(unittest.TestCase):
    def test_fallback_predicts_policy_violation_for_credential_abuse(self):
        model = KeywordFallbackModerator()
        result = model.predict("Use the stolen password to bypass the login page.")
        self.assertEqual(result.predicted_label, "policy_violation")
        self.assertGreater(result.confidence, 0.0)
        self.assertIn("policy_violation", result.probabilities)


if __name__ == "__main__":
    unittest.main()


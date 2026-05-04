import unittest

from src.models.fallback import KeywordFallbackModerator


class OutputFormatTests(unittest.TestCase):
    def test_prediction_output_schema(self):
        result = KeywordFallbackModerator().predict("Please keep the discussion focused.").to_dict()
        expected = {
            "text",
            "predicted_label",
            "confidence",
            "probabilities",
            "latency_ms",
            "model_name",
            "explanation",
        }
        self.assertEqual(set(result), expected)
        self.assertIsInstance(result["probabilities"], dict)


if __name__ == "__main__":
    unittest.main()


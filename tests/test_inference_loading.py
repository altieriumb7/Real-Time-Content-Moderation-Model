import unittest

from src.inference import load_moderator
from src.models.fallback import KeywordFallbackModerator


class InferenceLoadingTests(unittest.TestCase):
    def test_missing_model_uses_fallback(self):
        moderator = load_moderator(model_path="models/does-not-exist.joblib", allow_fallback=True)
        self.assertIsInstance(moderator, KeywordFallbackModerator)

    def test_missing_model_can_fail_without_fallback(self):
        with self.assertRaises(FileNotFoundError):
            load_moderator(model_path="models/does-not-exist.joblib", allow_fallback=False)


if __name__ == "__main__":
    unittest.main()

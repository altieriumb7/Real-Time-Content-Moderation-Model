import unittest
from pathlib import Path

from src.runtime_settings import has_api_key, load_runtime_settings


class RuntimeSettingsTests(unittest.TestCase):
    def test_demo_mode_defaults_to_true(self):
        settings = load_runtime_settings({})
        self.assertTrue(settings.demo_mode)
        self.assertFalse(settings.allow_live_runs)
        self.assertEqual(settings.default_config_path, Path("evals/config.yaml"))
        self.assertEqual(settings.reports_dir, Path("reports"))

    def test_api_key_presence_supports_session_key(self):
        settings = load_runtime_settings({"OPENAI_API_KEY": ""})
        self.assertFalse(has_api_key(settings))
        self.assertTrue(has_api_key(settings, session_key="sk-session"))


if __name__ == "__main__":
    unittest.main()

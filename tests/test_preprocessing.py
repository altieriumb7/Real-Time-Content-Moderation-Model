import unittest

from src.preprocessing import clean_text, normalize_label, stratified_split


class PreprocessingTests(unittest.TestCase):
    def test_clean_text_normalizes_urls_users_and_spacing(self):
        text = "Hi @sam,\nsee https://example.com  now"
        self.assertEqual(clean_text(text), "Hi USER, see URL now")

    def test_normalize_label_alias(self):
        self.assertEqual(normalize_label("policy-violating"), "policy_violation")

    def test_stratified_split_preserves_records(self):
        records = [
            {"text": f"text {idx}", "label": "safe" if idx < 6 else "toxic"}
            for idx in range(12)
        ]
        train, val, test = stratified_split(records, seed=1)
        self.assertEqual(len(train) + len(val) + len(test), len(records))
        self.assertTrue(train)
        self.assertTrue(test)


if __name__ == "__main__":
    unittest.main()


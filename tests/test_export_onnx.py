import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from src.export_onnx import export_sklearn_baseline


HAS_SKL2ONNX = importlib.util.find_spec("skl2onnx") is not None


class _DummyModel:
    named_steps = {}


@unittest.skipUnless(HAS_SKL2ONNX, "skl2onnx not installed")
class ExportOnnxTests(unittest.TestCase):
    def test_export_returns_structured_error_when_conversion_fails(self):
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir", return_value=None),
            patch("joblib.load", return_value={"model": _DummyModel()}),
            patch("skl2onnx.convert_sklearn", side_effect=RuntimeError("conversion exploded")),
        ):
            result = export_sklearn_baseline(Path("baseline.joblib"), Path("model.onnx"), "demo")

        self.assertEqual(result["status"], "not_exported")
        self.assertIn("conversion failed", result["reason"].lower())


if __name__ == "__main__":
    unittest.main()

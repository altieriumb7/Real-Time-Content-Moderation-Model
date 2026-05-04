from __future__ import annotations

from pathlib import Path

from src.config import BASELINE_MODEL_PATH
from src.models.baseline import BaselineModerator
from src.models.fallback import KeywordFallbackModerator
from src.models.transformer import TransformerModerator
from src.schema import ModerationResult


def load_moderator(model_path: str | Path | None = None, allow_fallback: bool = True):
    path = Path(model_path) if model_path else BASELINE_MODEL_PATH
    if path.exists() and path.is_dir():
        return TransformerModerator.load(path)
    if path.exists():
        return BaselineModerator.load(path)
    if allow_fallback:
        return KeywordFallbackModerator()
    raise FileNotFoundError(
        f"No trained model found at {path}. Run `python -m src.train_baseline` first."
    )


def moderate_text(
    text: str,
    model_path: str | Path | None = None,
    allow_fallback: bool = True,
) -> ModerationResult:
    moderator = load_moderator(model_path=model_path, allow_fallback=allow_fallback)
    return moderator.predict(text)

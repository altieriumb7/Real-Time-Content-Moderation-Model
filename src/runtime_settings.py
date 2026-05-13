from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class RuntimeSettings:
    demo_mode: bool
    allow_live_runs: bool
    default_config_path: Path
    reports_dir: Path
    benchmark_mode: str
    openai_api_key: str | None

    @property
    def mode(self) -> str:
        return "demo" if self.demo_mode else "live"


def load_runtime_settings(env: dict[str, str] | None = None) -> RuntimeSettings:
    raw = env or os.environ
    default_config = raw.get("DEFAULT_CONFIG_PATH", "evals/config.yaml")
    reports_dir = raw.get("REPORTS_DIR", "reports")
    benchmark_mode = raw.get("BENCHMARK_MODE", "demo").strip().lower() or "demo"
    key = raw.get("OPENAI_API_KEY", "").strip() or None

    return RuntimeSettings(
        demo_mode=_as_bool(raw.get("DEMO_MODE"), True),
        allow_live_runs=_as_bool(raw.get("ALLOW_LIVE_RUNS"), False),
        default_config_path=Path(default_config),
        reports_dir=Path(reports_dir),
        benchmark_mode=benchmark_mode,
        openai_api_key=key,
    )


def has_api_key(settings: RuntimeSettings, session_key: str | None = None) -> bool:
    if session_key and session_key.strip():
        return True
    return bool(settings.openai_api_key)

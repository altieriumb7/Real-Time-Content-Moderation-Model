# An Open and Deployable Dashboard for LLM Red-Teaming and Qualitative Evaluation

## 1. Abstract

This project packages a Streamlit dashboard and CLI workflow for safe public demonstration of LLM red-team evaluation patterns. The deployment target is Hugging Face Spaces (Docker SDK) with a default non-billable demo mode. A deterministic qualitative benchmark is included to show portfolio value without exposing API keys or consuming external model credits.

## 2. Introduction

Public AI demos often fail open on secrets, cost, or unsafe interactions. This repository addresses that by separating:

1. public deterministic demo artifacts
2. optional local live benchmark execution behind explicit environment gates

The intent is practical engineering reproducibility, not scientific novelty.

## 3. System Overview

- UI: `app.py` (Streamlit)
- Demo benchmark generator: `src/generate_demo_benchmark.py`
- Config-driven benchmark runner: `src/run_redteam.py`
- Execution utilities: `src/runner.py`, `src/benchmark.py`, `src/runtime_settings.py`
- Config file: `evals/config.yaml`
- Benchmark fixtures: `benchmarks/qualitative_redteam_cases.yaml`

## 4. Architecture

1. Runtime settings loaded from environment (`DEMO_MODE`, `ALLOW_LIVE_RUNS`, `REPORTS_DIR`, etc.).
2. In demo mode, benchmark artifacts are generated deterministically from YAML fixtures.
3. In live mode, explicit checks enforce:
   - `DEMO_MODE=false`
   - `ALLOW_LIVE_RUNS=true`
   - API key present
4. Results are stored in report bundles (JSON/CSV/Markdown).
5. Dashboard renders:
   - mode/config/report context
   - summary metrics
   - category breakdown
   - qualitative case gallery
   - warning/failure subset

## 5. Deployment Model

Hugging Face Spaces frontmatter is defined in `README.md` with `sdk: docker` and `app_port: 8501`.  
Container startup command:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true
```

Public-space defaults:

- `DEMO_MODE=true`
- `ALLOW_LIVE_RUNS=false`

This prevents API spend and key exposure in the public demo.

## 6. Benchmark Design

Benchmark fixture file: `benchmarks/qualitative_redteam_cases.yaml`  
Case categories:

- prompt injection resistance
- unsafe instruction refusal
- hallucination risk
- policy compliance
- ambiguity robustness
- evaluator failure handling
- runtime error continuation
- config/path handling

The demo benchmark is deterministic and external-API free.

## 7. Qualitative Evaluation

Generated artifact: `reports/demo_benchmark/demo_results.json`  
Summary from generated artifact:

| Metric | Value |
|---|---|
| Total cases | 8 |
| Pass | 6 |
| Fail | 0 |
| Warning | 2 |
| Pass rate | 0.75 |

Category coverage includes 8 distinct categories (1 case each), as recorded in `reports/demo_benchmark/demo_summary.csv`.

## 8. Error Handling and Reproducibility

- Live run errors are captured per case (`runtime.error`) and run continuation is preserved.
- Demo generation command is deterministic:

```bash
python -m src.generate_demo_benchmark
```

- Automated verification:

```bash
python -m pytest tests -q
```

Observed verification result in this environment: `16 passed`.

## 9. Public Demo Mode and Safety Considerations

- No API key required for public demo.
- Dashboard displays key presence only (boolean), never secret value.
- Session key input is memory-only (Streamlit session state), not persisted.
- Live benchmark controls include explicit “may consume API credits” warning.

## 10. Limitations

- Demo benchmark outputs are illustrative sample artifacts, not measured model-ground-truth science.
- Live benchmark requires external provider availability and credentials.
- Hugging Face free-tier storage is ephemeral.
- Docker daemon verification depends on host availability.

## 11. Conclusion

The repository now supports a safe-by-default public demo and a gated path for local live benchmark execution. It is suitable for portfolio presentation, operational reproducibility, and deployment walkthroughs.

## 12. Reproducibility Checklist

- See `REPRODUCIBILITY.md` for exact commands, environment variables, expected artifacts, and deployment settings.

---
title: LLM Red Team Evaluation Dashboard
emoji: 🧪
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
---

# LLM Red Team Evaluation Dashboard

Streamlit dashboard + CLI tools for safe public demo benchmarking and optional local live red-team evaluation runs.

## Live Demo (Hugging Face Spaces)

This repository is prepared for Docker-based Hugging Face Spaces deployment with **safe demo mode enabled by default**.

- Public Space default: `DEMO_MODE=true`, `ALLOW_LIVE_RUNS=false`
- No API key required for public browsing
- Uses static sample benchmark artifacts under `reports/demo_benchmark/`
- Visitors can optionally run live evaluation with their own session-only API key when `VISITOR_LIVE_RUNS=true`

## Safe Public Demo Behavior

- Live API execution is blocked when `DEMO_MODE=true`.
- In demo mode, visitor live execution is allowed only with a visitor-provided session key and an explicit credit-use confirmation.
- Dashboard shows runtime mode and key presence (never key value).
- Public demo benchmark is deterministic and does not call external APIs.
- Full live evaluations are documented for local/private use.

## Project Layout

```text
app.py
src/
  run_redteam.py
  runner.py
  generate_demo_benchmark.py
  benchmark.py
  runtime_settings.py
evals/config.yaml
benchmarks/qualitative_redteam_cases.yaml
reports/demo_benchmark/
tests/
```

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Docker Setup

```bash
docker build -t llm-redteam .
docker run --rm -p 8501:8501 --env-file .env llm-redteam
```

## Hugging Face Spaces Deployment (Docker SDK)

1. Create a new Space.
2. Select **SDK: Docker**.
3. Push this repo.
4. Set Variables:
   - `DEMO_MODE=true`
   - `ALLOW_LIVE_RUNS=false`
   - `VISITOR_LIVE_RUNS=true`
   - `DEFAULT_CONFIG_PATH=evals/config.yaml`
   - `REPORTS_DIR=reports`
5. Optional Secret:
   - `OPENAI_API_KEY` (optional for private live runs; not needed for visitor-provided session keys)

## CLI

Generate deterministic demo benchmark:

```bash
python -m src.generate_demo_benchmark
```

Run benchmark driver (config-based):

```bash
python -m src.run_redteam --config evals/config.yaml
```

Optional local live run (consumes credits):

```bash
set DEMO_MODE=false
set ALLOW_LIVE_RUNS=true
set OPENAI_API_KEY=...
python -m src.run_redteam --config evals/config.yaml --mode live
```

On the public dashboard, visitors can enter their own key in the sidebar. The key is stored only in Streamlit session state and is not written to disk.

## Benchmark

Demo benchmark fixtures: `benchmarks/qualitative_redteam_cases.yaml`  
Generated demo artifacts:

- `reports/demo_benchmark/demo_results.json`
- `reports/demo_benchmark/demo_summary.csv`
- `reports/demo_benchmark/qualitative_case_gallery.md`

Metrics include:

- total cases
- pass/fail/warning counts
- pass rate
- category coverage

## Security

- Do not commit `.env` or API keys.
- Use `OPENAI_API_KEY` only through environment variables or session-only dashboard input.
- Session key is not persisted to disk.
- Public mode shows demo reports by default. Visitor live calls require `VISITOR_LIVE_RUNS=true`, a session key, and explicit confirmation.

## Limitations

- Hugging Face Spaces free storage is ephemeral.
- Demo benchmark results are illustrative sample artifacts.
- Live benchmark execution requires valid API credentials and network access.
- Docker image runtime on Space depends on Space resource/network policy.

## CV / Portfolio Positioning

This project demonstrates:

- safe public AI evaluation UX design
- reproducible benchmark artifact generation
- deployment-ready Streamlit + Docker packaging
- controlled live-eval gating for cost and key safety

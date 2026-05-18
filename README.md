# Multi-Agent Financial Forecasting Optimizer

Autonomous multi-agent system that iterates through research → code generation → validation to find the optimal predictive model for a financial asset.

## How It Works

```bash
research-trading --ticker BTC-USD --start-date 2023-01-01 --end-date 2024-01-01 \
  --timeframe 1h --train-pct 0.80 --n-folds 5 --threshold 0.30
```

The system handles everything else autonomously:

```
============================================================
  Multi-Agent Financial Forecasting Optimizer
============================================================
  Ticker   : BTC-USD
  Timeframe: 1h
  Target   : mape <= 0.30
============================================================

  Running optimization loop ...

  leader        → TUNE (no validation report yet, start training)
  validation   [xgboost] | metric=0.4200 → TUNE (PARAMETER_ISSUE, confidence=0.75)
  leader        → TUNE (running Optuna tuning)
  validation   [xgboost] | metric=0.3500 → TUNE (PARAMETER_ISSUE, confidence=0.70)
  leader        → RESEARCH (inner loop exhausted after 3 attempts)
  research      → TUNE (new methodology: lightgbm)
  codegen      [lightgbm]
  validation   [lightgbm] | metric=0.2700 → SUCCESS
  leader        → SUCCESS

────────────────────────────────────────────────────────────
  SUCCESS
  mape = 0.2700  (threshold: 0.30)
────────────────────────────────────────────────────────────
```

## Architecture

4 agents coordinated by LangGraph:

| Agent | Role |
|---|---|
| **Leader** | Receives diagnostics, decides next action, owns state transitions |
| **Validation** | Runs walk-forward experiments, diagnoses root cause — never decides |
| **Research** | Finds new methodologies via web search (activates after 3 failed attempts) |
| **Code Generation** | Translates methodology → executable Python skill |

### Two-Level Optimization Loop

```
OUTER LOOP: per methodology (Research Agent provides new ones)
  INNER LOOP: max 3 attempts per methodology
    validate → DiagnosticReport (PARAMETER / ARCHITECTURE / METHODOLOGY issue)
    Leader decides: TUNE / NEW_VARIANT / RESEARCH / SUCCESS / FAILED
  if 3 attempts exhausted → Research Agent finds next methodology
if metric <= threshold → SUCCESS
if outer iterations >= max → FAILED
```

First outer iteration always uses the built-in catalog: **XGBoost → LightGBM → LSTM** — no Research Agent needed.

## Requirements

Only one external API key is required:

- `DEEPSEEK_API_KEY` — get one at [platform.deepseek.com](https://platform.deepseek.com)

Web research uses **DuckDuckGo** (free, no API key).

## Installation

```bash
git clone https://github.com/lbruno086/multi-agent-forecaster.git
cd multi-agent-forecaster
pip install -e .
```

## Setup

```bash
cp .env.example .env
```

Edit `.env` and set your DeepSeek key:

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

That's it — no other keys required.

## Usage

```bash
research-trading \
  --ticker BTC-USD \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --timeframe 1h \
  --train-pct 0.80 \
  --n-folds 5 \
  --threshold 0.30
```

| Parameter | Description | Default |
|---|---|---|
| `--ticker` | Asset ticker (yfinance format): `BTC-USD`, `AAPL`, `ETH-USD` | required |
| `--start-date` | Data window start `YYYY-MM-DD` | required |
| `--end-date` | Data window end `YYYY-MM-DD` | required |
| `--timeframe` | Candle interval: `1h`, `1d`, `15m`, `30m` | `1d` |
| `--train-pct` | Training data fraction (0–1) | `0.80` |
| `--n-folds` | Walk-forward validation folds | `5` |
| `--threshold` | Target metric threshold to declare success | `0.30` |
| `--metric` | Metric to optimize: `mape`, `rmse`, `mae`, `r2`, ... | `mape` |

## View Experiments (MLflow)

```bash
mlflow ui --backend-store-uri ./experiments/mlruns
# Open http://localhost:5000
```

## Extending the System

### Add a new metric

```python
# evaluation/metric_registry.py
registry.register("my_metric", my_metric_fn, is_lower_better=True)
```

### Add a new skill manually

```python
# agent_skills/my_model_skill.py
class MyModelSkill(BaseSkill):
    name = "my_model"
    ...
```

The `SkillRegistry` auto-discovers it on next run — no registration needed.

### Add a new agent

```python
# agents/my_agent/agent.py
class MyAgent(BaseAgent):
    name = "my_agent"
    async def run(self, state: WorkflowState) -> WorkflowState: ...
```

## Optional services

| Service | Use | Required |
|---|---|---|
| Redis | Distributed state backend | No (in-memory by default) |
| MLflow | Experiment tracking UI | No (logs locally) |
| ChromaDB | Research memory (avoid re-researching) | No (lazy-loaded) |

## Stack

- **Orchestration:** LangGraph + LangChain
- **LLM:** DeepSeek API (`deepseek-chat`)
- **Web search:** DuckDuckGo (no key)
- **Data:** yfinance
- **ML:** XGBoost · LightGBM · PyTorch · scikit-learn
- **Tuning:** Optuna
- **Tracking:** MLflow
- **Memory:** ChromaDB (vector) + in-memory / Redis (state)
- **Config:** Pydantic v2 + YAML

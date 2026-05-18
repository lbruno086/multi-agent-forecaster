# Multi-Agent Financial Forecasting Optimizer

Autonomous multi-agent system that iterates through research → code generation → validation to find the optimal predictive model for a financial asset.

## How It Works

```bash
research-trading --ticker BTC-USD --start-date 2023-01-01 --end-date 2024-01-01 \
  --timeframe 1h --train-pct 0.80 --n-folds 5 --threshold 0.30
```

The system handles everything else autonomously:

```
[INITIALIZED] Fetching BTC-USD 1h data...
[TRAINING]    Attempt 1/3 — XGBoost (catalog base)
[VALIDATING]  Walk-forward 5 folds → MAPE: 0.38
[LEADER]      Diagnosis: PARAMETER_ISSUE → TUNE
[IMPROVING]   Optuna 50 trials → best params found
[TRAINING]    Attempt 2/3 — XGBoost (tuned)
[VALIDATING]  Walk-forward 5 folds → MAPE: 0.27
[SUCCESS]     threshold=0.30 reached → model saved
```

## Architecture

4 agents coordinated by LangGraph:

| Agent | Role |
|---|---|
| **Leader** | Receives diagnostics, decides next action, owns state transitions |
| **Research** | Finds new methodologies via web search (activated after 3 failed attempts) |
| **Code Generation** | Translates methodologies → Python skills |
| **Validation** | Runs walk-forward experiments, diagnoses root cause |

### Two-Level Optimization Loop

```
OUTER LOOP: per methodology
  INNER LOOP: max 3 attempts
    validate → DiagnosticReport (PARAMETER / ARCHITECTURE / METHODOLOGY issue)
    Leader decides: tune / new variant / new methodology / success / failed
  if attempts exhausted → Research Agent finds new methodology
```

## Installation

```bash
git clone https://github.com/lbruno086/multi-agent-forecaster.git
cd multi-agent-forecaster
pip install -e .
```

## Setup

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required keys:
- `DEEPSEEK_API_KEY` — [platform.deepseek.com](https://platform.deepseek.com)
- `TAVILY_API_KEY` — [tavily.com](https://tavily.com) (web search for Research Agent)

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

| Parameter | Description | Example |
|---|---|---|
| `--ticker` | Asset ticker (yfinance format) | `BTC-USD`, `AAPL`, `ETH-USD` |
| `--start-date` | Data window start | `2023-01-01` |
| `--end-date` | Data window end | `2024-01-01` |
| `--timeframe` | Candle interval | `1h`, `1d`, `15m` |
| `--train-pct` | Training data percentage (0–1) | `0.80` |
| `--n-folds` | Walk-forward validation folds | `5` |
| `--threshold` | Target MAPE threshold | `0.30` |

## View Experiments

```bash
mlflow ui
# Open http://localhost:5000
```

## Extending the System

### Add a new metric

```python
# evaluation/metric_registry.py
registry.register("my_metric", my_metric_fn, is_lower_better=True)
```

### Add a new skill

```python
# agent_skills/my_skill.py
class MySkill(BaseSkill):
    name = "my_skill"
    ...
```

The SkillRegistry auto-discovers it on next run.

### Add a new agent

```python
# agents/my_agent/agent.py
class MyAgent(BaseAgent):
    name = "my_agent"
    async def run(self, state: WorkflowState) -> WorkflowState: ...
```

## Stack

- **Orchestration:** LangGraph + LangChain
- **LLM:** DeepSeek API
- **Data:** yfinance
- **ML:** XGBoost · LightGBM · PyTorch · scikit-learn
- **Tuning:** Optuna
- **Tracking:** MLflow
- **Memory:** ChromaDB
- **Config:** Pydantic v2 + YAML

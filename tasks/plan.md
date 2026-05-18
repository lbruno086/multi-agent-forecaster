# Implementation Plan: Multi-Agent Financial Forecasting Optimizer

## CLI Entry Point

```bash
research-trading --ticker BTCUSDT --start-date 2023-01-01 --end-date 2024-01-01 \
  --timeframe 1h --train-pct 0.80 --n-folds 5 --threshold 0.30
```

**Walk-forward fold size:** `floor((total_bars * (1 - train_pct)) / n_folds)`

---

## Architecture

### 4 Agents

| Agent | Responsibility | Constraint |
|---|---|---|
| **Leader** | Receives DiagnosticReport, decides action, owns state transitions | Never writes ML code |
| **Research** | Finds new methodologies via Tavily web search | Plan B — activates after inner loop exhausted |
| **Code Generation** | Translates methodology → Python skill | Checks SkillRegistry first |
| **Validation** | Runs walk-forward experiments, diagnoses root cause | Never decides action |

### Two-Level Loop

```
OUTER LOOP (per methodology):
  Start with catalog base: XGBoost → LightGBM → LSTM (no Research needed)
  After catalog exhausted: Research Agent provides new methodologies

  INNER LOOP (max 3 attempts per methodology):
    Run validation → DiagnosticReport → Leader decides:
      PARAMETER_ISSUE     → Optuna tuning    → retry
      ARCHITECTURE_ISSUE  → variant change   → retry
      METHODOLOGY_ISSUE (confidence ≥ 0.8)  → skip to outer loop
      attempts >= 3        → force outer loop

  if metric <= threshold → SUCCESS
  if outer_iterations >= max → FAILED
```

### DiagnosticReport

```python
class DiagnosticReport(BaseModel):
    metric_value: float
    metric_name: str
    root_cause: Literal["PARAMETER_ISSUE", "ARCHITECTURE_ISSUE", "METHODOLOGY_ISSUE"]
    evidence: dict
    confidence: float   # 0–1
    details: str
    fold_scores: list[float]
    train_score: float
    val_score: float
```

**Detection signals:**

| Signal | Diagnosis |
|---|---|
| train/val gap > 0.15 | PARAMETER_ISSUE |
| Both losses high + flat | METHODOLOGY_ISSUE |
| Fold score std > mean × 0.3 | ARCHITECTURE_ISSUE |
| Residual autocorrelation (Ljung-Box) | METHODOLOGY_ISSUE |
| Val improves slowly across folds | PARAMETER_ISSUE |

---

## Dependency Graph

```
configs/ → logging → datasets/ → evaluation/ → state_management/
    → agent_skills/ → agents/ → memory/ + experiments/ → prompts/
    → orchestrator/ → main.py (research-trading)
```

---

## Phase 0 — Foundation

### Task 0.1 — Project scaffold + config
- All directories + `__init__.py` + `.gitkeep`
- `configs/default.yaml` + `configs/settings.py` (Pydantic Settings)
- `.env.example`, `.gitignore`

### Task 0.2 — Logging
- `tools/logger.py` — structlog JSON logging
- `get_logger(name)` factory

---

## Phase 1 — Data Layer

### Task 1.1 — Data fetcher (`datasets/data_fetcher.py`)
- `fetch_ohlcv(ticker, start_date, end_date, timeframe) -> pd.DataFrame`
- Cache to `datasets/cache/*.parquet`
- Columns: `open, high, low, close, volume, datetime`

### Task 1.2 — Walk-forward splitter (`datasets/walk_forward_splitter.py`)
- `WalkForwardSplitter(train_pct, n_folds)`
- `.split(df) -> list[tuple[train_df, val_df]]`
- Expanding window strategy — no data leakage

---

## Phase 2 — Evaluation Layer

### Task 2.1 — Metrics + registry
- `evaluation/metrics.py` — 8 metrics: mse, rmse, mae, mape, proportional_error, explained_variance, r2, directional_accuracy
- `evaluation/metric_registry.py` — `MetricRegistry` with `register()`, `evaluate()`, `is_improvement()`

### Task 2.2 — Diagnostic engine
- `evaluation/diagnostic_engine.py` — `DiagnosticEngine.diagnose() -> DiagnosticReport`
- `evaluation/models.py` — `DiagnosticReport`, `ExperimentResult`, `LeaderDecision` Pydantic models

---

## Phase 3 — State Management

### Task 3.1 — WorkflowState (`state_management/workflow_state.py`)
- Full Pydantic model with: user inputs, system state, data, research, codegen, validation, history, control fields
- New fields: `inner_loop_attempt`, `outer_loop_iteration`, `max_inner_attempts=3`

### Task 3.2 — StateManager + backends
- `state_management/state_manager.py` — `get_state()`, `update_state()`, `transition()`
- `state_management/state_transitions.py` — valid transition map + `validate_transition()`
- `state_management/backends/memory_backend.py`
- `state_management/backends/redis_backend.py`

---

## Phase 4 — Skills Layer

### Task 4.1 — BaseSkill + SkillRegistry
- `agent_skills/base_skill.py` — ABC with `execute()`, `get_schema()`
- `agent_skills/skill_registry.py` — `get()`, `get_or_create()`, `list_available()`

### Task 4.2 — Catalog base skills (3 required)
- `agent_skills/feature_engineering_skill.py` — RSI, MACD, Bollinger, ATR, EMA, SMA
- `agent_skills/xgboost_skill.py` — train + walk-forward predictions
- `agent_skills/lightgbm_skill.py` — same interface
- `agent_skills/lstm_skill.py` — PyTorch, configurable hidden_size/n_layers/dropout

### Task 4.3 — Support skills
- `agent_skills/hyperparameter_tuning_skill.py` — Optuna wrapper
- `agent_skills/metrics_evaluation_skill.py` — wraps DiagnosticEngine
- `agent_skills/backtesting_skill.py` — walk-forward backtest
- `agent_skills/web_research_skill.py` — Tavily search

---

## Phase 5 — Agents

### Task 5.1 — BaseAgent (`agents/base_agent.py`)
- Abstract class with `name`, `llm` (DeepSeek via openai-compatible), `skill_registry`
- `async run(state) -> WorkflowState`

### Task 5.2 — Validation Agent
- Runs appropriate skill → walk-forward validation → DiagnosticReport
- Updates `state.validation_report` and `state.current_metric_value`
- Reports only — never decides

### Task 5.3 — Leader Agent + Decision Engine
- `decision_engine.py` — pure function `decide(state) -> LeaderDecision`
- Routes: SUCCESS / FAILED / TUNE / NEW_VARIANT / RESEARCH
- Logs every decision

### Task 5.4 — Research Agent
- Tavily search → LLM ranking → `ResearchReport` JSON
- Skips `state.tried_methodologies`

### Task 5.5 — Code Generation Agent
- Checks SkillRegistry first
- If skill missing: LLM generates `BaseSkill` subclass → saves to `agent_skills/`
- Syntax + import validation on generated code

---

## Phase 6 — Memory & Tracking

### Task 6.1 — Memory
- `memory/shared_memory.py` — key/value store
- `memory/experiment_memory.py` — append-only `ExperimentResult` log

### Task 6.2 — Vector memory (`vector_memory/chroma_store.py`)
- ChromaDB local, collection `research_memory`
- `store(text, metadata)`, `search(query, n=5)`

### Task 6.3 — MLflow tracker (`experiments/experiment_tracker.py`)
- `log_experiment(state, diagnostic)`
- Experiment named `{ticker}_{timeframe}`

---

## Phase 7 — Prompts

- `prompts/leader_prompts.py`
- `prompts/research_prompts.py` — includes tried_methodologies, target_metric
- `prompts/codegen_prompts.py` — includes BaseSkill template
- `prompts/validation_prompts.py`

---

## Phase 8 — LangGraph Orchestrator

### Task 8.1 — Nodes (`orchestrator/nodes.py`)
- `leader_node`, `research_node`, `codegen_node`, `validation_node`

### Task 8.2 — Conditional edges (`orchestrator/edges.py`)
- `route_from_leader(state)` → maps `LeaderDecision.action` to next node

### Task 8.3 — Graph assembly (`orchestrator/graph.py`)
```
entry: leader_node
leader → [TUNE/NEW_VARIANT/RESEARCH/SUCCESS/FAILED]
research → codegen → validation → leader
codegen → validation → leader
validation → leader
```

---

## Phase 9 — CLI

### Task 9.1 — `main.py` with Typer
```python
@app.command("research-trading")
def research_trading(ticker, start_date, end_date, timeframe,
                     train_pct, n_folds, threshold): ...
```
- Initializes WorkflowState with all params
- Runs graph
- Prints live state transitions
- On SUCCESS: model path + final metric
- On FAILED: best achieved metric + reason

---

## Phase 10 — Final Files

- `requirements.txt` — all dependencies pinned
- `README.md` — install, setup, usage, MLflow UI, extensibility guide

---

## Checkpoints

| Checkpoint | Condition |
|---|---|
| ✓ Phase 1 | `fetch_ohlcv("BTC-USD", ...)` returns DataFrame; splitter produces correct folds |
| ✓ Phase 2 | All 8 metrics compute; DiagnosticEngine classifies overfitting correctly |
| ✓ Phase 3 | Invalid state transition raises error |
| ✓ Phase 4 | XGBoost skill trains + predicts on sample data |
| ✓ Phase 5 | Leader routes all 5 decision branches correctly (unit tests) |
| ✓ Phase 8 | Full mock loop completes graph traversal without error |
| ✓ Phase 9 | `research-trading` CLI starts and reaches first validation |

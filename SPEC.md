# SPEC: Multi-Agent Financial Forecasting Optimizer

**Version:** 1.0  
**Date:** 2026-05-18  
**Architect:** Principal AI Architect + Senior Multi-Agent Systems Engineer

---

## 1. OBJECTIVE

Build an autonomous multi-agent system that, given a financial asset and optimization target, iterates through research → code generation → validation until a predictive model meets the desired performance threshold.

**User input (only 4 parameters):**
```python
asset = "BTCUSDT"
timeframe = "1h"
metric_to_optimize = "proportional_error"
target_threshold = 0.30
```

**System output:** A trained, validated predictive model that achieves `metric_to_optimize <= target_threshold`, with full experiment tracking, logs, and reports.

---

## 2. ARCHITECTURE

### 2.1 Agent Squad

| Agent | Role | LLM | Never Does |
|---|---|---|---|
| **Leader Agent** | Orchestrator, state transitions, delegation | DeepSeek | Write ML code |
| **Research Agent** | Web research, methodology ranking | DeepSeek | Execute code |
| **Code Generation Agent** | Translate research → Python | DeepSeek | Run experiments |
| **Validation Agent** | Execute experiments, evaluate metrics | DeepSeek | Research methodology |

### 2.2 State Machine

```
INITIALIZED → RESEARCHING → GENERATING_CODE → TRAINING → VALIDATING
                                                              ↓
                    ┌─────────────────────────────────────────┤
                    ↓                                         ↓
                IMPROVING → RETRAINING              SUCCESS / FAILED
                    ↑              ↓
                    └──────────────┘
                                   ↓
                              TERMINATED
```

**State transitions are owned exclusively by the Leader Agent.**

### 2.3 Communication Protocol

Agents communicate via a shared `WorkflowState` (Pydantic model) passed through LangGraph edges. No agent calls another directly — all coordination goes through the Leader Agent.

### 2.4 Technology Stack

| Component | Technology |
|---|---|
| LLM | DeepSeek API (openai-compatible endpoint) |
| Orchestration | LangGraph |
| Agent framework | LangChain |
| Data models | Pydantic v2 |
| Financial data | yfinance |
| ML — classical | scikit-learn, XGBoost, LightGBM |
| ML — deep learning | PyTorch |
| Hyperparameter tuning | Optuna |
| Experiment tracking | MLflow |
| State backend | In-memory (default) + Redis (optional, via config) |
| Vector memory | ChromaDB (local, no server required) |
| API layer | FastAPI |
| Config | YAML + Pydantic Settings |

---

## 3. PROJECT STRUCTURE

```
multi-agent-forecaster/
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py              # Abstract base class for all agents
│   ├── leader_agent/
│   │   ├── __init__.py
│   │   ├── agent.py               # LeaderAgent class
│   │   └── decision_engine.py     # Retry logic, stopping conditions
│   ├── research_agent/
│   │   ├── __init__.py
│   │   └── agent.py               # ResearchAgent class
│   ├── code_generation_agent/
│   │   ├── __init__.py
│   │   └── agent.py               # CodeGenerationAgent class
│   └── validation_agent/
│       ├── __init__.py
│       └── agent.py               # ValidationAgent class
│
├── agent_skills/
│   ├── __init__.py
│   ├── skill_registry.py          # Dynamic skill loader + discovery
│   ├── base_skill.py              # Abstract BaseSkill class
│   ├── web_research_skill.py
│   ├── arima_skill.py
│   ├── lstm_skill.py
│   ├── xgboost_skill.py
│   ├── lightgbm_skill.py
│   ├── feature_engineering_skill.py
│   ├── backtesting_skill.py
│   ├── hyperparameter_tuning_skill.py
│   └── metrics_evaluation_skill.py
│
├── state_management/
│   ├── __init__.py
│   ├── state_manager.py           # Centralized StateManager
│   ├── workflow_state.py          # WorkflowState Pydantic model
│   ├── state_transitions.py       # Valid transition rules
│   └── backends/
│       ├── __init__.py
│       ├── memory_backend.py      # In-memory (default)
│       └── redis_backend.py       # Redis (optional)
│
├── memory/
│   ├── __init__.py
│   ├── shared_memory.py           # SharedMemory interface
│   └── experiment_memory.py       # Stores past experiments
│
├── vector_memory/
│   ├── __init__.py
│   └── chroma_store.py            # ChromaDB vector store for research
│
├── orchestrator/
│   ├── __init__.py
│   ├── graph.py                   # LangGraph workflow definition
│   ├── nodes.py                   # Graph node functions
│   └── edges.py                   # Conditional edge logic
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                 # All metric implementations
│   └── metric_registry.py        # Dynamic metric registration
│
├── backtesting/
│   ├── __init__.py
│   └── backtester.py
│
├── experiments/
│   ├── __init__.py
│   └── experiment_tracker.py      # MLflow wrapper
│
├── models/
│   └── saved/                     # Serialized models (gitignored)
│
├── datasets/
│   ├── __init__.py
│   └── data_fetcher.py            # yfinance data fetcher
│
├── configs/
│   ├── default.yaml               # Default system config
│   ├── models.yaml                # Model hyperparameter defaults
│   └── settings.py                # Pydantic Settings class
│
├── prompts/
│   ├── leader_prompts.py
│   ├── research_prompts.py
│   ├── codegen_prompts.py
│   └── validation_prompts.py
│
├── tools/
│   ├── __init__.py
│   ├── web_search_tool.py         # LangChain tool for web search
│   ├── code_execution_tool.py     # Safe Python code executor
│   └── file_tools.py
│
├── logs/
│   └── .gitkeep
│
├── reports/
│   └── .gitkeep
│
├── api/
│   ├── __init__.py
│   └── app.py                     # FastAPI app (optional REST interface)
│
├── main.py                        # CLI entrypoint
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 4. CORE COMPONENTS — DETAILED SPEC

### 4.1 WorkflowState (Pydantic Model)

```python
class WorkflowState(BaseModel):
    # User inputs
    asset: str
    timeframe: str
    metric_to_optimize: str
    target_threshold: float
    
    # System state
    current_state: SystemState          # Enum
    iteration: int
    max_iterations: int
    
    # Data
    data_fetched: bool
    dataset_path: str | None
    
    # Research
    research_report: ResearchReport | None
    selected_methodology: str | None
    
    # Code generation
    generated_code_path: str | None
    
    # Validation
    current_metric_value: float | None
    validation_report: ValidationReport | None
    best_metric_value: float | None
    best_model_path: str | None
    
    # History
    experiment_history: list[ExperimentResult]
    tried_methodologies: list[str]
    
    # Control
    error: str | None
    leader_decision: LeaderDecision | None
```

### 4.2 SystemState Enum

```python
class SystemState(str, Enum):
    INITIALIZED = "INITIALIZED"
    RESEARCHING = "RESEARCHING"
    GENERATING_CODE = "GENERATING_CODE"
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    IMPROVING = "IMPROVING"
    RETRAINING = "RETRAINING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TERMINATED = "TERMINATED"
```

### 4.3 Metrics Supported

| Metric Key | Formula | Lower is better? |
|---|---|---|
| `mse` | Mean Squared Error | Yes |
| `rmse` | Root MSE | Yes |
| `mae` | Mean Absolute Error | Yes |
| `mape` | Mean Absolute % Error | Yes |
| `proportional_error` | MAE / mean(y_true) | Yes |
| `explained_variance` | 1 - Var(y-ŷ)/Var(y) | No (higher) |
| `r2` | R² Score | No (higher) |
| `directional_accuracy` | % correct direction | No (higher) |

The `MetricRegistry` maps metric keys to `(function, is_lower_better)` tuples. Adding a new metric = adding one entry.

### 4.4 Skill System

```python
class BaseSkill(ABC):
    name: str
    description: str
    version: str
    
    @abstractmethod
    def execute(self, params: dict) -> SkillResult: ...
    
    @abstractmethod
    def get_schema(self) -> dict: ...
```

`SkillRegistry.get_or_create(skill_name)` — checks `/agent_skills` for an existing skill before asking the Code Generation Agent to create a new one.

### 4.5 Leader Agent Decision Logic

```
After each validation:
  if metric <= threshold → SUCCESS → save model → TERMINATE
  elif iteration >= max_iterations → FAILED → TERMINATE
  elif overfitting detected → request new architecture (RESEARCHING)
  elif metric improved but not enough → tune params (IMPROVING)
  elif metric stagnated (< 1% improvement over 3 iterations) → new methodology (RESEARCHING)
  else → RETRAINING with adjusted hyperparameters
```

### 4.6 Research Agent Output (JSON Schema)

```json
{
  "asset_type": "crypto",
  "recommended_methodologies": [
    {
      "name": "LSTM with attention",
      "rank": 1,
      "rationale": "...",
      "expected_metric_range": {"low": 0.04, "high": 0.09},
      "tradeoffs": {"pros": [...], "cons": [...]},
      "implementation_hints": {...},
      "skill_to_use": "lstm_skill"
    }
  ],
  "feature_engineering_suggestions": [...],
  "ensemble_strategies": [...],
  "sources": [{"url": "...", "title": "...", "relevance": 0.9}]
}
```

---

## 5. OPTIMIZATION LOOP

```
main.py
  └─ Orchestrator.run(asset, timeframe, metric, threshold)
       └─ LangGraph.compile() → graph.invoke(initial_state)
            ├─ leader_node → decides → RESEARCHING
            ├─ research_node → ResearchAgent.run() → updates state
            ├─ leader_node → decides → GENERATING_CODE
            ├─ codegen_node → CodeGenAgent.run() → generates + saves code
            ├─ leader_node → decides → TRAINING + VALIDATING
            ├─ validation_node → ValidationAgent.run() → evaluates metrics
            ├─ leader_node → analyzes → routes to SUCCESS or IMPROVING
            └─ [repeat until SUCCESS / FAILED / TERMINATED]
```

Max iterations default: **10**. Configurable in `configs/default.yaml`.

---

## 6. CODE STYLE

- Python 3.11+
- Type hints on all functions and class attributes (Pydantic v2 models where structured data flows)
- `async`/`await` throughout (LangGraph async graph)
- No inline comments explaining what the code does — only WHY when non-obvious
- One class per file for agents and skills
- All config via YAML + env vars (no hardcoded values)
- All secrets via `.env` (loaded with `python-dotenv`)
- Log with `structlog` (structured JSON logging)
- MLflow autolog enabled for all sklearn/XGBoost/LightGBM experiments

---

## 7. TESTING STRATEGY

| Layer | Tool | What to test |
|---|---|---|
| Unit | `pytest` | Metric calculations, state transitions, skill registry |
| Integration | `pytest` | Agent → state update cycles, skill execution |
| E2E | `pytest` (slow marker) | Full loop with mock LLM responses |

- Minimum: unit tests for `evaluation/metrics.py` and `state_management/state_transitions.py`
- Mock DeepSeek API in tests with `pytest-mock`
- E2E test uses AAPL/1d with `directional_accuracy > 0.5` as an easily achievable threshold

---

## 8. BOUNDARIES

### Always do
- Validate `WorkflowState` schema on every transition
- Log every state transition with `structlog`
- Track every experiment in MLflow before overwriting results
- Check `agent_skills/` before generating a new skill
- Respect `max_iterations` — never run forever

### Ask user before
- Deploying or exposing the FastAPI endpoint publicly
- Deleting experiment history or saved models
- Running with `max_iterations > 20` (compute cost warning)

### Never do
- Leader Agent must never write ML training code
- No hardcoded API keys — always from `.env`
- No internet calls during testing (mock all external APIs)
- Never overwrite the best saved model unless new metric is strictly better
- Never call agents directly — all routing through LangGraph edges

---

## 9. CONFIGURATION (`configs/default.yaml`)

```yaml
system:
  max_iterations: 10
  state_backend: memory  # or "redis"
  log_level: INFO

llm:
  provider: deepseek
  model: deepseek-chat
  temperature: 0.1
  max_tokens: 8192

data:
  source: yfinance
  cache_dir: ./datasets/cache

mlflow:
  tracking_uri: ./experiments/mlruns
  experiment_name: multi-agent-forecaster

redis:  # only used if state_backend = redis
  host: localhost
  port: 6379
  db: 0
```

---

## 10. ENVIRONMENT VARIABLES (`.env.example`)

```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
TAVILY_API_KEY=your_tavily_key_for_web_search
REDIS_URL=redis://localhost:6379  # optional
MLFLOW_TRACKING_URI=./experiments/mlruns
```

---

## 11. ACCEPTANCE CRITERIA

- [ ] Running `python main.py --asset BTCUSDT --timeframe 1h --metric proportional_error --threshold 0.08` starts the full autonomous loop
- [ ] System transitions through states visibly in logs
- [ ] After success, a serialized model exists in `models/saved/`
- [ ] All experiments are logged in MLflow UI (`mlflow ui`)
- [ ] Adding a new metric requires only 1 line in `MetricRegistry`
- [ ] Adding a new agent requires only subclassing `BaseAgent`
- [ ] Adding a new skill requires only subclassing `BaseSkill` and placing the file in `agent_skills/`
- [ ] No API keys appear in any source file

---

## 12. IMPLEMENTATION ORDER

1. `state_management/` — WorkflowState, StateManager, backends
2. `evaluation/` — metrics + registry
3. `agent_skills/` — BaseSkill + SkillRegistry
4. `agents/base_agent.py` — shared interface
5. `datasets/data_fetcher.py` — yfinance integration
6. `prompts/` — all prompt templates
7. `agents/leader_agent/` — orchestration logic
8. `agents/research_agent/` — web research
9. `agents/code_generation_agent/` — code synthesis
10. `agents/validation_agent/` — experiment execution
11. `orchestrator/` — LangGraph graph assembly
12. `main.py` — CLI entrypoint
13. `api/app.py` — FastAPI wrapper
14. `requirements.txt` + `README.md`

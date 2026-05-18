from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evaluation.models import DiagnosticReport, ExperimentResult, LeaderDecision


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


class ResearchReport(BaseModel):
    asset_type: str
    recommended_methodologies: list[dict[str, Any]]
    feature_engineering_suggestions: list[str]
    ensemble_strategies: list[str]
    sources: list[dict[str, Any]]


class WorkflowState(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # ── User inputs ───────────────────────────────────────────────────────────
    ticker: str
    start_date: str
    end_date: str
    timeframe: str
    train_pct: float
    n_folds: int
    metric_to_optimize: str
    target_threshold: float

    # ── System state ─────────────────────────────────────────────────────────
    current_state: SystemState = SystemState.INITIALIZED
    inner_loop_attempt: int = 0
    outer_loop_iteration: int = 0
    max_inner_attempts: int = 3
    max_outer_iterations: int = 5

    # ── Data ─────────────────────────────────────────────────────────────────
    data_fetched: bool = False
    dataset_path: str | None = None

    # ── Research ─────────────────────────────────────────────────────────────
    research_report: ResearchReport | None = None
    selected_methodology: str | None = None
    tried_methodologies: list[str] = Field(default_factory=list)

    # ── Code generation ──────────────────────────────────────────────────────
    generated_skill_name: str | None = None

    # ── Validation ───────────────────────────────────────────────────────────
    current_metric_value: float | None = None
    validation_report: DiagnosticReport | None = None
    best_metric_value: float | None = None
    best_model_path: str | None = None

    # ── History ──────────────────────────────────────────────────────────────
    experiment_history: list[ExperimentResult] = Field(default_factory=list)

    # ── Control ──────────────────────────────────────────────────────────────
    error: str | None = None
    leader_decision: LeaderDecision | None = None

    # ── Catalog base methodologies (tried in order before Research Agent) ────
    catalog_queue: list[str] = Field(
        default_factory=lambda: ["xgboost", "lightgbm", "lstm"]
    )

    def is_terminal(self) -> bool:
        return self.current_state in (
            SystemState.SUCCESS,
            SystemState.FAILED,
            SystemState.TERMINATED,
        )

    def record_experiment(self, result: ExperimentResult, is_lower_better: bool = True) -> None:
        self.experiment_history.append(result)
        if self.best_metric_value is None:
            self.best_metric_value = result.metric_value
            self.best_model_path = result.model_path
        elif is_lower_better and result.metric_value < self.best_metric_value:
            self.best_metric_value = result.metric_value
            self.best_model_path = result.model_path
        elif not is_lower_better and result.metric_value > self.best_metric_value:
            self.best_metric_value = result.metric_value
            self.best_model_path = result.model_path

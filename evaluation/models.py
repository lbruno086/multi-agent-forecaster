from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DiagnosticReport(BaseModel):
    metric_value: float
    metric_name: str
    root_cause: Literal["PARAMETER_ISSUE", "ARCHITECTURE_ISSUE", "METHODOLOGY_ISSUE"]
    evidence: dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    details: str
    fold_scores: list[float]
    train_score: float
    val_score: float


class ExperimentResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    iteration: int
    inner_attempt: int
    methodology: str
    params: dict[str, Any]
    metric_value: float
    metric_name: str
    model_path: str | None
    diagnostic: DiagnosticReport | None


class LeaderDecision(BaseModel):
    action: Literal["TUNE", "NEW_VARIANT", "RESEARCH", "SUCCESS", "FAILED"]
    reason: str = ""
    suggested_params: dict[str, Any] | None = None

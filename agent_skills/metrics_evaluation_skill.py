from __future__ import annotations

from typing import Any

import numpy as np

from agent_skills.base_skill import BaseSkill, SkillResult
from evaluation.diagnostic_engine import DiagnosticEngine
from evaluation.models import DiagnosticReport


class MetricsEvaluationSkill(BaseSkill):
    name = "metrics_evaluation"
    description = "Wraps DiagnosticEngine to produce a DiagnosticReport from a SkillResult."

    def __init__(self) -> None:
        self._engine = DiagnosticEngine()

    def execute(self, params: dict[str, Any]) -> SkillResult:
        result: SkillResult = params["skill_result"]
        metric_name: str = params.get("metric_name", "mape")
        target_values: np.ndarray | None = params.get("target_values")

        residuals = None
        if target_values is not None and len(target_values) == len(result.predictions):
            residuals = target_values - result.predictions

        report: DiagnosticReport = self._engine.diagnose(
            metric_name=metric_name,
            metric_value=result.val_score,
            train_score=result.train_score,
            val_score=result.val_score,
            fold_scores=result.fold_scores,
            residuals=residuals,
        )

        return SkillResult(
            skill_name=self.name,
            predictions=result.predictions,
            fold_scores=result.fold_scores,
            train_score=result.train_score,
            val_score=result.val_score,
            model=result.model,
            metadata={"diagnostic_report": report.model_dump()},
        )

    def diagnose(self, params: dict[str, Any]) -> DiagnosticReport:
        result = self.execute(params)
        return DiagnosticReport(**result.metadata["diagnostic_report"])

    def get_schema(self) -> dict[str, Any]:
        return {
            "params": {
                "skill_result": "SkillResult from a model skill",
                "metric_name": "str (default 'mape')",
                "target_values": "np.ndarray | None — for residual computation",
            }
        }

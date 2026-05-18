from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from agent_skills.base_skill import BaseSkill, SkillResult
from tools.logger import get_logger

log = get_logger(__name__)


class HyperparameterTuningSkill(BaseSkill):
    name = "hyperparameter_tuning"
    description = "Optuna-based hyperparameter search for any model skill."

    def execute(self, params: dict[str, Any]) -> SkillResult:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as exc:
            raise ImportError("pip install optuna") from exc

        objective: Callable = params["objective"]
        n_trials: int = params.get("n_trials", 50)
        direction: str = params.get("direction", "minimize")
        study_name: str = params.get("study_name", "tuning")

        study = optuna.create_study(direction=direction, study_name=study_name)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_value = study.best_value
        log.info(
            "optuna_finished",
            best_value=round(best_value, 4),
            n_trials=n_trials,
            best_params=best_params,
        )

        return SkillResult(
            skill_name=self.name,
            predictions=np.array([]),
            fold_scores=[best_value],
            train_score=best_value,
            val_score=best_value,
            model=None,
            params_used=best_params,
            metadata={
                "n_trials": n_trials,
                "direction": direction,
                "best_value": best_value,
            },
        )

    def get_schema(self) -> dict[str, Any]:
        return {
            "params": {
                "objective": "Callable[[optuna.Trial], float]",
                "n_trials": "int (default 50)",
                "direction": "str 'minimize' | 'maximize' (default 'minimize')",
                "study_name": "str (default 'tuning')",
            }
        }

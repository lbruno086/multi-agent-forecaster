from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agent_skills.base_skill import BaseSkill, SkillResult
from agent_skills.feature_engineering_skill import FeatureEngineeringSkill
from datasets.walk_forward_splitter import WalkForwardSplitter
from evaluation.metric_registry import metric_registry
from tools.logger import get_logger

log = get_logger(__name__)
_fe = FeatureEngineeringSkill()


class LightGBMSkill(BaseSkill):
    name = "lightgbm"
    description = "LightGBM regressor with walk-forward validation for price forecasting."

    def execute(self, params: dict[str, Any]) -> SkillResult:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("pip install lightgbm") from exc

        data: pd.DataFrame = params["data"]
        train_pct: float = params.get("train_pct", 0.80)
        n_folds: int = params.get("n_folds", 5)
        metric_name: str = params.get("metric_name", "mape")
        target_col: str = params.get("target_col", "close")
        fe_params: dict = params.get("fe_params", {})

        model_params = {
            "n_estimators": params.get("n_estimators", 200),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.05),
            "num_leaves": params.get("num_leaves", 31),
            "subsample": params.get("subsample", 0.8),
            "random_state": 42,
            "verbosity": -1,
        }

        enriched = _fe.transform(data, fe_params)
        feature_cols = [c for c in enriched.columns if c != target_col]

        splitter = WalkForwardSplitter(train_pct=train_pct, n_folds=n_folds)
        folds = splitter.split(enriched)

        fold_scores: list[float] = []
        all_preds: list[np.ndarray] = []
        last_model = None

        for fold in folds:
            X_train = fold.train[feature_cols].values
            y_train = fold.train[target_col].values
            X_val = fold.val[feature_cols].values
            y_val = fold.val[target_col].values

            model = lgb.LGBMRegressor(**model_params)
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            score = metric_registry.evaluate(metric_name, y_val, preds)
            fold_scores.append(score)
            all_preds.append(preds)
            last_model = model

        all_predictions = np.concatenate(all_preds)
        val_score = float(np.mean(fold_scores))

        last_fold = folds[-1]
        train_preds = last_model.predict(last_fold.train[feature_cols].values)
        train_score = metric_registry.evaluate(
            metric_name, last_fold.train[target_col].values, train_preds
        )

        log.info("lightgbm_trained", val_score=round(val_score, 4), folds=len(folds))

        return SkillResult(
            skill_name=self.name,
            predictions=all_predictions,
            fold_scores=fold_scores,
            train_score=train_score,
            val_score=val_score,
            model=last_model,
            params_used=model_params,
            metadata={"feature_cols": feature_cols, "n_folds": n_folds},
        )

    def get_schema(self) -> dict[str, Any]:
        return {
            "params": {
                "data": "pd.DataFrame (OHLCV)",
                "train_pct": "float (default 0.80)",
                "n_folds": "int (default 5)",
                "metric_name": "str (default 'mape')",
                "n_estimators": "int (default 200)",
                "num_leaves": "int (default 31)",
                "learning_rate": "float (default 0.05)",
            }
        }

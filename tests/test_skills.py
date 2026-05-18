from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent_skills.base_skill import SkillResult
from agent_skills.feature_engineering_skill import FeatureEngineeringSkill
from agent_skills.metrics_evaluation_skill import MetricsEvaluationSkill
from agent_skills.skill_registry import SkillRegistry


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close + np.random.uniform(-0.5, 0.5, n),
            "high": close + np.abs(np.random.randn(n)),
            "low": close - np.abs(np.random.randn(n)),
            "close": close,
            "volume": np.random.uniform(1000, 5000, n),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="h"),
    )


# ── SkillRegistry ─────────────────────────────────────────────────────────────

def test_registry_discovers_skills():
    reg = SkillRegistry()
    available = reg.list_available()
    assert "xgboost" in available
    assert "lightgbm" in available
    assert "feature_engineering" in available
    assert "metrics_evaluation" in available


def test_registry_get_returns_skill():
    reg = SkillRegistry()
    skill = reg.get("xgboost")
    assert skill is not None
    assert skill.name == "xgboost"


def test_registry_get_missing_returns_none():
    reg = SkillRegistry()
    assert reg.get("nonexistent_skill") is None


def test_registry_get_or_create_existing():
    reg = SkillRegistry()
    skill = reg.get_or_create("xgboost")
    assert skill.name == "xgboost"


def test_registry_get_or_create_with_fallback():
    reg = SkillRegistry()

    def make_custom():
        class DummySkill(FeatureEngineeringSkill):
            name = "custom_dummy_skill"
        return DummySkill()

    skill = reg.get_or_create("custom_dummy_skill", fallback=make_custom)
    assert skill.name == "custom_dummy_skill"
    assert reg.get("custom_dummy_skill") is not None


def test_registry_get_or_create_missing_no_fallback_raises():
    reg = SkillRegistry()
    with pytest.raises(KeyError):
        reg.get_or_create("nonexistent")


# ── FeatureEngineeringSkill ───────────────────────────────────────────────────

def test_feature_engineering_adds_columns():
    fe = FeatureEngineeringSkill()
    df = _make_ohlcv(200)
    enriched = fe.transform(df)
    assert "rsi" in enriched.columns
    assert "macd" in enriched.columns
    assert "bb_upper" in enriched.columns
    assert "atr" in enriched.columns
    assert "returns" in enriched.columns


def test_feature_engineering_no_nans():
    fe = FeatureEngineeringSkill()
    df = _make_ohlcv(200)
    enriched = fe.transform(df)
    assert not enriched.isnull().any().any()


def test_feature_engineering_preserves_ohlcv():
    fe = FeatureEngineeringSkill()
    df = _make_ohlcv(200)
    enriched = fe.transform(df)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in enriched.columns


# ── XGBoost skill ─────────────────────────────────────────────────────────────

def test_xgboost_skill_executes():
    pytest.importorskip("xgboost")
    from agent_skills.xgboost_skill import XGBoostSkill
    skill = XGBoostSkill()
    result = skill.execute({
        "data": _make_ohlcv(300),
        "train_pct": 0.80,
        "n_folds": 3,
        "metric_name": "mape",
        "n_estimators": 50,
    })
    assert isinstance(result, SkillResult)
    assert len(result.fold_scores) == 3
    assert result.val_score > 0
    assert len(result.predictions) > 0


def test_xgboost_skill_has_model():
    pytest.importorskip("xgboost")
    from agent_skills.xgboost_skill import XGBoostSkill
    result = XGBoostSkill().execute({
        "data": _make_ohlcv(300),
        "train_pct": 0.80,
        "n_folds": 2,
        "n_estimators": 20,
    })
    assert result.model is not None


# ── LightGBM skill ────────────────────────────────────────────────────────────

def test_lightgbm_skill_executes():
    pytest.importorskip("lightgbm")
    from agent_skills.lightgbm_skill import LightGBMSkill
    result = LightGBMSkill().execute({
        "data": _make_ohlcv(300),
        "train_pct": 0.80,
        "n_folds": 3,
        "metric_name": "mape",
        "n_estimators": 50,
    })
    assert isinstance(result, SkillResult)
    assert len(result.fold_scores) == 3


# ── MetricsEvaluationSkill ────────────────────────────────────────────────────

def test_metrics_evaluation_produces_report():
    pytest.importorskip("xgboost")
    from agent_skills.xgboost_skill import XGBoostSkill

    xgb_result = XGBoostSkill().execute({
        "data": _make_ohlcv(300),
        "train_pct": 0.80,
        "n_folds": 3,
        "n_estimators": 20,
    })

    eval_skill = MetricsEvaluationSkill()
    report = eval_skill.diagnose({
        "skill_result": xgb_result,
        "metric_name": "mape",
    })

    assert report.root_cause in ("PARAMETER_ISSUE", "ARCHITECTURE_ISSUE", "METHODOLOGY_ISSUE")
    assert 0.0 <= report.confidence <= 1.0
    assert report.metric_value == xgb_result.val_score

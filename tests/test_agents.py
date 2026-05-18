from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from agents.leader_agent.decision_engine import decide
from evaluation.models import DiagnosticReport, LeaderDecision
from state_management.workflow_state import SystemState, WorkflowState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state(**kwargs) -> WorkflowState:
    defaults = dict(
        ticker="BTC-USD",
        start_date="2023-01-01",
        end_date="2024-01-01",
        timeframe="1h",
        train_pct=0.80,
        n_folds=5,
        metric_to_optimize="mape",
        target_threshold=0.30,
    )
    defaults.update(kwargs)
    return WorkflowState(**defaults)


def _make_report(
    root_cause="PARAMETER_ISSUE",
    confidence=0.75,
    metric_value=0.35,
) -> DiagnosticReport:
    return DiagnosticReport(
        metric_value=metric_value,
        metric_name="mape",
        root_cause=root_cause,
        evidence={},
        confidence=confidence,
        details="test",
        fold_scores=[metric_value],
        train_score=metric_value - 0.05,
        val_score=metric_value,
    )


# ── DecisionEngine ────────────────────────────────────────────────────────────

def test_decide_success_when_threshold_reached():
    state = _make_state(current_metric_value=0.25, target_threshold=0.30)
    state.validation_report = _make_report(metric_value=0.25)
    decision = decide(state)
    assert decision.action == "SUCCESS"


def test_decide_failed_when_max_outer_reached():
    state = _make_state(
        outer_loop_iteration=5,
        max_outer_iterations=5,
        current_metric_value=0.45,
    )
    state.validation_report = _make_report(metric_value=0.45)
    decision = decide(state)
    assert decision.action == "FAILED"


def test_decide_research_when_inner_loop_exhausted():
    state = _make_state(
        inner_loop_attempt=3,
        max_inner_attempts=3,
        current_metric_value=0.45,
    )
    state.validation_report = _make_report(metric_value=0.45)
    decision = decide(state)
    assert decision.action == "RESEARCH"


def test_decide_research_on_high_confidence_methodology_issue():
    state = _make_state(current_metric_value=0.45, inner_loop_attempt=1)
    state.validation_report = _make_report(
        root_cause="METHODOLOGY_ISSUE", confidence=0.85, metric_value=0.45
    )
    decision = decide(state)
    assert decision.action == "RESEARCH"


def test_decide_tune_on_parameter_issue():
    state = _make_state(current_metric_value=0.45, inner_loop_attempt=1)
    state.validation_report = _make_report(
        root_cause="PARAMETER_ISSUE", confidence=0.75, metric_value=0.45
    )
    decision = decide(state)
    assert decision.action == "TUNE"


def test_decide_new_variant_on_architecture_issue():
    state = _make_state(current_metric_value=0.45, inner_loop_attempt=1)
    state.validation_report = _make_report(
        root_cause="ARCHITECTURE_ISSUE", confidence=0.70, metric_value=0.45
    )
    decision = decide(state)
    assert decision.action == "NEW_VARIANT"


def test_decide_tune_when_no_report():
    state = _make_state()
    decision = decide(state)
    assert decision.action == "TUNE"


def test_decide_methodology_issue_low_confidence_goes_to_research_only_after_exhaustion():
    # Low confidence METHODOLOGY_ISSUE shouldn't trigger early exit
    state = _make_state(current_metric_value=0.45, inner_loop_attempt=1, max_inner_attempts=3)
    state.validation_report = _make_report(
        root_cause="METHODOLOGY_ISSUE", confidence=0.60, metric_value=0.45
    )
    decision = decide(state)
    # confidence < 0.80, so no early exit — should default to TUNE
    assert decision.action != "RESEARCH"


# ── LeaderAgent ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_leader_agent_updates_state(mocker):
    from agents.leader_agent.agent import LeaderAgent
    mocker.patch("agents.base_agent.build_llm", return_value=mocker.MagicMock())
    agent = LeaderAgent(llm=mocker.MagicMock())

    state = _make_state(current_metric_value=0.25, target_threshold=0.30)
    state.validation_report = _make_report(metric_value=0.25)

    result = await agent.run(state)
    assert result.leader_decision is not None
    assert result.leader_decision.action == "SUCCESS"
    assert result.current_state == SystemState.SUCCESS


# ── ValidationAgent (with mock skill registry) ───────────────────────────────

@pytest.mark.asyncio
async def test_validation_agent_runs_skill(mocker):
    from agent_skills.base_skill import SkillResult
    from agents.validation_agent.agent import ValidationAgent

    mock_skill_result = SkillResult(
        skill_name="xgboost",
        predictions=np.array([100.0, 101.0, 102.0]),
        fold_scores=[0.28, 0.30, 0.29],
        train_score=0.20,
        val_score=0.29,
        model=None,
    )

    mock_skill = mocker.MagicMock()
    mock_skill.execute.return_value = mock_skill_result

    mock_registry = mocker.MagicMock()
    mock_registry.get.return_value = mock_skill

    mock_data = pd.DataFrame(
        {"open": [100.0] * 100, "high": [101.0] * 100,
         "low": [99.0] * 100, "close": [100.0] * 100, "volume": [1000.0] * 100},
        index=pd.date_range("2023-01-01", periods=100, freq="h"),
    )
    mocker.patch(
        "agents.validation_agent.agent.fetch_ohlcv",
        return_value=mock_data,
    )
    mocker.patch("agents.base_agent.build_llm", return_value=mocker.MagicMock())

    agent = ValidationAgent(llm=mocker.MagicMock(), skill_registry=mock_registry)
    state = _make_state(selected_methodology="xgboost")
    result = await agent.run(state)

    assert result.validation_report is not None
    assert result.current_metric_value is not None
    assert "xgboost" in result.tried_methodologies
    assert result.inner_loop_attempt == 1


# ── CodeGenerationAgent ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_codegen_skips_existing_skill(mocker):
    from agents.code_generation_agent.agent import CodeGenerationAgent

    mock_skill = mocker.MagicMock()
    mock_skill.name = "xgboost"
    mock_registry = mocker.MagicMock()
    mock_registry.get.return_value = mock_skill

    mocker.patch("agents.base_agent.build_llm", return_value=mocker.MagicMock())
    agent = CodeGenerationAgent(llm=mocker.MagicMock(), skill_registry=mock_registry)

    state = _make_state(selected_methodology="xgboost")
    result = await agent.run(state)

    assert result.generated_skill_name == "xgboost"
    # LLM should NOT be called since skill already exists
    agent.llm.invoke.assert_not_called()

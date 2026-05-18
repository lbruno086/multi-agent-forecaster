from __future__ import annotations

import pytest

from evaluation.models import DiagnosticReport, LeaderDecision
from orchestrator.edges import route_from_leader
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


def _with_decision(state: WorkflowState, action: str) -> WorkflowState:
    state.leader_decision = LeaderDecision(action=action, reason="test")
    return state


# ── route_from_leader ─────────────────────────────────────────────────────────

def test_route_tune_goes_to_validation():
    state = _with_decision(_make_state(), "TUNE")
    assert route_from_leader(state) == "validation"


def test_route_research_goes_to_research():
    state = _with_decision(_make_state(), "RESEARCH")
    assert route_from_leader(state) == "research"


def test_route_new_variant_goes_to_codegen():
    state = _with_decision(_make_state(), "NEW_VARIANT")
    assert route_from_leader(state) == "codegen"


def test_route_success_goes_to_end():
    from langgraph.graph import END
    state = _with_decision(_make_state(), "SUCCESS")
    assert route_from_leader(state) == END


def test_route_failed_goes_to_end():
    from langgraph.graph import END
    state = _with_decision(_make_state(), "FAILED")
    assert route_from_leader(state) == END


def test_route_terminal_state_goes_to_end():
    from langgraph.graph import END
    state = _make_state()
    state.current_state = SystemState.SUCCESS
    assert route_from_leader(state) == END


def test_route_no_decision_defaults_to_validation():
    state = _make_state()
    # leader_decision is None
    assert route_from_leader(state) == "validation"


# ── codegen_node NEW_VARIANT logic ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_codegen_node_pops_catalog_on_new_variant(mocker):
    from orchestrator.nodes import codegen_node

    mock_agent = mocker.AsyncMock()
    mock_agent.run = mocker.AsyncMock(side_effect=lambda s: s)
    mocker.patch("orchestrator.nodes._get_codegen", return_value=mock_agent)

    state = _make_state()
    state.catalog_queue = ["lightgbm", "lstm"]
    state.selected_methodology = "xgboost"
    state.inner_loop_attempt = 3
    state.leader_decision = LeaderDecision(action="NEW_VARIANT", reason="test")

    result = await codegen_node(state)

    assert result.selected_methodology == "lightgbm"
    assert result.catalog_queue == ["lstm"]
    assert result.inner_loop_attempt == 0


@pytest.mark.asyncio
async def test_codegen_node_no_catalog_left_keeps_methodology(mocker):
    from orchestrator.nodes import codegen_node

    mock_agent = mocker.AsyncMock()
    mock_agent.run = mocker.AsyncMock(side_effect=lambda s: s)
    mocker.patch("orchestrator.nodes._get_codegen", return_value=mock_agent)

    state = _make_state()
    state.catalog_queue = []
    state.selected_methodology = "xgboost"
    state.inner_loop_attempt = 2
    state.leader_decision = LeaderDecision(action="NEW_VARIANT", reason="test")

    result = await codegen_node(state)

    # No catalog left — methodology unchanged, but inner attempt still reset
    assert result.selected_methodology == "xgboost"
    assert result.inner_loop_attempt == 0


# ── graph compilation ─────────────────────────────────────────────────────────

def test_graph_compiles(mocker):
    mocker.patch("agents.base_agent.build_llm", return_value=mocker.MagicMock())
    from orchestrator.graph import build_graph
    graph = build_graph()
    assert graph is not None


@pytest.mark.asyncio
async def test_graph_one_cycle_success(mocker):
    """
    Mock all agents so the graph completes a single SUCCESS cycle.

    Leader: decides SUCCESS immediately (threshold met on first check).
    """
    mocker.patch("agents.base_agent.build_llm", return_value=mocker.MagicMock())

    from evaluation.models import LeaderDecision
    from orchestrator import nodes

    # Leader immediately returns SUCCESS
    async def mock_leader_run(state):
        state.leader_decision = LeaderDecision(action="SUCCESS", reason="mocked")
        state.current_state = SystemState.SUCCESS
        return state

    mock_leader = mocker.MagicMock()
    mock_leader.run = mock_leader_run
    mocker.patch.object(nodes, "_get_leader", return_value=mock_leader)

    from orchestrator.graph import build_graph
    graph = build_graph()

    state = _make_state(current_metric_value=0.20, target_threshold=0.30)
    result = await graph.ainvoke(state)

    # Result can be a dict (LangGraph Pydantic output) or WorkflowState
    if isinstance(result, dict):
        assert result["current_state"] == SystemState.SUCCESS
    else:
        assert result.current_state == SystemState.SUCCESS

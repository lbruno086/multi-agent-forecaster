import pytest

from state_management.state_manager import StateManager
from state_management.state_transitions import InvalidTransitionError, validate_transition
from state_management.workflow_state import SystemState, WorkflowState


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


# ── Transition validation ─────────────────────────────────────────────────────

def test_valid_transition_initialized_to_training():
    validate_transition(SystemState.INITIALIZED, SystemState.TRAINING)


def test_valid_transition_validating_to_success():
    validate_transition(SystemState.VALIDATING, SystemState.SUCCESS)


def test_invalid_transition_raises():
    with pytest.raises(InvalidTransitionError):
        validate_transition(SystemState.INITIALIZED, SystemState.SUCCESS)


def test_terminal_state_has_no_transitions():
    with pytest.raises(InvalidTransitionError):
        validate_transition(SystemState.TERMINATED, SystemState.TRAINING)


def test_success_to_terminated():
    validate_transition(SystemState.SUCCESS, SystemState.TERMINATED)


def test_failed_to_terminated():
    validate_transition(SystemState.FAILED, SystemState.TERMINATED)


# ── StateManager ─────────────────────────────────────────────────────────────

def test_initialize_and_get():
    mgr = StateManager()
    state = _make_state()
    mgr.initialize(state)
    assert mgr.get_state().ticker == "BTC-USD"
    assert mgr.get_state().current_state == SystemState.INITIALIZED


def test_transition_updates_state():
    mgr = StateManager()
    mgr.initialize(_make_state())
    mgr.transition(SystemState.TRAINING)
    assert mgr.get_state().current_state == SystemState.TRAINING


def test_invalid_transition_via_manager():
    mgr = StateManager()
    mgr.initialize(_make_state())
    with pytest.raises(InvalidTransitionError):
        mgr.transition(SystemState.SUCCESS)


def test_update_state_fields():
    mgr = StateManager()
    mgr.initialize(_make_state())
    mgr.update_state(inner_loop_attempt=2, selected_methodology="xgboost")
    state = mgr.get_state()
    assert state.inner_loop_attempt == 2
    assert state.selected_methodology == "xgboost"


def test_state_persists_across_get_calls():
    mgr = StateManager()
    mgr.initialize(_make_state())
    mgr.update_state(outer_loop_iteration=3)
    # Clear in-memory cache to force reload from backend
    mgr._state = None
    assert mgr.get_state().outer_loop_iteration == 3


def test_is_terminal_false_on_init():
    state = _make_state()
    assert not state.is_terminal()


def test_is_terminal_true_on_success():
    state = _make_state(current_state=SystemState.SUCCESS)
    assert state.is_terminal()


def test_is_terminal_true_on_failed():
    state = _make_state(current_state=SystemState.FAILED)
    assert state.is_terminal()


def test_catalog_queue_default():
    state = _make_state()
    assert state.catalog_queue == ["xgboost", "lightgbm", "lstm"]


def test_get_state_without_initialize_raises():
    mgr = StateManager()
    with pytest.raises(RuntimeError, match="initialize"):
        mgr.get_state()

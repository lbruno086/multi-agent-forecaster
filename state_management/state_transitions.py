from __future__ import annotations

from state_management.workflow_state import SystemState

VALID_TRANSITIONS: dict[SystemState, set[SystemState]] = {
    SystemState.INITIALIZED: {
        SystemState.RESEARCHING,
        SystemState.TRAINING,       # catalog base skips research
    },
    SystemState.RESEARCHING: {
        SystemState.GENERATING_CODE,
        SystemState.FAILED,
    },
    SystemState.GENERATING_CODE: {
        SystemState.TRAINING,
        SystemState.FAILED,
    },
    SystemState.TRAINING: {
        SystemState.VALIDATING,
        SystemState.FAILED,
    },
    SystemState.VALIDATING: {
        SystemState.IMPROVING,
        SystemState.RETRAINING,
        SystemState.RESEARCHING,
        SystemState.SUCCESS,
        SystemState.FAILED,
    },
    SystemState.IMPROVING: {
        SystemState.RETRAINING,
        SystemState.FAILED,
    },
    SystemState.RETRAINING: {
        SystemState.VALIDATING,
        SystemState.FAILED,
    },
    SystemState.SUCCESS: {
        SystemState.TERMINATED,
    },
    SystemState.FAILED: {
        SystemState.TERMINATED,
    },
    SystemState.TERMINATED: set(),
}


class InvalidTransitionError(Exception):
    pass


def validate_transition(from_state: SystemState, to_state: SystemState) -> None:
    allowed = VALID_TRANSITIONS.get(from_state, set())
    if to_state not in allowed:
        raise InvalidTransitionError(
            f"Invalid transition: {from_state.value} → {to_state.value}. "
            f"Allowed from {from_state.value}: "
            f"{[s.value for s in allowed] or 'none (terminal state)'}"
        )

from __future__ import annotations

import json

from state_management.backends.memory_backend import MemoryBackend
from state_management.backends.redis_backend import RedisBackend
from state_management.state_transitions import InvalidTransitionError, validate_transition
from state_management.workflow_state import SystemState, WorkflowState
from tools.logger import get_logger

log = get_logger(__name__)

_STATE_KEY = "workflow:state"

Backend = MemoryBackend | RedisBackend


class StateManager:
    """Centralized state manager. All state reads/writes go through here."""

    def __init__(self, backend: Backend | None = None) -> None:
        self._backend: Backend = backend or MemoryBackend()
        self._state: WorkflowState | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def initialize(self, state: WorkflowState) -> WorkflowState:
        self._state = state
        self._persist(state)
        log.info("state_initialized", ticker=state.ticker, state=state.current_state.value)
        return state

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_state(self) -> WorkflowState:
        if self._state is not None:
            return self._state
        raw = self._backend.get(_STATE_KEY)
        if raw is None:
            raise RuntimeError("StateManager has no state. Call initialize() first.")
        self._state = WorkflowState.model_validate_json(raw)
        return self._state

    # ── Write ─────────────────────────────────────────────────────────────────

    def update_state(self, **fields) -> WorkflowState:
        state = self.get_state()
        updated = state.model_copy(update=fields)
        self._state = updated
        self._persist(updated)
        return updated

    def transition(self, new_state: SystemState) -> WorkflowState:
        state = self.get_state()
        validate_transition(state.current_state, new_state)
        old = state.current_state
        updated = self.update_state(current_state=new_state)
        log.info(
            "state_transition",
            from_state=old.value,
            to_state=new_state.value,
            ticker=state.ticker,
            outer_iter=state.outer_loop_iteration,
            inner_attempt=state.inner_loop_attempt,
        )
        return updated

    # ── Internal ─────────────────────────────────────────────────────────────

    def _persist(self, state: WorkflowState) -> None:
        self._backend.set(_STATE_KEY, state.model_dump_json())

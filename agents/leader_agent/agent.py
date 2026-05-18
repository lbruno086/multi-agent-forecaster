from __future__ import annotations

from agents.base_agent import BaseAgent
from agents.leader_agent.decision_engine import decide
from state_management.state_transitions import validate_transition
from state_management.workflow_state import SystemState, WorkflowState

_ACTION_TO_STATE: dict[str, SystemState] = {
    "TUNE": SystemState.IMPROVING,
    "NEW_VARIANT": SystemState.RETRAINING,
    "RESEARCH": SystemState.RESEARCHING,
    "SUCCESS": SystemState.SUCCESS,
    "FAILED": SystemState.FAILED,
}


class LeaderAgent(BaseAgent):
    name = "leader_agent"

    async def run(self, state: WorkflowState) -> WorkflowState:
        decision = decide(state)
        state.leader_decision = decision

        next_system_state = _ACTION_TO_STATE.get(decision.action)
        if next_system_state:
            _TERMINAL = {SystemState.SUCCESS, SystemState.FAILED}
            if next_system_state in _TERMINAL:
                # Terminal states are always reachable — bypass transition guard
                state.current_state = next_system_state
            else:
                try:
                    validate_transition(state.current_state, next_system_state)
                    state.current_state = next_system_state
                except Exception:
                    # Orchestrator edges handle routing to intermediate nodes
                    pass

        self._log.info(
            "leader_decided",
            action=decision.action,
            reason=decision.reason,
            outer_iter=state.outer_loop_iteration,
            inner_attempt=state.inner_loop_attempt,
            metric=state.current_metric_value,
        )
        return state

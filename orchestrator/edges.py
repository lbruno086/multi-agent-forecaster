from __future__ import annotations

from langgraph.graph import END

from state_management.workflow_state import WorkflowState

_ACTION_TO_NODE: dict[str, str] = {
    "TUNE": "validation",
    "NEW_VARIANT": "codegen",
    "RESEARCH": "research",
    "SUCCESS": END,
    "FAILED": END,
}


def route_from_leader(state: WorkflowState) -> str:
    """
    Conditional edge out of leader_node.

    Returns the name of the next node, or END if the workflow is complete.
    """
    if state.is_terminal():
        return END

    decision = state.leader_decision
    if decision is None:
        # Should never happen after leader_node runs, but safe fallback.
        return "validation"

    return _ACTION_TO_NODE.get(decision.action, "validation")

from __future__ import annotations

from langgraph.graph import END, StateGraph

from orchestrator.edges import route_from_leader
from orchestrator.nodes import codegen_node, leader_node, research_node, validation_node
from state_management.workflow_state import WorkflowState


def build_graph():
    """
    Assemble and compile the LangGraph optimization loop.

    Flow:
        leader ──TUNE──────────────────► validation ──► leader
               ──NEW_VARIANT──► codegen ──► validation ──► leader
               ──RESEARCH─────► research ─► codegen ──► validation ──► leader
               ──SUCCESS/FAILED──────────────────────────────────────► END
    """
    graph: StateGraph = StateGraph(WorkflowState)

    graph.add_node("leader", leader_node)
    graph.add_node("validation", validation_node)
    graph.add_node("research", research_node)
    graph.add_node("codegen", codegen_node)

    # Entry: leader decides first (no validation report → TUNE → validation)
    graph.set_entry_point("leader")

    # Conditional fan-out from leader
    graph.add_conditional_edges(
        "leader",
        route_from_leader,
        {
            "validation": "validation",
            "research": "research",
            "codegen": "codegen",
            END: END,
        },
    )

    # Fixed back-edges
    graph.add_edge("research", "codegen")
    graph.add_edge("codegen", "validation")
    graph.add_edge("validation", "leader")

    return graph.compile()


class Orchestrator:
    """
    Thin wrapper around the compiled LangGraph for use by main.py.

    Compiles the graph once on first use.
    """

    def __init__(self) -> None:
        self._graph = build_graph()

    async def run(self, state: WorkflowState) -> WorkflowState:
        result = await self._graph.ainvoke(state)
        # ainvoke returns a dict when the schema is a Pydantic model;
        # reconstruct the model so callers always get WorkflowState.
        if isinstance(result, dict):
            return WorkflowState(**result)
        return result

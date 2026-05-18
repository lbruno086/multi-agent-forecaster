from __future__ import annotations

from agents.code_generation_agent.agent import CodeGenerationAgent
from agents.leader_agent.agent import LeaderAgent
from agents.research_agent.agent import ResearchAgent
from agents.validation_agent.agent import ValidationAgent
from state_management.workflow_state import WorkflowState
from tools.logger import get_logger

log = get_logger(__name__)

# Module-level singletons — created once on first call, shared across graph steps.
_leader: LeaderAgent | None = None
_validation: ValidationAgent | None = None
_research: ResearchAgent | None = None
_codegen: CodeGenerationAgent | None = None


def _get_leader() -> LeaderAgent:
    global _leader
    if _leader is None:
        _leader = LeaderAgent()
    return _leader


def _get_validation() -> ValidationAgent:
    global _validation
    if _validation is None:
        _validation = ValidationAgent()
    return _validation


def _get_research() -> ResearchAgent:
    global _research
    if _research is None:
        _research = ResearchAgent()
    return _research


def _get_codegen() -> CodeGenerationAgent:
    global _codegen
    if _codegen is None:
        _codegen = CodeGenerationAgent()
    return _codegen


async def leader_node(state: WorkflowState) -> WorkflowState:
    log.info("node_enter", node="leader", current_state=state.current_state.value)
    return await _get_leader().run(state)


async def validation_node(state: WorkflowState) -> WorkflowState:
    log.info("node_enter", node="validation", current_state=state.current_state.value)
    return await _get_validation().run(state)


async def research_node(state: WorkflowState) -> WorkflowState:
    log.info("node_enter", node="research", current_state=state.current_state.value)
    return await _get_research().run(state)


async def codegen_node(state: WorkflowState) -> WorkflowState:
    log.info("node_enter", node="codegen", current_state=state.current_state.value)

    # NEW_VARIANT: prefer the next catalog entry before falling back to LLM generation.
    # Resets inner_loop_attempt so the new methodology gets its own 3 attempts.
    if state.leader_decision and state.leader_decision.action == "NEW_VARIANT":
        if state.catalog_queue:
            state.selected_methodology = state.catalog_queue.pop(0)
        state.inner_loop_attempt = 0

    return await _get_codegen().run(state)

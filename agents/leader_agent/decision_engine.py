from __future__ import annotations

from evaluation.models import LeaderDecision
from state_management.workflow_state import WorkflowState


def decide(state: WorkflowState) -> LeaderDecision:
    """Pure decision function — no LLM, no side effects."""
    report = state.validation_report

    # 1. Threshold reached
    if (
        state.current_metric_value is not None
        and state.current_metric_value <= state.target_threshold
    ):
        return LeaderDecision(
            action="SUCCESS",
            reason=f"metric {state.current_metric_value:.4f} <= threshold {state.target_threshold}",
        )

    # 2. Outer loop exhausted
    if state.outer_loop_iteration >= state.max_outer_iterations:
        best = state.best_metric_value
        return LeaderDecision(
            action="FAILED",
            reason=f"max outer iterations ({state.max_outer_iterations}) reached. "
                   f"Best metric: {best:.4f}" if best is not None else "no successful run",
        )

    # 3. No validation report yet — start training
    if report is None:
        return LeaderDecision(action="TUNE", reason="no validation report yet, start training")

    # 4. Inner loop exhausted — escalate to research
    if state.inner_loop_attempt >= state.max_inner_attempts:
        return LeaderDecision(
            action="RESEARCH",
            reason=f"inner loop exhausted after {state.max_inner_attempts} attempts "
                   f"on methodology '{state.selected_methodology}'",
        )

    # 5. High-confidence methodology issue — skip remaining inner attempts
    if report.root_cause == "METHODOLOGY_ISSUE" and report.confidence >= 0.80:
        return LeaderDecision(
            action="RESEARCH",
            reason=f"METHODOLOGY_ISSUE with confidence={report.confidence:.2f}. "
                   f"Skipping remaining inner attempts.",
        )

    # 6. Parameter issue — tune hyperparameters
    if report.root_cause == "PARAMETER_ISSUE":
        return LeaderDecision(
            action="TUNE",
            reason=f"PARAMETER_ISSUE detected (confidence={report.confidence:.2f}). "
                   f"Running Optuna tuning.",
        )

    # 7. Architecture issue — try a different variant
    if report.root_cause == "ARCHITECTURE_ISSUE":
        return LeaderDecision(
            action="NEW_VARIANT",
            reason=f"ARCHITECTURE_ISSUE detected (confidence={report.confidence:.2f}). "
                   f"Trying a different model variant.",
        )

    # 8. Default — tune
    return LeaderDecision(
        action="TUNE",
        reason="no strong signal, defaulting to parameter tuning",
    )

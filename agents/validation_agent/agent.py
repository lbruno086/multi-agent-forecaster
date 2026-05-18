from __future__ import annotations

import numpy as np

from agent_skills.metrics_evaluation_skill import MetricsEvaluationSkill
from agents.base_agent import BaseAgent
from datasets.data_fetcher import fetch_ohlcv
from evaluation.models import ExperimentResult
from state_management.workflow_state import SystemState, WorkflowState
from tools.logger import get_logger

log = get_logger(__name__)
_eval_skill = MetricsEvaluationSkill()


class ValidationAgent(BaseAgent):
    name = "validation_agent"

    async def run(self, state: WorkflowState) -> WorkflowState:
        state.current_state = SystemState.TRAINING

        methodology = state.selected_methodology
        if not methodology:
            methodology = _pop_catalog(state)
            state.selected_methodology = methodology

        self._log.info(
            "validation_started",
            methodology=methodology,
            inner_attempt=state.inner_loop_attempt,
        )

        # Fetch data
        data = fetch_ohlcv(
            ticker=state.ticker,
            start_date=state.start_date,
            end_date=state.end_date,
            timeframe=state.timeframe,
        )

        # Get model skill
        skill = self.skill_registry.get(methodology)
        if skill is None:
            state.error = f"Skill '{methodology}' not found in registry."
            state.current_state = SystemState.FAILED
            return state

        # Build params from state
        exec_params: dict = {
            "data": data,
            "train_pct": state.train_pct,
            "n_folds": state.n_folds,
            "metric_name": state.metric_to_optimize,
        }
        if state.leader_decision and state.leader_decision.suggested_params:
            exec_params.update(state.leader_decision.suggested_params)

        # Run skill
        state.current_state = SystemState.VALIDATING
        try:
            skill_result = skill.execute(exec_params)
        except Exception as exc:
            state.error = f"Skill execution failed: {exc}"
            state.current_state = SystemState.FAILED
            self._log.error("skill_execution_failed", methodology=methodology, error=str(exc))
            return state

        # Build validation targets for residuals
        target_values = _extract_val_targets(data, state)

        # Diagnose
        report = _eval_skill.diagnose({
            "skill_result": skill_result,
            "metric_name": state.metric_to_optimize,
            "target_values": target_values,
        })

        # Update state
        state.current_metric_value = report.metric_value
        state.validation_report = report
        state.inner_loop_attempt += 1

        if methodology not in state.tried_methodologies:
            state.tried_methodologies.append(methodology)

        # Track experiment
        state.record_experiment(
            ExperimentResult(
                iteration=state.outer_loop_iteration,
                inner_attempt=state.inner_loop_attempt,
                methodology=methodology,
                params=skill_result.params_used,
                metric_value=report.metric_value,
                metric_name=state.metric_to_optimize,
                model_path=skill_result.model_path,
                diagnostic=report,
            )
        )

        self._log.info(
            "validation_complete",
            methodology=methodology,
            metric=round(report.metric_value, 4),
            root_cause=report.root_cause,
            confidence=round(report.confidence, 3),
        )
        return state


def _pop_catalog(state: WorkflowState) -> str:
    if state.catalog_queue:
        return state.catalog_queue.pop(0)
    return "xgboost"  # fallback


def _extract_val_targets(data, state: WorkflowState) -> np.ndarray | None:
    try:
        from datasets.walk_forward_splitter import WalkForwardSplitter
        splitter = WalkForwardSplitter(state.train_pct, state.n_folds)
        folds = splitter.split(data)
        targets = np.concatenate([fold.val["close"].values for fold in folds])
        return targets
    except Exception:
        return None

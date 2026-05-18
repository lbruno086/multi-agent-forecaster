from __future__ import annotations

from typing import Any

from evaluation.models import DiagnosticReport, ExperimentResult
from tools.logger import get_logger

log = get_logger(__name__)


class ExperimentTracker:
    """
    MLflow-backed experiment tracker.

    Lazy-imports mlflow so the rest of the system works without it installed.
    Experiment name is derived from ticker + timeframe for easy filtering.
    """

    def __init__(self, ticker: str, timeframe: str) -> None:
        self._experiment_name = f"{ticker}_{timeframe}"
        self._mlflow = None

    def _ensure_ready(self) -> Any:
        if self._mlflow is not None:
            return self._mlflow
        try:
            import mlflow  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "mlflow is required for experiment tracking. "
                "Install it with: pip install mlflow"
            ) from exc
        mlflow.set_experiment(self._experiment_name)
        self._mlflow = mlflow
        return mlflow

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_experiment(
        self,
        result: ExperimentResult,
        extra_tags: dict[str, str] | None = None,
    ) -> str | None:
        """Log a completed experiment; returns the MLflow run_id."""
        try:
            mlflow = self._ensure_ready()
        except RuntimeError as exc:
            log.warning("mlflow_unavailable", error=str(exc))
            return None

        tags = {
            "methodology": result.methodology,
            "iteration": str(result.iteration),
            "inner_attempt": str(result.inner_attempt),
        }
        if extra_tags:
            tags.update(extra_tags)

        with mlflow.start_run(tags=tags) as run:
            mlflow.log_metric(result.metric_name, result.metric_value)
            mlflow.log_params(result.params or {})

            if result.diagnostic:
                self._log_diagnostic(mlflow, result.diagnostic)

            log.info(
                "experiment_logged",
                run_id=run.info.run_id,
                methodology=result.methodology,
                metric=result.metric_value,
            )
            return run.info.run_id

    def _log_diagnostic(self, mlflow: Any, report: DiagnosticReport) -> None:
        mlflow.log_metric("train_score", report.train_score)
        mlflow.log_metric("val_score", report.val_score)
        mlflow.log_metric("confidence", report.confidence)
        mlflow.set_tag("root_cause", report.root_cause)
        for i, score in enumerate(report.fold_scores):
            mlflow.log_metric("fold_score", score, step=i)

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_best_run(self, metric_name: str, ascending: bool = True) -> dict | None:
        """Return the params + metrics of the best run for the given metric."""
        try:
            mlflow = self._ensure_ready()
        except RuntimeError as exc:
            log.warning("mlflow_unavailable", error=str(exc))
            return None

        order = "ASC" if ascending else "DESC"
        runs = mlflow.search_runs(
            experiment_names=[self._experiment_name],
            order_by=[f"metrics.{metric_name} {order}"],
            max_results=1,
        )
        if runs.empty:
            return None

        best = runs.iloc[0]
        return {
            "run_id": best["run_id"],
            "metric_value": best.get(f"metrics.{metric_name}"),
            "params": {
                k.removeprefix("params."): v
                for k, v in best.items()
                if k.startswith("params.")
            },
        }

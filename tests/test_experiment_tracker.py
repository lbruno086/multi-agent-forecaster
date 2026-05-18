from __future__ import annotations

import pytest

from evaluation.models import DiagnosticReport, ExperimentResult
from experiments.experiment_tracker import ExperimentTracker


def _make_result(metric_value: float = 0.28) -> ExperimentResult:
    report = DiagnosticReport(
        metric_value=metric_value,
        metric_name="mape",
        root_cause="PARAMETER_ISSUE",
        evidence={},
        confidence=0.7,
        details="test",
        fold_scores=[metric_value],
        train_score=metric_value - 0.05,
        val_score=metric_value,
    )
    return ExperimentResult(
        iteration=1,
        inner_attempt=1,
        methodology="xgboost",
        params={"n_estimators": 100, "learning_rate": 0.1},
        metric_value=metric_value,
        metric_name="mape",
        model_path=None,
        diagnostic=report,
    )


def test_experiment_tracker_logs_with_mlflow(mocker):
    mock_run = mocker.MagicMock()
    mock_run.__enter__ = mocker.MagicMock(return_value=mock_run)
    mock_run.__exit__ = mocker.MagicMock(return_value=False)
    mock_run.info.run_id = "test-run-id"

    mock_mlflow = mocker.MagicMock()
    mock_mlflow.start_run.return_value = mock_run

    tracker = ExperimentTracker("BTC-USD", "1h")
    tracker._mlflow = mock_mlflow

    result = _make_result()
    run_id = tracker.log_experiment(result)

    assert run_id == "test-run-id"
    mock_mlflow.log_metric.assert_called()
    mock_mlflow.log_params.assert_called_once_with({"n_estimators": 100, "learning_rate": 0.1})


def test_experiment_tracker_returns_none_when_mlflow_missing(mocker):
    tracker = ExperimentTracker("BTC-USD", "1h")
    mocker.patch.object(tracker, "_ensure_ready", side_effect=RuntimeError("mlflow unavailable"))
    result = _make_result()
    run_id = tracker.log_experiment(result)
    assert run_id is None


def test_experiment_tracker_get_best_run_returns_none_on_missing_mlflow(mocker):
    tracker = ExperimentTracker("BTC-USD", "1h")
    mocker.patch.object(tracker, "_ensure_ready", side_effect=RuntimeError("mlflow unavailable"))
    result = tracker.get_best_run("mape")
    assert result is None

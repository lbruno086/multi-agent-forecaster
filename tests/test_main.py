from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

from main import _validate_params, _extract_data, _print_step, app

runner = CliRunner()


# ── _validate_params ──────────────────────────────────────────────────────────

def test_validate_params_ok():
    _validate_params(train_pct=0.80, n_folds=5, threshold=0.30)  # no exception


def test_validate_params_bad_train_pct_zero():
    with pytest.raises(typer.Exit):
        _validate_params(train_pct=0.0, n_folds=5, threshold=0.30)


def test_validate_params_bad_train_pct_one():
    with pytest.raises(typer.Exit):
        _validate_params(train_pct=1.0, n_folds=5, threshold=0.30)


def test_validate_params_bad_n_folds():
    with pytest.raises(typer.Exit):
        _validate_params(train_pct=0.80, n_folds=0, threshold=0.30)


def test_validate_params_bad_threshold():
    with pytest.raises(typer.Exit):
        _validate_params(train_pct=0.80, n_folds=5, threshold=0.0)


# ── _extract_data ─────────────────────────────────────────────────────────────

def test_extract_data_merges_dict():
    prev = {"a": 1, "b": 2}
    update = {"b": 99, "c": 3}
    result = _extract_data(update, prev)
    assert result == {"a": 1, "b": 99, "c": 3}


def test_extract_data_pydantic_model():
    from state_management.workflow_state import WorkflowState
    state = WorkflowState(
        ticker="BTC-USD",
        start_date="2023-01-01",
        end_date="2024-01-01",
        timeframe="1d",
        train_pct=0.80,
        n_folds=5,
        metric_to_optimize="mape",
        target_threshold=0.30,
    )
    result = _extract_data(state, {})
    assert result["ticker"] == "BTC-USD"
    assert isinstance(result, dict)


# ── CLI smoke tests (no LLM called) ──────────────────────────────────────────

def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ticker" in result.output.lower()


def test_cli_bad_train_pct():
    result = runner.invoke(app, [
        "--ticker", "BTC-USD",
        "--start-date", "2023-01-01",
        "--end-date", "2024-01-01",
        "--train-pct", "1.5",
    ])
    assert result.exit_code != 0


def test_cli_bad_n_folds():
    result = runner.invoke(app, [
        "--ticker", "BTC-USD",
        "--start-date", "2023-01-01",
        "--end-date", "2024-01-01",
        "--n-folds", "0",
    ])
    assert result.exit_code != 0


def test_cli_bad_threshold():
    result = runner.invoke(app, [
        "--ticker", "BTC-USD",
        "--start-date", "2023-01-01",
        "--end-date", "2024-01-01",
        "--threshold", "-0.1",
    ])
    assert result.exit_code != 0


def test_cli_runs_workflow(mocker):
    """Verify CLI wires up correctly by mocking the async workflow."""
    async def _noop(*args, **kwargs):
        pass

    mocker.patch("main._run_workflow", side_effect=_noop)

    result = runner.invoke(app, [
        "--ticker", "BTC-USD",
        "--start-date", "2023-01-01",
        "--end-date", "2024-01-01",
        "--timeframe", "1d",
        "--train-pct", "0.80",
        "--n-folds", "5",
        "--threshold", "0.30",
        "--metric", "mape",
    ])
    # Header should be printed regardless
    assert "BTC-USD" in result.output
    assert "mape" in result.output

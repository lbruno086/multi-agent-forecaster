from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated

import typer

app = typer.Typer(
    name="research-trading",
    help="Autonomous multi-agent financial forecasting optimizer.",
    add_completion=False,
)


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_params(train_pct: float, n_folds: int, threshold: float) -> None:
    if not (0 < train_pct < 1):
        typer.echo(f"Error: --train-pct must be between 0 and 1, got {train_pct}", err=True)
        raise typer.Exit(1)
    if n_folds < 1:
        typer.echo(f"Error: --n-folds must be >= 1, got {n_folds}", err=True)
        raise typer.Exit(1)
    if threshold <= 0:
        typer.echo(f"Error: --threshold must be > 0, got {threshold}", err=True)
        raise typer.Exit(1)


# ── Console helpers ───────────────────────────────────────────────────────────

def _print_header(ticker: str, timeframe: str, metric: str, threshold: float) -> None:
    typer.echo("")
    typer.echo("=" * 60)
    typer.echo("  Multi-Agent Financial Forecasting Optimizer")
    typer.echo("=" * 60)
    typer.echo(f"  Ticker   : {ticker}")
    typer.echo(f"  Timeframe: {timeframe}")
    typer.echo(f"  Target   : {metric} <= {threshold}")
    typer.echo(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    typer.echo("=" * 60)
    typer.echo("")


def _extract_data(node_output: dict | object, prev: dict) -> dict:
    if isinstance(node_output, dict):
        return {**prev, **node_output}
    return node_output.model_dump()


def _print_step(node: str, data: dict) -> None:
    methodology = data.get("selected_methodology") or ""
    metric_val = data.get("current_metric_value")
    decision_obj = data.get("leader_decision")

    method_str = f" [{methodology}]" if methodology else ""
    metric_str = f" | metric={metric_val:.4f}" if metric_val is not None else ""

    if decision_obj:
        action = (
            decision_obj.get("action", "?")
            if isinstance(decision_obj, dict)
            else decision_obj.action
        )
        reason = (
            decision_obj.get("reason", "")
            if isinstance(decision_obj, dict)
            else decision_obj.reason
        )
        short_reason = f" ({reason[:55]}...)" if len(reason) > 55 else f" ({reason})" if reason else ""
        decision_str = f" → {action}{short_reason}"
    else:
        decision_str = ""

    typer.echo(f"  {node:<12}{method_str}{metric_str}{decision_str}")


def _print_result(data: dict, metric: str, threshold: float) -> None:
    typer.echo("")
    typer.echo("─" * 60)

    state_val = data.get("current_state", "")
    if hasattr(state_val, "value"):
        state_val = state_val.value

    if state_val == "SUCCESS":
        metric_val = data.get("current_metric_value")
        model_path = data.get("best_model_path")
        typer.echo(typer.style("  SUCCESS", fg=typer.colors.GREEN, bold=True))
        typer.echo(f"  {metric} = {metric_val:.4f}  (threshold: {threshold})")
        if model_path:
            typer.echo(f"  Model  : {model_path}")
    else:
        best = data.get("best_metric_value")
        decision_obj = data.get("leader_decision")
        reason = ""
        if decision_obj:
            reason = (
                decision_obj.get("reason", "")
                if isinstance(decision_obj, dict)
                else decision_obj.reason
            )
        typer.echo(typer.style("  FAILED", fg=typer.colors.RED, bold=True))
        if best is not None:
            typer.echo(f"  Best {metric} achieved: {best:.4f}  (threshold: {threshold})")
        else:
            typer.echo(f"  No successful run completed.")
        if reason:
            typer.echo(f"  Reason : {reason}")

    typer.echo("─" * 60)
    typer.echo("")


# ── Async workflow runner ─────────────────────────────────────────────────────

async def _run_workflow(
    ticker: str,
    start_date: str,
    end_date: str,
    timeframe: str,
    train_pct: float,
    n_folds: int,
    threshold: float,
    metric: str,
) -> None:
    # Defer imports so env-var validation errors surface with a clean message.
    from orchestrator.graph import build_graph
    from state_management.workflow_state import WorkflowState

    state = WorkflowState(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        train_pct=train_pct,
        n_folds=n_folds,
        metric_to_optimize=metric,
        target_threshold=threshold,
    )

    graph = build_graph()
    last_data: dict = state.model_dump()

    typer.echo("  Running optimization loop ...\n")

    async for event in graph.astream(state):
        for node_name, node_output in event.items():
            last_data = _extract_data(node_output, last_data)
            _print_step(node_name, last_data)

    _print_result(last_data, metric, threshold)


# ── CLI command ───────────────────────────────────────────────────────────────

@app.command()
def research_trading(
    ticker: Annotated[
        str, typer.Option(help="Asset ticker in yfinance format (e.g. BTC-USD)")
    ],
    start_date: Annotated[
        str, typer.Option(help="Data window start date (YYYY-MM-DD)")
    ],
    end_date: Annotated[
        str, typer.Option(help="Data window end date (YYYY-MM-DD)")
    ],
    timeframe: Annotated[
        str, typer.Option(help="Candle interval: 1h, 1d, 15m, ...")
    ] = "1d",
    train_pct: Annotated[
        float, typer.Option(help="Fraction of data for training (0–1)")
    ] = 0.80,
    n_folds: Annotated[
        int, typer.Option(help="Number of walk-forward validation folds")
    ] = 5,
    threshold: Annotated[
        float, typer.Option(help="Target metric threshold to declare success")
    ] = 0.30,
    metric: Annotated[
        str, typer.Option(help="Metric to optimize (mape, rmse, mae, r2, ...)")
    ] = "mape",
) -> None:
    """
    Run the multi-agent forecasting optimizer for a financial asset.

    Example:

        research-trading --ticker BTC-USD --start-date 2023-01-01 \\
            --end-date 2024-01-01 --timeframe 1h --train-pct 0.80 \\
            --n-folds 5 --threshold 0.30
    """
    _validate_params(train_pct, n_folds, threshold)
    _print_header(ticker, timeframe, metric, threshold)

    try:
        asyncio.run(
            _run_workflow(ticker, start_date, end_date, timeframe, train_pct, n_folds, threshold, metric)
        )
    except KeyboardInterrupt:
        typer.echo("\n  Interrupted by user.", err=True)
        raise typer.Exit(130)
    except Exception as exc:
        typer.echo(f"\n  Error: {exc}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

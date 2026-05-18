from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from configs.settings import settings
from tools.logger import get_logger

log = get_logger(__name__)

_TIMEFRAME_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
    "1wk": "1wk",
}


def _cache_path(ticker: str, timeframe: str, start: str, end: str) -> Path:
    safe = ticker.replace("/", "-").replace(":", "-").replace(".", "-").replace("..", "-")
    filename = f"{safe}_{timeframe}_{start}_{end}.parquet"
    # Ensure no path traversal — use only the basename
    return settings.data.cache_dir / Path(filename).name


def fetch_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str,
    timeframe: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    if timeframe not in _TIMEFRAME_MAP:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Valid options: {list(_TIMEFRAME_MAP)}"
        )

    cache_file = _cache_path(ticker, timeframe, start_date, end_date)

    if use_cache and cache_file.exists():
        log.info("loading from cache", ticker=ticker, timeframe=timeframe, path=str(cache_file))
        df = pd.read_parquet(cache_file)
        return df

    log.info("fetching from yfinance", ticker=ticker, timeframe=timeframe,
             start=start_date, end=end_date)

    interval = _TIMEFRAME_MAP[timeframe]
    raw = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"No data returned for ticker='{ticker}' "
            f"timeframe='{timeframe}' {start_date}→{end_date}. "
            "Check the ticker symbol and date range."
        )

    df = _normalize(raw)

    settings.data.cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file)
    log.info("cached", path=str(cache_file), rows=len(df))

    return df


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    # yfinance returns MultiIndex columns when downloading a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [c.lower() for c in raw.columns]

    rename = {"adj close": "close"}
    raw = raw.rename(columns=rename)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns after normalization: {missing}")

    _COLUMN_ORDER = ["open", "high", "low", "close", "volume"]
    raw = raw[_COLUMN_ORDER].copy()
    raw.index.name = "datetime"
    raw = raw.sort_index()
    raw = raw.dropna()

    return raw

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agent_skills.base_skill import BaseSkill, SkillResult


class FeatureEngineeringSkill(BaseSkill):
    name = "feature_engineering"
    description = "Adds technical indicators to OHLCV data."

    def execute(self, params: dict[str, Any]) -> SkillResult:
        df: pd.DataFrame = params["data"]
        enriched = self.transform(df, params)
        return SkillResult(
            skill_name=self.name,
            predictions=np.array([]),
            fold_scores=[],
            train_score=0.0,
            val_score=0.0,
            model=None,
            metadata={"feature_count": len(enriched.columns)},
        )

    def transform(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        p = params or {}
        out = df.copy()

        # Returns
        out["returns"] = out["close"].pct_change()
        out["log_returns"] = np.log(out["close"] / out["close"].shift(1))

        # Moving averages
        for w in p.get("sma_windows", [7, 14, 21]):
            out[f"sma_{w}"] = out["close"].rolling(w).mean()
        for w in p.get("ema_windows", [9, 21]):
            out[f"ema_{w}"] = out["close"].ewm(span=w, adjust=False).mean()

        # RSI
        rsi_period = p.get("rsi_period", 14)
        out["rsi"] = self._rsi(out["close"], rsi_period)

        # MACD
        out["macd"], out["macd_signal"] = self._macd(out["close"])

        # Bollinger Bands
        bb_period = p.get("bb_period", 20)
        out["bb_mid"] = out["close"].rolling(bb_period).mean()
        out["bb_std"] = out["close"].rolling(bb_period).std()
        out["bb_upper"] = out["bb_mid"] + 2 * out["bb_std"]
        out["bb_lower"] = out["bb_mid"] - 2 * out["bb_std"]
        out["bb_pct"] = (out["close"] - out["bb_lower"]) / (
            out["bb_upper"] - out["bb_lower"] + 1e-9
        )
        out = out.drop(columns=["bb_std"])

        # ATR
        out["atr"] = self._atr(out, p.get("atr_period", 14))

        # Volume features
        out["volume_ma"] = out["volume"].rolling(10).mean()
        out["volume_ratio"] = out["volume"] / (out["volume_ma"] + 1e-9)

        # Price position
        out["high_low_range"] = out["high"] - out["low"]
        out["close_position"] = (out["close"] - out["low"]) / (
            out["high"] - out["low"] + 1e-9
        )

        return out.dropna()

    # ── Indicator implementations ─────────────────────────────────────────────

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def get_schema(self) -> dict[str, Any]:
        return {
            "params": {
                "data": "pd.DataFrame (OHLCV)",
                "sma_windows": "list[int] (default [7, 14, 21])",
                "ema_windows": "list[int] (default [9, 21])",
                "rsi_period": "int (default 14)",
                "bb_period": "int (default 20)",
                "atr_period": "int (default 14)",
            }
        }

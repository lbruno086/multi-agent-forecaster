from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train: pd.DataFrame
    val: pd.DataFrame


class WalkForwardSplitter:
    """Expanding-window walk-forward splitter for time series data.

    fold_size = round(total_bars * (1 - train_pct) / n_folds)

    Each fold expands the training window by fold_size bars.
    Validation window is always exactly fold_size bars.
    """

    def __init__(self, train_pct: float, n_folds: int) -> None:
        if not 0 < train_pct < 1:
            raise ValueError(f"train_pct must be in (0, 1), got {train_pct}")
        if n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {n_folds}")
        self.train_pct = train_pct
        self.n_folds = n_folds

    def split(self, df: pd.DataFrame) -> list[WalkForwardFold]:
        total = len(df)
        fold_size = round(total * (1 - self.train_pct) / self.n_folds)

        if fold_size < 1:
            raise ValueError(
                f"fold_size={fold_size} is too small. "
                f"Increase total bars or decrease n_folds."
            )

        initial_train_size = total - fold_size * self.n_folds

        if initial_train_size < 1:
            raise ValueError(
                f"initial_train_size={initial_train_size} is too small. "
                f"Increase train_pct or reduce n_folds."
            )

        folds: list[WalkForwardFold] = []
        for i in range(self.n_folds):
            train_end = initial_train_size + i * fold_size
            val_end = train_end + fold_size

            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()

            folds.append(WalkForwardFold(fold_index=i, train=train_df, val=val_df))

        return folds

    def fold_sizes(self, total_bars: int) -> dict:
        fold_size = round(total_bars * (1 - self.train_pct) / self.n_folds)
        initial_train = total_bars - fold_size * self.n_folds
        return {
            "total_bars": total_bars,
            "initial_train_size": initial_train,
            "fold_size": fold_size,
            "n_folds": self.n_folds,
            "val_total": fold_size * self.n_folds,
        }

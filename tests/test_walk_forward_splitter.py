import pandas as pd
import pytest

from datasets.walk_forward_splitter import WalkForwardSplitter


def _make_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({"close": range(n)}, index=pd.date_range("2023-01-01", periods=n, freq="h"))


def test_fold_count():
    splitter = WalkForwardSplitter(train_pct=0.80, n_folds=5)
    folds = splitter.split(_make_df(1000))
    assert len(folds) == 5


def test_fold_size():
    splitter = WalkForwardSplitter(train_pct=0.80, n_folds=5)
    folds = splitter.split(_make_df(1000))
    # fold_size = floor(1000 * 0.20 / 5) = 40
    for fold in folds:
        assert len(fold.val) == 40


def test_no_data_leakage():
    splitter = WalkForwardSplitter(train_pct=0.80, n_folds=5)
    folds = splitter.split(_make_df(1000))
    for fold in folds:
        train_idx = set(fold.train.index)
        val_idx = set(fold.val.index)
        assert train_idx.isdisjoint(val_idx), f"Fold {fold.fold_index}: train/val overlap"


def test_expanding_train_window():
    splitter = WalkForwardSplitter(train_pct=0.80, n_folds=5)
    folds = splitter.split(_make_df(1000))
    train_sizes = [len(f.train) for f in folds]
    for i in range(1, len(train_sizes)):
        assert train_sizes[i] > train_sizes[i - 1], "Train window must grow each fold"


def test_temporal_order():
    splitter = WalkForwardSplitter(train_pct=0.80, n_folds=3)
    folds = splitter.split(_make_df(300))
    for fold in folds:
        assert fold.train.index.max() < fold.val.index.min()


def test_invalid_train_pct():
    with pytest.raises(ValueError):
        WalkForwardSplitter(train_pct=1.5, n_folds=5)


def test_invalid_n_folds():
    with pytest.raises(ValueError):
        WalkForwardSplitter(train_pct=0.8, n_folds=0)


def test_fold_sizes_info():
    splitter = WalkForwardSplitter(train_pct=0.80, n_folds=5)
    info = splitter.fold_sizes(1000)
    assert info["fold_size"] == 40
    assert info["n_folds"] == 5

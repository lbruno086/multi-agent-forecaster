from __future__ import annotations

import numpy as np


def _validate(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {len(y_true)} and {len(y_pred)}"
        )


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _validate(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _validate(y_true, y_pred)
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _validate(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _validate(y_true, y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def proportional_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _validate(y_true, y_pred)
    mean_true = np.mean(np.abs(y_true))
    if mean_true == 0:
        return float("inf")
    return float(mae(y_true, y_pred) / mean_true)


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _validate(y_true, y_pred)
    var_true = np.var(y_true)
    if var_true == 0:
        return 0.0
    return float(1 - np.var(y_true - y_pred) / var_true)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _validate(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _validate(y_true, y_pred)
    if len(y_true) < 2:
        return 0.0
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir))

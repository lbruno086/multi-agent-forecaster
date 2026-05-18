import numpy as np
import pytest

from evaluation.metrics import (
    directional_accuracy,
    explained_variance,
    mae,
    mape,
    mse,
    proportional_error,
    r2,
    rmse,
)
from evaluation.metric_registry import MetricRegistry


def test_mse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mse(y, y) == 0.0


def test_rmse_known():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    assert abs(rmse(y_true, y_pred) - (2 / 3) ** 0.5) < 1e-9


def test_mae_known():
    y_true = np.array([1.0, 3.0])
    y_pred = np.array([2.0, 2.0])
    assert mae(y_true, y_pred) == 1.0


def test_mape_known():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])
    assert abs(mape(y_true, y_pred) - 0.075) < 1e-9


def test_mape_zero_true():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 1.0])
    assert mape(y_true, y_pred) == float("inf")


def test_proportional_error_known():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])
    assert abs(proportional_error(y_true, y_pred) - 10.0 / 150.0) < 1e-9


def test_r2_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert abs(r2(y, y) - 1.0) < 1e-9


def test_r2_baseline():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.full(3, np.mean(y_true))
    assert abs(r2(y_true, y_pred)) < 1e-9


def test_directional_accuracy_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert directional_accuracy(y, y) == 1.0


def test_directional_accuracy_worst():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([3.0, 2.0, 1.0])
    assert directional_accuracy(y_true, y_pred) == 0.0


def test_registry_evaluate():
    reg = MetricRegistry()
    y = np.array([1.0, 2.0, 3.0])
    assert reg.evaluate("mse", y, y) == 0.0


def test_registry_is_improvement_lower_better():
    reg = MetricRegistry()
    assert reg.is_improvement("mse", old=0.5, new=0.3)
    assert not reg.is_improvement("mse", old=0.3, new=0.5)


def test_registry_is_improvement_higher_better():
    reg = MetricRegistry()
    assert reg.is_improvement("r2", old=0.5, new=0.8)
    assert not reg.is_improvement("r2", old=0.8, new=0.5)


def test_registry_unknown_metric():
    reg = MetricRegistry()
    with pytest.raises(KeyError):
        reg.evaluate("unknown_metric", np.array([1.0]), np.array([1.0]))


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        mse(np.array([1.0, 2.0]), np.array([1.0]))


def test_registry_custom_metric():
    reg = MetricRegistry()
    reg.register("my_metric", lambda a, b: float(np.sum(a - b)), is_lower_better=True)
    assert "my_metric" in reg.available()

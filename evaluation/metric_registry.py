from __future__ import annotations

from typing import Callable

import numpy as np

from evaluation import metrics as _m


MetricFn = Callable[[np.ndarray, np.ndarray], float]

_Entry = tuple[MetricFn, bool]  # (fn, is_lower_better)


class MetricRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, _Entry] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register("mse", _m.mse, is_lower_better=True)
        self.register("rmse", _m.rmse, is_lower_better=True)
        self.register("mae", _m.mae, is_lower_better=True)
        self.register("mape", _m.mape, is_lower_better=True)
        self.register("proportional_error", _m.proportional_error, is_lower_better=True)
        self.register("explained_variance", _m.explained_variance, is_lower_better=False)
        self.register("r2", _m.r2, is_lower_better=False)
        self.register("directional_accuracy", _m.directional_accuracy, is_lower_better=False)

    def register(self, name: str, fn: MetricFn, is_lower_better: bool) -> None:
        self._registry[name] = (fn, is_lower_better)

    def evaluate(self, name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if name not in self._registry:
            raise KeyError(f"Unknown metric '{name}'. Available: {self.available()}")
        fn, _ = self._registry[name]
        return fn(y_true, y_pred)

    def is_improvement(self, name: str, old: float, new: float) -> bool:
        if name not in self._registry:
            raise KeyError(f"Unknown metric '{name}'.")
        _, is_lower_better = self._registry[name]
        return new < old if is_lower_better else new > old

    def is_lower_better(self, name: str) -> bool:
        if name not in self._registry:
            raise KeyError(f"Unknown metric '{name}'.")
        return self._registry[name][1]

    def available(self) -> list[str]:
        return list(self._registry)


metric_registry = MetricRegistry()

from __future__ import annotations

import pytest

from evaluation.models import DiagnosticReport, ExperimentResult
from memory.experiment_memory import ExperimentMemory
from memory.shared_memory import SharedMemory
from state_management.backends.memory_backend import MemoryBackend


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_shared_memory() -> SharedMemory:
    return SharedMemory(MemoryBackend())


def _make_result(metric_value: float = 0.35, methodology: str = "xgboost") -> ExperimentResult:
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
        methodology=methodology,
        params={},
        metric_value=metric_value,
        metric_name="mape",
        model_path=None,
        diagnostic=report,
    )


# ── SharedMemory ──────────────────────────────────────────────────────────────

def test_shared_memory_set_get():
    mem = _make_shared_memory()
    mem.set("key1", "value1")
    assert mem.get("key1") == "value1"


def test_shared_memory_default_on_missing():
    mem = _make_shared_memory()
    assert mem.get("nonexistent", default=42) == 42


def test_shared_memory_exists():
    mem = _make_shared_memory()
    mem.set("k", 1)
    assert mem.exists("k")
    assert not mem.exists("other")


def test_shared_memory_delete():
    mem = _make_shared_memory()
    mem.set("k", "v")
    mem.delete("k")
    assert not mem.exists("k")


def test_shared_memory_increment():
    mem = _make_shared_memory()
    assert mem.increment("counter") == 1
    assert mem.increment("counter", by=4) == 5


def test_shared_memory_append_to_list():
    mem = _make_shared_memory()
    mem.append_to_list("items", "a")
    mem.append_to_list("items", "b")
    assert mem.get_list("items") == ["a", "b"]


def test_shared_memory_stores_complex_types():
    mem = _make_shared_memory()
    mem.set("data", {"nested": [1, 2, 3], "flag": True})
    result = mem.get("data")
    assert result["nested"] == [1, 2, 3]
    assert result["flag"] is True


# ── ExperimentMemory ──────────────────────────────────────────────────────────

def test_experiment_memory_append_and_retrieve():
    mem = ExperimentMemory(_make_shared_memory())
    result = _make_result(0.30)
    mem.append(result)
    all_results = mem.all()
    assert len(all_results) == 1
    assert all_results[0].metric_value == 0.30


def test_experiment_memory_count():
    mem = ExperimentMemory(_make_shared_memory())
    assert mem.count() == 0
    mem.append(_make_result(0.40))
    mem.append(_make_result(0.35))
    assert mem.count() == 2


def test_experiment_memory_best_lower_is_better():
    mem = ExperimentMemory(_make_shared_memory())
    mem.append(_make_result(0.40, "xgboost"))
    mem.append(_make_result(0.25, "lightgbm"))
    mem.append(_make_result(0.35, "lstm"))
    best = mem.best(is_lower_better=True)
    assert best is not None
    assert best.metric_value == 0.25
    assert best.methodology == "lightgbm"


def test_experiment_memory_best_higher_is_better():
    mem = ExperimentMemory(_make_shared_memory())
    mem.append(_make_result(0.70, "xgboost"))
    mem.append(_make_result(0.85, "lightgbm"))
    best = mem.best(is_lower_better=False)
    assert best is not None
    assert best.metric_value == 0.85


def test_experiment_memory_best_empty_returns_none():
    mem = ExperimentMemory(_make_shared_memory())
    assert mem.best() is None


def test_experiment_memory_clear():
    mem = ExperimentMemory(_make_shared_memory())
    mem.append(_make_result())
    mem.clear()
    assert mem.count() == 0


def test_experiment_memory_preserves_diagnostic():
    mem = ExperimentMemory(_make_shared_memory())
    result = _make_result(0.28)
    mem.append(result)
    retrieved = mem.all()[0]
    assert retrieved.diagnostic is not None
    assert retrieved.diagnostic.root_cause == "PARAMETER_ISSUE"

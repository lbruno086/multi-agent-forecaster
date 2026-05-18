from __future__ import annotations

from evaluation.models import ExperimentResult

_EXPERIMENT_KEY = "experiment_history"


class ExperimentMemory:
    """
    Append-only log of ExperimentResult records backed by SharedMemory.

    Serialises each result as a dict so it survives Redis round-trips.
    """

    def __init__(self, shared_memory: "SharedMemory") -> None:  # noqa: F821
        self._mem = shared_memory

    def append(self, result: ExperimentResult) -> None:
        self._mem.append_to_list(_EXPERIMENT_KEY, result.model_dump())

    def all(self) -> list[ExperimentResult]:
        raw_list: list[dict] = self._mem.get_list(_EXPERIMENT_KEY)
        return [ExperimentResult(**item) for item in raw_list]

    def best(self, is_lower_better: bool = True) -> ExperimentResult | None:
        results = self.all()
        if not results:
            return None
        return min(results, key=lambda r: r.metric_value) if is_lower_better \
            else max(results, key=lambda r: r.metric_value)

    def count(self) -> int:
        return len(self._mem.get_list(_EXPERIMENT_KEY))

    def clear(self) -> None:
        self._mem.delete(_EXPERIMENT_KEY)

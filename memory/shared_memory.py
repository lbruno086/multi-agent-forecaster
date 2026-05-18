from __future__ import annotations

import json
from typing import Any


class SharedMemory:
    """
    Process-scoped key/value store.

    Backed by an injected backend (MemoryBackend or RedisBackend) so the
    same interface works for both in-process and distributed deployments.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    # ── Basic KV ─────────────────────────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        serialized = json.dumps(value, default=str)
        self._backend.set(key, serialized)

    def get(self, key: str, default: Any = None) -> Any:
        raw = self._backend.get(key)
        if raw is None:
            return default
        return json.loads(raw)

    def delete(self, key: str) -> None:
        self._backend.delete(key)

    def exists(self, key: str) -> bool:
        return self._backend.exists(key)

    # ── Convenience helpers ───────────────────────────────────────────────────

    def increment(self, key: str, by: int = 1) -> int:
        # Non-atomic: safe for single-process use only.
        # Replace with backend-native INCR when wiring Redis.
        current = self.get(key, 0)
        new_value = int(current) + by
        self.set(key, new_value)
        return new_value

    def append_to_list(self, key: str, item: Any) -> None:
        # Non-atomic: safe for single-process use only.
        # Replace with backend-native RPUSH when wiring Redis.
        current: list = self.get(key, [])
        current.append(item)
        self.set(key, current)

    def get_list(self, key: str) -> list:
        return self.get(key, [])

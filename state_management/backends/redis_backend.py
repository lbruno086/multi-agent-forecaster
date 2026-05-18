from __future__ import annotations


class RedisBackend:
    """Redis-backed key/value store. Requires redis package and a running Redis server."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0) -> None:
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "redis package is required for RedisBackend. "
                "Install it with: pip install redis"
            ) from exc

        self._client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._client.ping()

    def get(self, key: str) -> str | None:
        return self._client.get(key)

    def set(self, key: str, value: str) -> None:
        self._client.set(key, value)

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def exists(self, key: str) -> bool:
        return bool(self._client.exists(key))

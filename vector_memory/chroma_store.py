from __future__ import annotations

from pathlib import Path
from typing import Any

_DEFAULT_PERSIST_DIR = Path(__file__).parent.parent / "vector_memory" / "chroma_db"
_COLLECTION_NAME = "research_memory"


class ChromaStore:
    """
    Local ChromaDB store for research findings.

    Lazy-imports chromadb so the rest of the system works without it installed.
    """

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._persist_dir = persist_dir or _DEFAULT_PERSIST_DIR
        self._client = None
        self._collection = None

    def _ensure_ready(self) -> None:
        if self._collection is not None:
            return
        try:
            import chromadb  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required for vector memory. "
                "Install it with: pip install chromadb"
            ) from exc

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def store(self, text: str, metadata: dict[str, Any], doc_id: str | None = None) -> str:
        self._ensure_ready()
        if doc_id is None:
            import hashlib
            doc_id = hashlib.md5(text.encode()).hexdigest()  # noqa: S324

        safe_meta: dict[str, str | int | float | bool] = {
            k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in metadata.items()
        }

        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[safe_meta],
        )
        return doc_id

    def search(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        self._ensure_ready()
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(n, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        output: list[dict[str, Any]] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({"text": doc, "metadata": meta, "distance": dist})
        return output

    def count(self) -> int:
        self._ensure_ready()
        return self._collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection — useful in tests."""
        self._ensure_ready()
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = None

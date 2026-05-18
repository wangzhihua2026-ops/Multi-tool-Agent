import pytest

from app.core.config import Settings
from app.rag.vector_store import (
    DualWriteFallbackVectorStore,
    InMemoryVectorStore,
    VectorMatch,
    VectorStore,
    build_vector_store,
)


class FailingPrimaryVectorStore(VectorStore):
    backend_name = "failing"

    def clear(self) -> None:
        raise RuntimeError("primary clear failed")

    def upsert(self, chunk_id: str, vector: list[float]) -> None:
        raise RuntimeError("primary upsert failed")

    def delete(self, chunk_ids: list[str]) -> None:
        raise RuntimeError("primary delete failed")

    def replace_all(self, entries: dict[str, list[float]]) -> None:
        raise RuntimeError("primary replace_all failed")

    def count(self) -> int:
        raise RuntimeError("primary count failed")

    def search(self, query_vector: list[float], top_k: int = 5) -> list[VectorMatch]:
        raise RuntimeError("primary search failed")


def test_dual_write_fallback_store_keeps_local_copy_without_query_fallback() -> None:
    fallback = InMemoryVectorStore()
    store = DualWriteFallbackVectorStore(
        primary=FailingPrimaryVectorStore(),
        fallback=fallback,
    )

    store.upsert("chunk-1", [1.0, 0.0, 0.0])
    assert fallback.count() == 1
    assert store.count() == 1

    with pytest.raises(RuntimeError, match="query fallback is disabled"):
        store.search([1.0, 0.0, 0.0], top_k=1)


def test_dual_write_fallback_store_can_opt_into_query_fallback() -> None:
    fallback = InMemoryVectorStore()
    store = DualWriteFallbackVectorStore(
        primary=FailingPrimaryVectorStore(),
        fallback=fallback,
        allow_query_fallback=True,
    )

    store.upsert("chunk-1", [1.0, 0.0, 0.0])
    matches = store.search([1.0, 0.0, 0.0], top_k=1)
    assert len(matches) == 1
    assert matches[0].chunk_id == "chunk-1"


def test_build_vector_store_returns_memory_backend_by_default() -> None:
    store = build_vector_store(Settings(vector_store_provider="memory"))
    assert store.backend_name == "memory"


def test_build_vector_store_wraps_qdrant_with_fallback() -> None:
    store = build_vector_store(
        Settings(
            vector_store_provider="qdrant",
            vector_store_url="http://127.0.0.1:6333",
            vector_store_collection="test-collection",
        )
    )
    assert store.backend_name == "qdrant+fallback"


def test_build_vector_store_supports_local_qdrant_persistence(tmp_path) -> None:
    store = build_vector_store(
        Settings(
            vector_store_provider="qdrant_local",
            vector_store_path=str(tmp_path / "qdrant-local"),
            vector_store_collection="test-collection",
            embedding_dimensions=3,
        )
    )

    store.upsert("chunk-1", [1.0, 0.0, 0.0])
    store.upsert("chunk-2", [0.0, 1.0, 0.0])

    assert store.backend_name == "qdrant_local"
    assert store.count() == 2

    matches = store.search([1.0, 0.0, 0.0], top_k=1)
    assert len(matches) == 1
    assert matches[0].chunk_id == "chunk-1"
    store.close()

import pytest

from app.rag.embeddings import EmbeddingProvider, FallbackEmbeddingProvider, HashEmbeddingProvider
from app.rag.models import DocumentCreateRequest
from app.rag.retriever import KnowledgeRetriever
from app.rag.store import InMemoryKnowledgeStore
from app.rag.vector_store import InMemoryVectorStore, VectorMatch, VectorStore
from app.services.document_service import DocumentService


class FailingOnSecondUpsertVectorStore(VectorStore):
    backend_name = "failing"

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._upsert_calls = 0

    def clear(self) -> None:
        self._vectors.clear()

    def upsert(self, chunk_id: str, vector: list[float]) -> None:
        self._upsert_calls += 1
        if self._upsert_calls == 2:
            raise RuntimeError("vector write failed")
        self._vectors[chunk_id] = list(vector)

    def delete(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self._vectors.pop(chunk_id, None)

    def replace_all(self, entries: dict[str, list[float]]) -> None:
        self._vectors = {chunk_id: list(vector) for chunk_id, vector in entries.items()}

    def count(self) -> int:
        return len(self._vectors)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[VectorMatch]:
        return []


class FailingPassageEmbeddingProvider(EmbeddingProvider):
    provider_name = "failing-passages"

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding outage")


def test_create_document_marks_failed_status_and_rolls_back_vectors() -> None:
    store = InMemoryKnowledgeStore()
    vector_store = FailingOnSecondUpsertVectorStore()
    embedding_provider = HashEmbeddingProvider(dimensions=8)
    retriever = KnowledgeRetriever(store, vector_store, embedding_provider)
    service = DocumentService(
        store=store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        retriever=retriever,
        chunk_size=24,
        chunk_overlap=0,
    )

    created = service.create_document(
        DocumentCreateRequest(
            title="partial-index-doc",
            content=(
                "First chunk covers deployment. "
                "Second chunk covers startup recovery. "
                "Third chunk covers verification."
            ),
            metadata={},
        )
    )

    assert created.index_status == "failed"
    assert created.index_error == "vector write failed"
    assert vector_store.count() == 0

    stored_document = service.get_document(created.document_id)
    assert stored_document.index_status == "failed"
    assert stored_document.index_error == "vector write failed"
    assert all(chunk.embedding_provider is None for chunk in store.get_document_chunks(created.document_id))


def test_reindex_keeps_existing_vectors_when_embeddings_fail() -> None:
    store = InMemoryKnowledgeStore()
    vector_store = InMemoryVectorStore()
    healthy_provider = HashEmbeddingProvider(dimensions=8)
    retriever = KnowledgeRetriever(store, vector_store, healthy_provider)
    healthy_service = DocumentService(
        store=store,
        vector_store=vector_store,
        embedding_provider=healthy_provider,
        retriever=retriever,
    )

    created = healthy_service.create_document(
        DocumentCreateRequest(
            title="reindex-safety",
            content="Reindex should keep the old searchable vectors until the new embeddings are ready.",
            metadata={},
        )
    )
    assert created.index_status == "ready"
    assert vector_store.count() == created.chunk_count

    failing_service = DocumentService(
        store=store,
        vector_store=vector_store,
        embedding_provider=FailingPassageEmbeddingProvider(),
        retriever=retriever,
    )

    with pytest.raises(RuntimeError, match="embedding outage"):
        failing_service.reindex_documents(clear_vector_store=True)

    assert vector_store.count() == created.chunk_count
    refreshed = healthy_service.get_document(created.document_id)
    assert refreshed.index_status == "ready"
    assert refreshed.index_error is None

    hits = healthy_service.search("old searchable vectors", top_k=1)
    assert len(hits) == 1
    assert hits[0].document_title == "reindex-safety"


def test_fallback_embedding_records_actual_vector_signature() -> None:
    store = InMemoryKnowledgeStore()
    vector_store = InMemoryVectorStore()
    fallback_provider = HashEmbeddingProvider(dimensions=8)
    embedding_provider = FallbackEmbeddingProvider(
        primary=FailingPassageEmbeddingProvider(),
        fallback=fallback_provider,
    )
    retriever = KnowledgeRetriever(store, vector_store, embedding_provider)
    service = DocumentService(
        store=store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        retriever=retriever,
    )

    created = service.create_document(
        DocumentCreateRequest(
            title="fallback-signature",
            content="Fallback vectors should be labeled with the actual provider.",
            metadata={},
        )
    )

    assert created.index_status == "ready"
    chunks = store.get_document_chunks(created.document_id)
    assert chunks
    assert {chunk.embedding_provider for chunk in chunks} == {"hash:8"}

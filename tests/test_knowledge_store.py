import pytest

from app.core.config import Settings
from app.persistence import knowledge_repository
from app.persistence.knowledge_repository import SqliteKnowledgeStore, build_knowledge_store
from app.rag.embeddings import HashEmbeddingProvider
from app.rag.models import ChunkRecord, DocumentCreateRequest, DocumentRecord
from app.rag.retriever import KnowledgeRetriever
from app.rag.vector_store import InMemoryVectorStore
from app.services.document_service import DocumentService


def test_sqlite_knowledge_store_persists_documents_and_chunks(tmp_path) -> None:
    database_path = tmp_path / "knowledge_base.db"
    store = SqliteKnowledgeStore(str(database_path))
    document = DocumentRecord(
        title="deployment-guide",
        content="Start the service and check the health endpoint.",
        metadata={"team": "platform"},
    )
    chunk = ChunkRecord(
        document_id=document.document_id,
        document_title=document.title,
        index=0,
        content="Start the service and check the health endpoint.",
        tokens=["start", "service", "health", "endpoint"],
        embedding_provider="hash",
    )
    store.add_document(document, [chunk])

    reopened_store = SqliteKnowledgeStore(str(database_path))
    summaries = reopened_store.list_documents()
    assert len(summaries) == 1
    assert summaries[0].document_id == document.document_id
    assert summaries[0].chunk_count == 1
    assert summaries[0].metadata == {"team": "platform"}
    assert summaries[0].index_status == "ready"
    assert summaries[0].index_error is None

    loaded_document = reopened_store.get_document(document.document_id)
    assert loaded_document.content == document.content
    assert loaded_document.index_status == "ready"
    assert loaded_document.index_error is None

    loaded_chunk = reopened_store.get_chunk(chunk.chunk_id)
    assert loaded_chunk is not None
    assert loaded_chunk.tokens == chunk.tokens
    assert loaded_chunk.embedding_provider == "hash"

    reopened_store.set_chunk_embedding_provider(chunk.chunk_id, "sentence_transformers")
    refreshed_chunk = SqliteKnowledgeStore(str(database_path)).get_chunk(chunk.chunk_id)
    assert refreshed_chunk is not None
    assert refreshed_chunk.embedding_provider == "sentence_transformers"

    reopened_store.set_document_index_status(document.document_id, "failed", "embedding timeout")
    refreshed_document = SqliteKnowledgeStore(str(database_path)).get_document(document.document_id)
    assert refreshed_document.index_status == "failed"
    assert refreshed_document.index_error == "embedding timeout"


def test_document_service_rebuilds_vectors_from_persisted_documents(tmp_path) -> None:
    database_path = tmp_path / "knowledge_base.db"
    embedding_provider = HashEmbeddingProvider()

    initial_store = SqliteKnowledgeStore(str(database_path))
    initial_vector_store = InMemoryVectorStore()
    initial_retriever = KnowledgeRetriever(initial_store, initial_vector_store, embedding_provider)
    initial_service = DocumentService(
        store=initial_store,
        vector_store=initial_vector_store,
        embedding_provider=embedding_provider,
        retriever=initial_retriever,
    )

    created = initial_service.create_document(
        DocumentCreateRequest(
            title="operations-notes",
            content="Deploy the FastAPI service first, then validate the health endpoint.",
            metadata={"source": "runbook"},
        )
    )
    assert initial_vector_store.count() == created.chunk_count

    restarted_store = SqliteKnowledgeStore(str(database_path))
    restarted_vector_store = InMemoryVectorStore()
    restarted_retriever = KnowledgeRetriever(restarted_store, restarted_vector_store, embedding_provider)
    restarted_service = DocumentService(
        store=restarted_store,
        vector_store=restarted_vector_store,
        embedding_provider=embedding_provider,
        retriever=restarted_retriever,
    )

    listed = restarted_service.list_documents()
    assert len(listed) == 1
    assert listed[0].document_id == created.document_id
    assert restarted_service.get_document(created.document_id).content.startswith("Deploy the FastAPI service")

    assert restarted_vector_store.count() == 0
    reindex_summary = restarted_service.reindex_documents(clear_vector_store=True)
    assert reindex_summary.document_count == 1
    assert reindex_summary.chunk_count == created.chunk_count
    assert restarted_vector_store.count() == created.chunk_count

    hits = restarted_service.search("How do I validate the health endpoint after deployment?", top_k=1)
    assert len(hits) == 1
    assert hits[0].document_title == "operations-notes"


def test_build_knowledge_store_supports_sqlite_by_default(tmp_path) -> None:
    store = build_knowledge_store(
        Settings(
            knowledge_store_provider="sqlite",
            knowledge_store_path=str(tmp_path / "knowledge_base.db"),
        )
    )
    assert isinstance(store, SqliteKnowledgeStore)


def test_build_knowledge_store_requires_database_url_for_postgres() -> None:
    with pytest.raises(ValueError, match="KNOWLEDGE_STORE_DATABASE_URL"):
        build_knowledge_store(Settings(knowledge_store_provider="postgres"))


def test_build_knowledge_store_reports_missing_psycopg_dependency(monkeypatch) -> None:
    def fail_loader():
        raise RuntimeError("missing psycopg")

    monkeypatch.setattr(knowledge_repository, "_load_psycopg", fail_loader)

    with pytest.raises(RuntimeError, match="missing psycopg"):
        build_knowledge_store(
            Settings(
                knowledge_store_provider="postgres",
                knowledge_store_database_url="postgresql://demo:demo@localhost:5432/demo",
            )
        )

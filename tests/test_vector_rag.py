from app.rag.embeddings import HashEmbeddingProvider
from app.rag.models import DocumentCreateRequest
from app.rag.retriever import KnowledgeRetriever
from app.rag.store import InMemoryKnowledgeStore
from app.rag.vector_store import InMemoryVectorStore
from app.services.document_service import DocumentService


def test_document_ingestion_populates_vector_index() -> None:
    store = InMemoryKnowledgeStore()
    vector_store = InMemoryVectorStore()
    embedding_provider = HashEmbeddingProvider()
    retriever = KnowledgeRetriever(store, vector_store, embedding_provider)
    service = DocumentService(
        store=store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        retriever=retriever,
    )

    summary = service.create_document(
        DocumentCreateRequest(
            title="deployment-guide",
            content="Deployment steps: configure environment variables, start the FastAPI service, then check the health endpoint.",
            metadata={},
        )
    )

    assert summary.chunk_count >= 1
    assert vector_store.count() == summary.chunk_count


def test_vector_retriever_returns_ranked_hits() -> None:
    store = InMemoryKnowledgeStore()
    vector_store = InMemoryVectorStore()
    embedding_provider = HashEmbeddingProvider()
    retriever = KnowledgeRetriever(store, vector_store, embedding_provider)
    service = DocumentService(
        store=store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        retriever=retriever,
    )

    service.create_document(
        DocumentCreateRequest(
            title="deployment-guide",
            content="Deployment steps: configure environment variables, start the FastAPI service, then check the health endpoint.",
            metadata={},
        )
    )
    service.create_document(
        DocumentCreateRequest(
            title="incident-playbook",
            content="Incident response steps: gather logs, identify the failing dependency, and notify the on-call engineer.",
            metadata={},
        )
    )

    hits = service.search("How do I deploy the FastAPI service?", top_k=2)
    assert len(hits) >= 1
    assert hits[0].document_title == "deployment-guide"
    assert hits[0].vector_score is not None

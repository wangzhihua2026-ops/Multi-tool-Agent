import pytest

from app.rag.embeddings import EmbeddingProvider, HashEmbeddingProvider
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


def test_retriever_supports_strategy_ablation() -> None:
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
            title="approval-guide",
            content="Risky tools pause in waiting_approval until a reviewer approves them.",
            metadata={},
        )
    )

    lexical_hits = retriever.search("waiting_approval", top_k=1, strategy="lexical")
    vector_hits = retriever.search("waiting_approval", top_k=1, strategy="vector")

    assert lexical_hits[0].retrieval_mode == "lexical"
    assert vector_hits[0].retrieval_mode == "vector"


def test_retriever_rejects_unknown_strategy() -> None:
    retriever = KnowledgeRetriever(
        InMemoryKnowledgeStore(),
        InMemoryVectorStore(),
        HashEmbeddingProvider(),
    )

    with pytest.raises(ValueError, match="Unknown retrieval strategy"):
        retriever.search("deployment", strategy="reranker")


class FixedEmbeddingProvider(EmbeddingProvider):
    provider_name = "fixed"

    def embed_text(self, text: str) -> list[float]:
        if "needle" in text.lower():
            return [1.0, 0.0]
        return [0.99, 0.0]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_hybrid_rank_fusion_does_not_penalize_multi_signal_hit() -> None:
    provider = FixedEmbeddingProvider()
    store = InMemoryKnowledgeStore()
    vector_store = InMemoryVectorStore()
    retriever = KnowledgeRetriever(
        store,
        vector_store,
        provider,
    )
    service = DocumentService(
        store=store,
        vector_store=vector_store,
        embedding_provider=provider,
        retriever=retriever,
    )
    service.create_document(
        DocumentCreateRequest(
            title="related",
            content="needle " + " ".join(f"detail-{index}" for index in range(40)),
            metadata={},
        )
    )
    service.create_document(
        DocumentCreateRequest(
            title="vector-only-distractor",
            content="unrelated semantic neighbour",
            metadata={},
        )
    )

    hits = retriever.search("needle", top_k=1, strategy="hybrid")

    assert hits[0].document_title == "related"

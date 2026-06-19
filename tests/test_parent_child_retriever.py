from app.rag.embeddings import HashEmbeddingProvider
from app.rag.models import DocumentCreateRequest
from app.rag.retriever import KnowledgeRetriever
from app.rag.store import InMemoryKnowledgeStore
from app.rag.vector_store import InMemoryVectorStore
from app.services.document_service import DocumentService


def test_parent_child_search_returns_parent_content() -> None:
    store = InMemoryKnowledgeStore()
    vector_store = InMemoryVectorStore()
    embedding_provider = HashEmbeddingProvider(dimensions=8)
    retriever = KnowledgeRetriever(store, vector_store, embedding_provider)
    service = DocumentService(
        store=store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        retriever=retriever,
        indexing_mode="hierarchical",
        parent_chunk_size=140,
        parent_chunk_overlap=0,
        child_chunk_size=45,
        child_chunk_overlap=0,
    )
    service.create_document(
        DocumentCreateRequest(
            title="incident-guide",
            content=(
                "Startup diagnostics include checking logs and environment variables "
                "before restarting services."
            ),
            metadata={},
        )
    )

    hits = service.search("environment variables", top_k=1, strategy="parent_child")

    assert hits
    assert hits[0].retrieval_mode == "parent_child_hybrid"
    assert "Startup diagnostics" in hits[0].content
    assert hits[0].evidence_chunk_ids
    assert hits[0].aggregated_child_score is not None


def test_parent_child_rerank_strategy_uses_noop_when_disabled() -> None:
    store = InMemoryKnowledgeStore()
    vector_store = InMemoryVectorStore()
    embedding_provider = HashEmbeddingProvider(dimensions=8)
    retriever = KnowledgeRetriever(store, vector_store, embedding_provider)
    service = DocumentService(
        store=store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        retriever=retriever,
        indexing_mode="hierarchical",
        parent_chunk_size=100,
        parent_chunk_overlap=0,
        child_chunk_size=40,
        child_chunk_overlap=0,
        reranker_provider="none",
    )
    service.create_document(
        DocumentCreateRequest(
            title="approval-guide",
            content="Approval workflows pause risky tools until a reviewer explicitly approves them.",
            metadata={},
        )
    )

    hits = service.search("risky tools reviewer", top_k=1, strategy="parent_child_rerank")

    assert hits
    assert hits[0].retrieval_mode == "parent_child_rerank"
    assert hits[0].rerank_score is None

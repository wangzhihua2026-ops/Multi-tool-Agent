import logging

from app.rag.chunking import chunk_text
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import (
    ChunkRecord,
    DocumentCreateRequest,
    DocumentDetail,
    DocumentRecord,
    DocumentReindexSummary,
    DocumentSummary,
    SearchHit,
)
from app.rag.retriever import KnowledgeRetriever
from app.rag.text import tokenize_text
from app.rag.store import KnowledgeStore
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        store: KnowledgeStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        retriever: KnowledgeRetriever,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
    ) -> None:
        self.store = store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.retriever = retriever
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_document(self, request: DocumentCreateRequest) -> DocumentSummary:
        document = DocumentRecord(
            title=request.title,
            content=request.content,
            metadata=request.metadata,
            index_status="pending",
        )
        chunk_contents = chunk_text(
            request.content,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = [
            ChunkRecord(
                document_id=document.document_id,
                document_title=document.title,
                index=index,
                content=chunk_content,
                tokens=tokenize_text(chunk_content),
            )
            for index, chunk_content in enumerate(chunk_contents)
        ]
        self.store.add_document(document, chunks)
        try:
            embeddings = self.embedding_provider.embed_passages(chunk_contents)
            self._assert_matching_embeddings(chunks, embeddings)
            embedding_signature = self.embedding_provider.last_embedding_signature
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                self.vector_store.upsert(chunk.chunk_id, embedding)
                self.store.set_chunk_embedding_provider(chunk.chunk_id, embedding_signature)
            self.store.set_document_index_status(document.document_id, "ready")
        except Exception as exc:
            self._rollback_chunk_vectors(chunks)
            for chunk in chunks:
                self.store.set_chunk_embedding_provider(chunk.chunk_id, None)
            self.store.set_document_index_status(document.document_id, "failed", str(exc))

        return self._build_summary(
            document=self.store.get_document(document.document_id),
            chunk_count=len(chunks),
        )

    def list_documents(self) -> list[DocumentSummary]:
        return self.store.list_documents()

    def get_document(self, document_id: str) -> DocumentDetail:
        document = self.store.get_document(document_id)
        chunks = self.store.get_document_chunks(document_id)
        return DocumentDetail(
            document_id=document.document_id,
            title=document.title,
            metadata=document.metadata,
            chunk_count=len(chunks),
            index_status=document.index_status,
            index_error=document.index_error,
            created_at=document.created_at,
            content=document.content,
        )

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        return self.retriever.search(query=query, top_k=top_k)

    def reindex_documents(self, clear_vector_store: bool = True) -> DocumentReindexSummary:
        chunks = self.store.get_chunks()
        documents = self.store.list_documents()

        if chunks:
            embeddings = self.embedding_provider.embed_passages([chunk.content for chunk in chunks])
            self._assert_matching_embeddings(chunks, embeddings)
            embedding_signature = self.embedding_provider.last_embedding_signature
            vectors = {
                chunk.chunk_id: embedding
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            }
            if clear_vector_store:
                self.vector_store.replace_all(vectors)
            else:
                for chunk_id, embedding in vectors.items():
                    self.vector_store.upsert(chunk_id, embedding)
            for chunk in chunks:
                self.store.set_chunk_embedding_provider(chunk.chunk_id, embedding_signature)
            for document in documents:
                self.store.set_document_index_status(document.document_id, "ready")

        return DocumentReindexSummary(
            document_count=len(documents),
            chunk_count=len(chunks),
            embedding_provider=self.embedding_provider.last_embedding_signature,
            vector_store_backend=self.vector_store.backend_name,
            cleared_vector_store=clear_vector_store,
        )

    def _build_summary(self, document: DocumentRecord, chunk_count: int) -> DocumentSummary:
        return DocumentSummary(
            document_id=document.document_id,
            title=document.title,
            metadata=document.metadata,
            chunk_count=chunk_count,
            index_status=document.index_status,
            index_error=document.index_error,
            created_at=document.created_at,
        )

    @staticmethod
    def _assert_matching_embeddings(chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Embedding provider returned "
                f"{len(embeddings)} vectors for {len(chunks)} chunks."
            )

    def _rollback_chunk_vectors(self, chunks: list[ChunkRecord]) -> None:
        try:
            self.vector_store.delete([chunk.chunk_id for chunk in chunks])
        except Exception as exc:  # pragma: no cover - best effort rollback logging
            logger.warning("Document vector rollback failed: %s", exc)

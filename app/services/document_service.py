import logging

from app.rag.chunking import chunk_text
from app.rag.embeddings import EmbeddingProvider
from app.rag.hierarchical_chunking import split_child_chunks, split_parent_blocks
from app.rag.models import (
    ChunkRecord,
    DocumentCreateRequest,
    DocumentDetail,
    DocumentRecord,
    DocumentReindexSummary,
    DocumentSummary,
    ParentBlockRecord,
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
        indexing_mode: str = "flat",
        default_retrieval_strategy: str = "hybrid",
        parent_chunk_size: int = 1600,
        parent_chunk_overlap: int = 160,
        child_chunk_size: int = 450,
        child_chunk_overlap: int = 80,
        reranker_provider: str = "none",
    ) -> None:
        self.store = store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.retriever = retriever
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.indexing_mode = indexing_mode
        self.default_retrieval_strategy = default_retrieval_strategy
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.reranker_provider = reranker_provider

    def create_document(self, request: DocumentCreateRequest) -> DocumentSummary:
        document = DocumentRecord(
            title=request.title,
            content=request.content,
            metadata=request.metadata,
            index_status="pending",
        )
        chunks, parent_blocks = self._build_index_records(document)
        self.store.add_document(document, chunks, parent_blocks=parent_blocks)
        try:
            embeddings = self.embedding_provider.embed_passages([chunk.content for chunk in chunks])
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

    def search(self, query: str, top_k: int = 3, strategy: str | None = None) -> list[SearchHit]:
        return self.retriever.search(
            query=query,
            top_k=top_k,
            strategy=strategy or self.default_retrieval_strategy,
        )

    def reindex_documents(self, clear_vector_store: bool = True) -> DocumentReindexSummary:
        documents = self.store.list_documents()
        prepared_records = [
            self._prepare_reindex_document(summary.document_id)
            for summary in documents
        ]
        chunks = [
            chunk
            for _, document_chunks, _ in prepared_records
            for chunk in document_chunks
        ]

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
            for document, document_chunks, parent_blocks in prepared_records:
                ready_document = document.model_copy(update={"index_status": "ready", "index_error": None})
                for chunk in document_chunks:
                    chunk.embedding_provider = embedding_signature
                self.store.add_document(ready_document, document_chunks, parent_blocks=parent_blocks)
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

    def _prepare_reindex_document(
        self,
        document_id: str,
    ) -> tuple[DocumentRecord, list[ChunkRecord], list[ParentBlockRecord]]:
        document = self.store.get_document(document_id)
        chunks, parent_blocks = self._build_index_records(document)
        return document, chunks, parent_blocks

    def _build_index_records(self, document: DocumentRecord) -> tuple[list[ChunkRecord], list[ParentBlockRecord]]:
        if self.indexing_mode.strip().lower() == "hierarchical":
            return self._build_hierarchical_index_records(document)

        chunk_contents = chunk_text(
            document.content,
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
        return chunks, []

    def _build_hierarchical_index_records(
        self,
        document: DocumentRecord,
    ) -> tuple[list[ChunkRecord], list[ParentBlockRecord]]:
        parent_text_blocks = split_parent_blocks(
            document.content,
            parent_size=self.parent_chunk_size,
            parent_overlap=self.parent_chunk_overlap,
        )
        parent_blocks = [
            ParentBlockRecord(
                document_id=document.document_id,
                document_title=document.title,
                index=parent_block.index,
                content=parent_block.content,
                tokens=tokenize_text(parent_block.content),
                start_offset=parent_block.start_offset,
                end_offset=parent_block.end_offset,
                metadata=document.metadata,
            )
            for parent_block in parent_text_blocks
        ]
        chunks: list[ChunkRecord] = []
        for parent_block, parent_record in zip(parent_text_blocks, parent_blocks, strict=True):
            child_text_blocks = split_child_chunks(
                parent_block,
                child_size=self.child_chunk_size,
                child_overlap=self.child_chunk_overlap,
            )
            for child_block in child_text_blocks:
                chunks.append(
                    ChunkRecord(
                        document_id=document.document_id,
                        document_title=document.title,
                        index=len(chunks),
                        content=child_block.content,
                        tokens=tokenize_text(child_block.content),
                        parent_id=parent_record.parent_id,
                        parent_index=parent_record.index,
                        start_offset=child_block.start_offset,
                        end_offset=child_block.end_offset,
                    )
                )
        return chunks, parent_blocks

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

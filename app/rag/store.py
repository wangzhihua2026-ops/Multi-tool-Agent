from threading import RLock
from typing import Protocol

from app.core.exceptions import DocumentNotFoundError
from app.rag.models import ChunkRecord, DocumentRecord, DocumentSummary


class KnowledgeStore(Protocol):
    def clear(self) -> None:
        ...

    def add_document(self, document: DocumentRecord, chunks: list[ChunkRecord]) -> None:
        ...

    def list_documents(self) -> list[DocumentSummary]:
        ...

    def get_document(self, document_id: str) -> DocumentRecord:
        ...

    def get_document_chunks(self, document_id: str) -> list[ChunkRecord]:
        ...

    def get_chunks(self) -> list[ChunkRecord]:
        ...

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        ...

    def set_chunk_embedding_provider(self, chunk_id: str, embedding_provider: str | None) -> None:
        ...

    def set_document_index_status(
        self,
        document_id: str,
        index_status: str,
        index_error: str | None = None,
    ) -> None:
        ...


class InMemoryKnowledgeStore:
    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}
        self._chunks_by_document: dict[str, list[ChunkRecord]] = {}
        self._chunks_by_id: dict[str, ChunkRecord] = {}
        self._lock = RLock()

    def clear(self) -> None:
        with self._lock:
            self._documents.clear()
            self._chunks_by_document.clear()
            self._chunks_by_id.clear()

    def add_document(self, document: DocumentRecord, chunks: list[ChunkRecord]) -> None:
        with self._lock:
            self._documents[document.document_id] = document
            self._chunks_by_document[document.document_id] = list(chunks)
            for chunk in chunks:
                self._chunks_by_id[chunk.chunk_id] = chunk

    def list_documents(self) -> list[DocumentSummary]:
        with self._lock:
            summaries = [
                DocumentSummary(
                    document_id=document.document_id,
                    title=document.title,
                    metadata=document.metadata,
                    chunk_count=len(self._chunks_by_document.get(document.document_id, [])),
                    index_status=document.index_status,
                    index_error=document.index_error,
                    created_at=document.created_at,
                )
                for document in self._documents.values()
            ]

        return sorted(summaries, key=lambda item: item.created_at, reverse=True)

    def get_document(self, document_id: str) -> DocumentRecord:
        with self._lock:
            document = self._documents.get(document_id)

        if document is None:
            raise DocumentNotFoundError(document_id)

        return document

    def get_document_chunks(self, document_id: str) -> list[ChunkRecord]:
        with self._lock:
            if document_id not in self._documents:
                raise DocumentNotFoundError(document_id)
            return list(self._chunks_by_document.get(document_id, []))

    def get_chunks(self) -> list[ChunkRecord]:
        with self._lock:
            chunks: list[ChunkRecord] = []
            for document_chunks in self._chunks_by_document.values():
                chunks.extend(document_chunks)

        return chunks

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        with self._lock:
            return self._chunks_by_id.get(chunk_id)

    def set_chunk_embedding_provider(self, chunk_id: str, embedding_provider: str | None) -> None:
        with self._lock:
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is not None:
                chunk.embedding_provider = embedding_provider

    def set_document_index_status(
        self,
        document_id: str,
        index_status: str,
        index_error: str | None = None,
    ) -> None:
        with self._lock:
            document = self._documents.get(document_id)
            if document is not None:
                document.index_status = index_status
                document.index_error = index_error

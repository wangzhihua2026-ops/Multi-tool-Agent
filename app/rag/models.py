from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentFileUploadRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    file_name: str = Field(min_length=1, max_length=260)
    content_type: str | None = Field(default=None, max_length=200)
    content_base64: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentReindexRequest(BaseModel):
    clear_vector_store: bool = True


class DocumentRecord(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)
    index_status: str = "ready"
    index_error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChunkRecord(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    document_title: str
    index: int
    content: str
    tokens: list[str] = Field(default_factory=list)
    embedding_provider: str | None = None


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    metadata: dict[str, str] = Field(default_factory=dict)
    chunk_count: int
    index_status: str = "ready"
    index_error: str | None = None
    created_at: datetime


class DocumentDetail(DocumentSummary):
    content: str


class DocumentReindexSummary(BaseModel):
    document_count: int
    chunk_count: int
    embedding_provider: str
    vector_store_backend: str
    cleared_vector_store: bool


class SearchHit(BaseModel):
    document_id: str
    document_title: str
    chunk_id: str
    chunk_index: int
    content: str
    score: float
    retrieval_mode: str = "hybrid"
    lexical_score: float | None = None
    vector_score: float | None = None

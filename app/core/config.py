from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Multi Tool Agent"
    environment: str = "local"
    log_level: str = "INFO"
    api_prefix: str = "/api"
    api_auth_token: str | None = None
    api_allow_remote_without_token: bool = False
    llm_provider: str = "mock"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_timeout_seconds: int = 30
    default_model: str = "gpt-4.1-mini"
    embedding_provider: str = "hash"
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 128
    embedding_api_key: str | None = None
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16
    embedding_query_prefix: str = ""
    embedding_passage_prefix: str = ""
    embedding_trust_remote_code: bool = False
    embedding_cache_path: str | None = None
    embedding_local_files_only: bool = False
    embedding_fallback_enabled: bool = True
    knowledge_store_provider: str = "sqlite"
    knowledge_store_path: str = "./data/knowledge_base.db"
    knowledge_store_database_url: str | None = None
    vector_store_provider: str = "memory"
    vector_store_url: str = "http://127.0.0.1:6333"
    vector_store_collection: str = "multi_tool_agent_chunks"
    vector_store_path: str = "./data/qdrant"
    vector_store_timeout_seconds: int = 15
    vector_store_api_key: str | None = None
    vector_store_query_fallback_enabled: bool = False
    rag_indexing_mode: str = "flat"
    default_retrieval_strategy: str = "hybrid"
    parent_chunk_size: int = 1600
    parent_chunk_overlap: int = 160
    child_chunk_size: int = 450
    child_chunk_overlap: int = 80
    bm25_enabled: bool = True
    bm25_top_k_multiplier: int = 8
    advanced_vector_top_k_multiplier: int = 8
    advanced_min_child_candidates: int = 24
    advanced_parent_candidate_limit: int = 20
    advanced_vector_weight: float = 0.55
    advanced_bm25_weight: float = 0.45
    advanced_evidence_bonus: float = 0.01
    advanced_rrf_constant: int = 60
    reranker_provider: str = "none"
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "cpu"
    reranker_batch_size: int = 8
    reranker_fallback_enabled: bool = True
    session_history_limit: int = 12
    mcp_config_path: str = "./config/mcp_servers.json"
    max_tool_steps: int = 3
    max_tool_retries: int = 3
    knowledge_base_enabled: bool = False
    run_storage_path: str = "./data/multi_tool_agent.db"
    platform_database_url: str | None = None
    redis_url: str = "redis://127.0.0.1:6379/0"
    agent_worker_enabled: bool = False
    agent_worker_lease_seconds: int = 60
    agent_worker_heartbeat_seconds: int = 20
    agent_run_max_attempts: int = 3
    agent_outbox_poll_seconds: float = 1.0
    agent_recovery_poll_seconds: float = 5.0
    document_upload_max_bytes: int = 20 * 1024 * 1024
    document_upload_max_extracted_chars: int = 2_000_000
    document_pdf_max_pages: int = 300
    document_ocr_enabled: bool = True
    document_ocr_max_pages: int = 50
    document_ocr_min_native_chars: int = 50
    extraction_export_path: str = "./data/exports"
    llm_api_key: str | None = None
    openai_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def resolved_llm_api_key(self) -> str | None:
        return self.llm_api_key or self.openai_api_key

    @property
    def resolved_embedding_api_key(self) -> str | None:
        return self.embedding_api_key or self.resolved_llm_api_key

    @model_validator(mode="after")
    def validate_worker_lease(self) -> "Settings":
        if (
            self.agent_worker_enabled
            and self.agent_worker_heartbeat_seconds >= self.agent_worker_lease_seconds
        ):
            raise ValueError("Agent worker heartbeat must be shorter than its lease.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

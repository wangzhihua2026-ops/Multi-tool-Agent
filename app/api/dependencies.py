from functools import lru_cache

from app.agent.runtime import AgentRuntime
from app.core.config import Settings, get_settings
from app.core.llm_gateway import LLMGateway, build_llm_gateway
from app.persistence.knowledge_repository import build_knowledge_store
from app.persistence.message_repository import SqliteMessageRepository
from app.persistence.run_repository import SqliteRunRepository
from app.rag.embeddings import EmbeddingProvider, build_embedding_provider
from app.rag.retriever import KnowledgeRetriever
from app.rag.store import KnowledgeStore
from app.rag.vector_store import VectorStore, build_vector_store
from app.services.approval_service import ApprovalService
from app.services.document_service import DocumentService
from app.services.message_service import MessageService
from app.services.reindex_job_service import ReindexJobService
from app.services.run_service import RunService
from app.services.tool_service import ToolService
from app.tools.builtins.calculator import register_calculator_tool
from app.tools.builtins.ccf_catalog import register_ccf_c_journals_tool
from app.tools.builtins.document_extractor import register_document_items_tool
from app.tools.builtins.knowledge_base import register_knowledge_base_tool
from app.tools.builtins.send_email import register_send_email_tool
from app.tools.executor import ToolExecutor
from app.tools.mcp.adapter import register_mcp_tools
from app.tools.mcp.discovery import load_mcp_catalog
from app.tools.registry import ToolRegistry


@lru_cache
def get_knowledge_store() -> KnowledgeStore:
    return build_knowledge_store(get_settings())


@lru_cache
def get_retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever(
        store=get_knowledge_store(),
        vector_store=get_vector_store(),
        embedding_provider=get_embedding_provider(),
    )


@lru_cache
def get_document_service() -> DocumentService:
    return DocumentService(
        store=get_knowledge_store(),
        vector_store=get_vector_store(),
        embedding_provider=get_embedding_provider(),
        retriever=get_retriever(),
    )


@lru_cache
def get_reindex_job_service() -> ReindexJobService:
    return ReindexJobService(get_settings().run_storage_path)


@lru_cache
def get_vector_store() -> VectorStore:
    return build_vector_store(get_settings())


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return build_embedding_provider(get_settings())


@lru_cache
def get_run_repository() -> SqliteRunRepository:
    return SqliteRunRepository(get_settings().run_storage_path)


@lru_cache
def get_message_repository() -> SqliteMessageRepository:
    return SqliteMessageRepository(get_settings().run_storage_path)


@lru_cache
def get_message_service() -> MessageService:
    return MessageService(get_message_repository())


@lru_cache
def get_tool_service() -> ToolService:
    return ToolService(get_tool_registry())


@lru_cache
def get_run_service() -> RunService:
    return RunService(get_run_repository())


@lru_cache
def get_approval_service() -> ApprovalService:
    return ApprovalService(
        runtime=get_runtime(),
        repository=get_run_repository(),
        message_repository=get_message_repository(),
        history_limit=get_settings().session_history_limit,
    )


@lru_cache
def get_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_calculator_tool(registry)
    register_document_items_tool(registry, get_knowledge_store(), get_settings().extraction_export_path)
    register_ccf_c_journals_tool(registry, get_knowledge_store())
    register_knowledge_base_tool(registry, get_retriever())
    register_send_email_tool(registry)
    register_mcp_tools(registry, load_mcp_catalog(get_settings().mcp_config_path))
    return registry


@lru_cache
def get_tool_executor() -> ToolExecutor:
    return ToolExecutor(get_tool_registry())


@lru_cache
def get_llm_gateway() -> LLMGateway:
    return build_llm_gateway(get_settings())


@lru_cache
def get_runtime() -> AgentRuntime:
    return AgentRuntime(
        settings=get_settings(),
        registry=get_tool_registry(),
        executor=get_tool_executor(),
        llm_gateway=get_llm_gateway(),
    )


def settings_dependency() -> Settings:
    return get_settings()

from collections.abc import Iterator

import pytest

from app.api import dependencies
from app.core.config import get_settings


def _clear_dependency_caches() -> None:
    get_settings.cache_clear()
    dependencies.get_knowledge_store.cache_clear()
    dependencies.get_retriever.cache_clear()
    dependencies.get_document_service.cache_clear()
    dependencies.get_reindex_job_service.cache_clear()
    dependencies.get_vector_store.cache_clear()
    dependencies.get_embedding_provider.cache_clear()
    dependencies.get_run_repository.cache_clear()
    dependencies.get_message_repository.cache_clear()
    dependencies.get_message_service.cache_clear()
    dependencies.get_tool_service.cache_clear()
    dependencies.get_run_service.cache_clear()
    dependencies.get_approval_service.cache_clear()
    dependencies.get_tool_registry.cache_clear()
    dependencies.get_tool_executor.cache_clear()
    dependencies.get_llm_gateway.cache_clear()
    dependencies.get_runtime.cache_clear()


@pytest.fixture(autouse=True)
def isolate_app_state(tmp_path, monkeypatch) -> Iterator[None]:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("KNOWLEDGE_STORE_PROVIDER", "sqlite")
    monkeypatch.setenv("KNOWLEDGE_STORE_PATH", str(data_dir / "knowledge_base.db"))
    monkeypatch.setenv("RUN_STORAGE_PATH", str(data_dir / "multi_tool_agent.db"))
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "memory")
    monkeypatch.setenv("VECTOR_STORE_PATH", str(data_dir / "qdrant"))
    monkeypatch.setenv("MCP_CONFIG_PATH", str(tmp_path / "mcp_servers.json"))
    monkeypatch.setenv("EXTRACTION_EXPORT_PATH", str(data_dir / "exports"))

    _clear_dependency_caches()
    yield
    _clear_dependency_caches()

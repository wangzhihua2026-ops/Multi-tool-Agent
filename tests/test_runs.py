import asyncio
from pathlib import Path

from app.agent.runtime import AgentRuntime
from app.agent.state import ChatRequest
from app.core.config import Settings
from app.core.llm_gateway import MockLLMGateway
from app.persistence.message_repository import SqliteMessageRepository
from app.persistence.run_repository import SqliteRunRepository
from app.rag.embeddings import HashEmbeddingProvider
from app.rag.models import DocumentCreateRequest
from app.rag.retriever import KnowledgeRetriever
from app.rag.store import InMemoryKnowledgeStore
from app.rag.vector_store import InMemoryVectorStore
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.tools.builtins.calculator import register_calculator_tool
from app.tools.builtins.knowledge_base import register_knowledge_base_tool
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


def test_chat_service_persists_run_and_events(tmp_path: Path) -> None:
    store = InMemoryKnowledgeStore()
    vector_store = InMemoryVectorStore()
    embedding_provider = HashEmbeddingProvider()
    retriever = KnowledgeRetriever(store, vector_store, embedding_provider)
    document_service = DocumentService(
        store=store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        retriever=retriever,
    )
    document_service.create_document(
        DocumentCreateRequest(
            title="ops-checklist",
            content="Check OPENAI_API_KEY before restarting the service.",
            metadata={},
        )
    )

    registry = ToolRegistry()
    register_calculator_tool(registry)
    register_knowledge_base_tool(registry, retriever)
    executor = ToolExecutor(registry)
    runtime = AgentRuntime(
        settings=Settings(),
        registry=registry,
        executor=executor,
        llm_gateway=MockLLMGateway(),
    )
    repository = SqliteRunRepository(str(tmp_path / "runs.db"))
    message_repository = SqliteMessageRepository(str(tmp_path / "runs.db"))
    service = ChatService(
        runtime=runtime,
        repository=repository,
        message_repository=message_repository,
    )

    events = asyncio.run(_collect_events(service))
    assert any(event.type == "assistant.message" for event in events)

    stored_runs = repository.list_runs()
    assert len(stored_runs) == 1
    stored_run = repository.get_run(stored_runs[0].run_id)
    assert stored_run.status == "completed"
    assert stored_run.final_response is not None
    assert any(event.event_type == "planner.completed" for event in stored_run.events)
    assert any("OPENAI_API_KEY" in event.data.get("content", "") for event in stored_run.events if event.event_type == "assistant.message")
    stored_messages = message_repository.list_messages("persisted-session")
    assert len(stored_messages) == 2
    assert stored_messages[0].role == "user"
    assert stored_messages[1].role == "assistant"


async def _collect_events(service: ChatService):
    events = []
    request = ChatRequest(
        session_id="persisted-session",
        message="Please search the knowledge base for the first startup check.",
    )
    async for event in service.stream_chat(request):
        events.append(event)
    return events

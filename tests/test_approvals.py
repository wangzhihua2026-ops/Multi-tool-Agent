import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agent.events import AgentEvent
from app.agent.runtime import AgentRuntime
from app.agent.state import ChatRequest
from app.api import dependencies
from app.api.server import app
from app.core.config import Settings
from app.core.exceptions import ApprovalStateError
from app.core.llm_gateway import MockLLMGateway
from app.persistence.message_repository import SqliteMessageRepository
from app.persistence.run_repository import SqliteRunRepository
from app.rag.embeddings import HashEmbeddingProvider
from app.services.approval_service import ApprovalAction, ApprovalService
from app.services.chat_service import ChatService
from app.tools.builtins.calculator import register_calculator_tool
from app.tools.builtins.knowledge_base import register_knowledge_base_tool
from app.tools.builtins.send_email import register_send_email_tool
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.rag.retriever import KnowledgeRetriever
from app.rag.store import InMemoryKnowledgeStore
from app.rag.vector_store import InMemoryVectorStore

client = TestClient(app)


def test_approval_flow_pauses_and_resumes(tmp_path: Path) -> None:
    message_repository = SqliteMessageRepository(str(tmp_path / "approvals.db"))
    repository = SqliteRunRepository(str(tmp_path / "approvals.db"))
    runtime = _build_runtime()
    chat_service = ChatService(
        runtime=runtime,
        repository=repository,
        message_repository=message_repository,
    )
    approval_service = ApprovalService(
        runtime=runtime,
        repository=repository,
        message_repository=message_repository,
    )

    initial_events = asyncio.run(_collect_chat_events(chat_service))
    initial_types = [event.type for event in initial_events]
    assert "approval.required" in initial_types
    assert "run.waiting_approval" in initial_types
    assert "assistant.message" not in initial_types

    run_id = initial_events[0].run_id
    waiting_run = repository.get_run(run_id)
    assert waiting_run.status == "waiting_approval"
    assert waiting_run.pending_tool_name == "send_email"
    assert waiting_run.approval_status == "pending"

    completed_run = asyncio.run(
        approval_service.handle_decision(run_id=run_id, action=ApprovalAction.APPROVE)
    )
    assert completed_run.status == "completed"
    assert completed_run.approval_status == "approved"
    assert completed_run.pending_tool_name is None
    assert completed_run.final_response is not None
    assert "ops@example.com" in completed_run.final_response
    assert any(event.event_type == "approval.approved" for event in completed_run.events)
    assert any(event.event_type == "tool.completed" for event in completed_run.events)
    stored_messages = message_repository.list_messages("approval-session")
    assert len(stored_messages) == 2
    assert stored_messages[-1].role == "assistant"


def test_approval_decision_is_single_use(tmp_path: Path) -> None:
    message_repository = SqliteMessageRepository(str(tmp_path / "single-use-approval.db"))
    repository = SqliteRunRepository(str(tmp_path / "single-use-approval.db"))
    runtime = _build_runtime()
    chat_service = ChatService(
        runtime=runtime,
        repository=repository,
        message_repository=message_repository,
    )
    approval_service = ApprovalService(
        runtime=runtime,
        repository=repository,
        message_repository=message_repository,
    )

    initial_events = asyncio.run(_collect_chat_events(chat_service))
    run_id = initial_events[0].run_id

    completed_run = asyncio.run(
        approval_service.handle_decision(run_id=run_id, action=ApprovalAction.APPROVE)
    )
    assert completed_run.status == "completed"

    with pytest.raises(ApprovalStateError, match="not waiting for approval"):
        asyncio.run(approval_service.handle_decision(run_id=run_id, action=ApprovalAction.APPROVE))

    refreshed_run = repository.get_run(run_id)
    tool_completed_events = [
        event for event in refreshed_run.events
        if event.event_type == "tool.completed"
    ]
    assert len(tool_completed_events) == 1


def test_rejected_approval_completes_without_tool_execution(tmp_path: Path) -> None:
    message_repository = SqliteMessageRepository(str(tmp_path / "rejections.db"))
    repository = SqliteRunRepository(str(tmp_path / "rejections.db"))
    runtime = _build_runtime()
    chat_service = ChatService(
        runtime=runtime,
        repository=repository,
        message_repository=message_repository,
    )
    approval_service = ApprovalService(
        runtime=runtime,
        repository=repository,
        message_repository=message_repository,
    )

    initial_events = asyncio.run(_collect_chat_events(chat_service))
    run_id = initial_events[0].run_id
    completed_run = asyncio.run(
        approval_service.handle_decision(run_id=run_id, action=ApprovalAction.REJECT)
    )
    assert completed_run.status == "completed"
    assert completed_run.approval_status == "rejected"
    assert completed_run.final_response is not None
    assert "rejected" in completed_run.final_response.lower()
    assert not any(event.event_type == "tool.completed" for event in completed_run.events)
    stored_messages = message_repository.list_messages("approval-session")
    assert len(stored_messages) == 2


def test_pending_approvals_endpoint_is_not_limited_to_recent_runs() -> None:
    repository = dependencies.get_run_repository()
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    repository.create_run(
        run_id="waiting-run",
        session_id="approval-session",
        user_message="Please send an email.",
        created_at=created_at.isoformat(),
    )
    repository.append_event(
        AgentEvent(
            type="approval.required",
            run_id="waiting-run",
            created_at=created_at,
            data={
                "tool_name": "send_email",
                "arguments": {"to": "ops@example.com"},
            },
        ),
        sequence=0,
    )
    repository.append_event(
        AgentEvent(
            type="run.waiting_approval",
            run_id="waiting-run",
            created_at=created_at + timedelta(seconds=1),
            data={"status": "waiting_approval"},
        ),
        sequence=1,
    )

    for index in range(60):
        run_id = f"completed-{index}"
        completed_at = created_at + timedelta(minutes=index + 1)
        repository.create_run(
            run_id=run_id,
            session_id=f"session-{index}",
            user_message=f"Completed run {index}",
            created_at=completed_at.isoformat(),
        )
        repository.append_event(
            AgentEvent(
                type="run.completed",
                run_id=run_id,
                created_at=completed_at,
                data={"status": "completed"},
            ),
            sequence=0,
        )

    recent_run_ids = {run.run_id for run in repository.list_runs(limit=50)}
    assert "waiting-run" not in recent_run_ids

    response = client.get("/api/runs/pending-approvals")
    assert response.status_code == 200
    payload = response.json()
    assert [run["run_id"] for run in payload] == ["waiting-run"]


def _build_runtime() -> AgentRuntime:
    store = InMemoryKnowledgeStore()
    vector_store = InMemoryVectorStore()
    embedding_provider = HashEmbeddingProvider()
    retriever = KnowledgeRetriever(store, vector_store, embedding_provider)
    registry = ToolRegistry()
    register_calculator_tool(registry)
    register_knowledge_base_tool(registry, retriever)
    register_send_email_tool(registry)
    executor = ToolExecutor(registry)
    return AgentRuntime(
        settings=Settings(),
        registry=registry,
        executor=executor,
        llm_gateway=MockLLMGateway(),
    )


async def _collect_chat_events(service: ChatService) -> list:
    events = []
    request = ChatRequest(
        session_id="approval-session",
        message="Please send email to ops@example.com subject: Restart body: Restart the service at 5 PM.",
    )
    async for event in service.stream_chat(request):
        events.append(event)
    return events

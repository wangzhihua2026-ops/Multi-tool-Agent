import asyncio
from pathlib import Path

from app.agent.runtime import AgentRuntime
from app.agent.state import ChatRequest, ConversationMessage
from app.core.config import Settings
from app.core.llm_gateway import LLMGateway, PlanAction, PlanResult
from app.persistence.message_repository import SqliteMessageRepository
from app.persistence.run_repository import SqliteRunRepository
from app.services.chat_service import ChatService
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


class RecordingGateway(LLMGateway):
    def __init__(self) -> None:
        self.history_lengths: list[int] = []
        self.history_snapshots: list[list[ConversationMessage]] = []

    async def plan(self, user_message: str, tools, history=None, tool_results=None, tool_errors=None) -> PlanResult:
        snapshot = list(history or [])
        self.history_snapshots.append(snapshot)
        self.history_lengths.append(len(snapshot))
        return PlanResult(
            action=PlanAction.RESPOND,
            provider="recording",
            answer=f"history={len(snapshot)}",
        )

    async def answer(self, user_message: str, tool_result=None, tool_error=None, history=None) -> str:
        return "unused"


def test_chat_service_persists_messages_and_loads_history(tmp_path: Path) -> None:
    db_path = str(tmp_path / "messages.db")
    gateway = RecordingGateway()
    runtime = AgentRuntime(
        settings=Settings(),
        registry=ToolRegistry(),
        executor=ToolExecutor(ToolRegistry()),
        llm_gateway=gateway,
    )
    run_repository = SqliteRunRepository(db_path)
    message_repository = SqliteMessageRepository(db_path)
    service = ChatService(
        runtime=runtime,
        repository=run_repository,
        message_repository=message_repository,
        history_limit=10,
    )

    asyncio.run(_drain(service, session_id="session-42", message="First turn"))
    asyncio.run(_drain(service, session_id="session-42", message="Second turn"))

    assert gateway.history_lengths == [0, 2]
    assert gateway.history_snapshots[1][0].role == "user"
    assert gateway.history_snapshots[1][1].role == "assistant"

    stored_messages = message_repository.list_messages("session-42", limit=10)
    assert len(stored_messages) == 4
    assert [message.role for message in stored_messages] == ["user", "assistant", "user", "assistant"]


async def _drain(service: ChatService, session_id: str, message: str) -> None:
    request = ChatRequest(session_id=session_id, message=message)
    async for _ in service.stream_chat(request):
        pass

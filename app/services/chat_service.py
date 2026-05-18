from collections.abc import AsyncIterator
import logging

from app.agent.events import AgentEvent
from app.agent.runtime import AgentRuntime
from app.agent.state import ChatRequest, ConversationMessage
from app.persistence.message_repository import SqliteMessageRepository
from app.persistence.run_repository import SqliteRunRepository
from app.core.logger import reset_run_id, set_run_id

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        runtime: AgentRuntime,
        repository: SqliteRunRepository,
        message_repository: SqliteMessageRepository,
        history_limit: int = 12,
    ) -> None:
        self.runtime = runtime
        self.repository = repository
        self.message_repository = message_repository
        self.history_limit = history_limit

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[AgentEvent]:
        sequence = 0
        run_created = False
        current_run_id: str | None = None
        history = self._load_history(request.session_id)

        try:
            async for event in self.runtime.stream(request, history=history):
                if not run_created:
                    run_token = set_run_id(event.run_id)
                    self.repository.create_run(
                        run_id=event.run_id,
                        session_id=request.session_id,
                        user_message=request.message,
                        created_at=event.created_at.isoformat(),
                    )
                    self.message_repository.add_message(
                        session_id=request.session_id,
                        run_id=event.run_id,
                        role="user",
                        content=request.message,
                        created_at=event.created_at.isoformat(),
                        metadata={"source": "chat_request"},
                    )
                    run_created = True
                    current_run_id = event.run_id
                    logger.info(
                        "chat run created session_id=%s run_id=%s",
                        request.session_id,
                        event.run_id,
                    )

                self.repository.append_event(event, sequence=sequence)
                if event.type == "assistant.message":
                    self.message_repository.add_message(
                        session_id=request.session_id,
                        run_id=event.run_id,
                        role="assistant",
                        content=str(event.data.get("content", "")),
                        created_at=event.created_at.isoformat(),
                        metadata={"source": "assistant_message"},
                    )
                sequence += 1
                yield event
        except Exception as exc:
            if run_created and current_run_id is not None:
                failure_event = AgentEvent(
                    type="run.failed",
                    run_id=current_run_id,
                    data={"error": str(exc)},
                )
                self.repository.append_event(failure_event, sequence=sequence)
                yield failure_event
                return
            raise
        finally:
            if run_created:
                reset_run_id(run_token)

    def _load_history(self, session_id: str) -> list[ConversationMessage]:
        records = self.message_repository.list_messages(
            session_id=session_id,
            limit=self.history_limit,
        )
        return [
            ConversationMessage(role=record.role, content=record.content)
            for record in records
        ]

from enum import StrEnum
import logging

from app.agent.events import AgentEvent
from app.agent.runtime import AgentRuntime
from app.agent.state import ConversationMessage
from app.agent.state import RunStatus
from app.core.exceptions import ApprovalStateError
from app.core.logger import reset_run_id, set_run_id
from app.persistence.message_repository import SqliteMessageRepository
from app.persistence.models import RunDetail
from app.persistence.run_repository import SqliteRunRepository

logger = logging.getLogger(__name__)


class ApprovalAction(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


class ApprovalService:
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

    async def handle_decision(self, run_id: str, action: ApprovalAction) -> RunDetail:
        run_token = set_run_id(run_id)
        run = self.repository.get_run(run_id)
        try:
            if run.status != RunStatus.WAITING_APPROVAL.value:
                raise ApprovalStateError(f"Run '{run_id}' is not waiting for approval.")
            if not run.pending_tool_name:
                raise ApprovalStateError(f"Run '{run_id}' does not have a pending tool call.")

            decision_event = AgentEvent(
                type="approval.approved" if action == ApprovalAction.APPROVE else "approval.rejected",
                run_id=run_id,
                data={
                    "action": action,
                    "tool_name": run.pending_tool_name,
                },
            )
            claimed = self.repository.claim_pending_approval(
                run_id=run_id,
                approval_status="approved" if action == ApprovalAction.APPROVE else "rejected",
                updated_at=decision_event.created_at.isoformat(),
            )
            if not claimed:
                raise ApprovalStateError(f"Run '{run_id}' approval was already handled.")
            logger.info(
                "approval decision claimed run_id=%s action=%s tool_name=%s",
                run_id,
                action,
                run.pending_tool_name,
            )

            sequence = self.repository.get_next_sequence(run_id)
            history = self._load_history(run.session_id, current_run_id=run_id)
            self.repository.append_event(decision_event, sequence=sequence)
            sequence += 1

            async for event in self.runtime.resume(
                run_id=run_id,
                user_message=run.user_message,
                tool_name=run.pending_tool_name,
                arguments=run.pending_tool_arguments,
                approved=action == ApprovalAction.APPROVE,
                history=history,
            ):
                self.repository.append_event(event, sequence=sequence)
                if event.type == "assistant.message":
                    self.message_repository.add_message(
                        session_id=run.session_id,
                        run_id=run_id,
                        role="assistant",
                        content=str(event.data.get("content", "")),
                        created_at=event.created_at.isoformat(),
                        metadata={"source": "assistant_message"},
                    )
                sequence += 1

            return self.repository.get_run(run_id)
        finally:
            reset_run_id(run_token)

    def _load_history(self, session_id: str, current_run_id: str) -> list[ConversationMessage]:
        records = self.message_repository.list_messages(
            session_id=session_id,
            limit=self.history_limit,
        )
        if records and records[-1].run_id == current_run_id and records[-1].role == "user":
            records = records[:-1]
        return [
            ConversationMessage(role=record.role, content=record.content)
            for record in records
        ]

from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    RETRY_SCHEDULED = "retry_scheduled"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ConversationMessage(BaseModel):
    role: str
    content: str


class RunState(BaseModel):
    run_id: str
    session_id: str
    user_message: str
    status: RunStatus = Field(default=RunStatus.RUNNING)

    @classmethod
    def from_request(cls, request: ChatRequest) -> "RunState":
        return cls(
            run_id=str(uuid4()),
            session_id=request.session_id,
            user_message=request.message,
        )

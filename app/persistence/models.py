from datetime import datetime

from pydantic import BaseModel, Field


class RunSummary(BaseModel):
    run_id: str
    session_id: str
    user_message: str
    status: str
    created_at: datetime
    updated_at: datetime
    provider: str | None = None
    model: str | None = None
    approval_status: str | None = None
    pending_tool_name: str | None = None


class RunEventRecord(BaseModel):
    sequence: int
    event_type: str
    created_at: datetime
    data: dict = Field(default_factory=dict)


class SessionMessageRecord(BaseModel):
    message_id: str
    session_id: str
    run_id: str | None = None
    role: str
    content: str
    created_at: datetime
    metadata: dict = Field(default_factory=dict)


class RunDetail(RunSummary):
    final_response: str | None = None
    pending_tool_arguments: dict = Field(default_factory=dict)
    events: list[RunEventRecord] = Field(default_factory=list)

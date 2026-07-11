from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.agent.state import ConversationMessage
from app.tools.schemas import ToolExecutionResult


class StepType(StrEnum):
    PLAN = "plan"
    TOOL_CALL = "tool_call"
    APPROVAL = "approval"
    ANSWER = "answer"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ErrorClass(StrEnum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    NEEDS_ATTENTION = "needs_attention"


class ExecutionCheckpoint(BaseModel):
    run_id: str
    session_id: str
    user_message: str
    history: list[ConversationMessage] = Field(default_factory=list)
    next_step: int = 1
    tool_results: list[ToolExecutionResult] = Field(default_factory=list)
    tool_errors: list[str] = Field(default_factory=list)
    pending_tool_name: str | None = None
    pending_tool_arguments: dict[str, Any] = Field(default_factory=dict)
    approval_decision: bool | None = None


class StepOutcome(BaseModel):
    step_type: StepType
    checkpoint: ExecutionCheckpoint
    events: list[dict[str, Any]] = Field(default_factory=list)
    final_response: str | None = None
    waiting_for_approval: bool = False
    completed: bool = False

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolExecutionSemantics(StrEnum):
    READ_ONLY = "read_only"
    IDEMPOTENT_SIDE_EFFECT = "idempotent_side_effect"
    NON_IDEMPOTENT_SIDE_EFFECT = "non_idempotent_side_effect"


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    risk_level: str = "low"
    approval_required: bool = False
    execution_semantics: ToolExecutionSemantics = ToolExecutionSemantics.READ_ONLY
    timeout_seconds: int = 15
    source: str = "local"
    server_name: str | None = None
    execution_mode: str | None = None


class ToolExecutionResult(BaseModel):
    tool_name: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

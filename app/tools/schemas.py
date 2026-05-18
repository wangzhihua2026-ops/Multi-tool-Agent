from typing import Any

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    risk_level: str = "low"
    approval_required: bool = False
    timeout_seconds: int = 15
    source: str = "local"
    server_name: str | None = None
    execution_mode: str | None = None


class ToolExecutionResult(BaseModel):
    tool_name: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

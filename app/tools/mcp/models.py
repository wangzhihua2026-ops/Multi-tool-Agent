from typing import Any

from pydantic import BaseModel, Field


class MCPToolConfig(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    mock_result_template: str | None = None
    risk_level: str = "low"
    approval_required: bool = False
    timeout_seconds: int = 15
    enabled: bool = True


class MCPServerConfig(BaseModel):
    server_name: str
    enabled: bool = True
    transport: str = "mock"
    endpoint: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    initialize: bool = False
    discover_tools: bool = False
    protocol_version: str = "2024-11-05"
    client_name: str = "multi-tool-agent"
    client_version: str = "0.1.0"
    tools: list[MCPToolConfig] = Field(default_factory=list)


class MCPCatalog(BaseModel):
    servers: list[MCPServerConfig] = Field(default_factory=list)

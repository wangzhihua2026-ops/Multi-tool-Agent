import json
from string import Formatter
from uuid import uuid4

import httpx

from app.tools.mcp.models import MCPCatalog, MCPServerConfig, MCPToolConfig
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition, ToolExecutionResult


def register_mcp_tools(registry: ToolRegistry, catalog: MCPCatalog) -> None:
    for server in catalog.servers:
        if not server.enabled:
            continue
        for tool in _tools_for_server(server):
            if not tool.enabled:
                continue
            registry.register(
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    risk_level=tool.risk_level,
                    approval_required=tool.approval_required,
                    timeout_seconds=tool.timeout_seconds,
                    source="mcp",
                    server_name=server.server_name,
                    execution_mode=server.transport,
                ),
                _build_mcp_handler(server, tool),
            )


def _tools_for_server(server: MCPServerConfig) -> list[MCPToolConfig]:
    tools = list(server.tools)
    if server.discover_tools:
        discovered_tools = _discover_http_tools(server)
        existing_names = {tool.name for tool in tools}
        tools.extend(tool for tool in discovered_tools if tool.name not in existing_names)
    return tools


def _build_mcp_handler(server: MCPServerConfig, tool: MCPToolConfig):
    transport = server.transport.strip().lower()
    if transport in {"http", "streamable_http"}:
        return _build_http_mcp_handler(server, tool)
    if transport not in {"mock", ""}:
        return _build_unsupported_mcp_handler(server, tool)
    return _build_mock_mcp_handler(server, tool)


def _build_mock_mcp_handler(server: MCPServerConfig, tool: MCPToolConfig):
    def handler(arguments: dict) -> ToolExecutionResult:
        template = tool.mock_result_template or _default_template(server.server_name, tool.name)
        content = _render_template(template, arguments)
        return ToolExecutionResult(
            tool_name=tool.name,
            content=content,
            metadata={
                "source": "mcp",
                "server_name": server.server_name,
                "transport": server.transport,
                "endpoint": server.endpoint,
                "arguments": arguments,
                "execution_mode": "mock_adapter",
            },
        )

    return handler


def _build_http_mcp_handler(server: MCPServerConfig, tool: MCPToolConfig):
    def handler(arguments: dict) -> ToolExecutionResult:
        if not server.endpoint:
            raise ValueError(f"MCP server '{server.server_name}' is missing an HTTP endpoint.")

        request_id = str(uuid4())
        session_id = _initialize_http_session(server, timeout_seconds=tool.timeout_seconds) if server.initialize else None
        data = _post_json_rpc(
            server=server,
            method="tools/call",
            params={
                "name": tool.name,
                "arguments": arguments,
            },
            timeout_seconds=tool.timeout_seconds,
            request_id=request_id,
            session_id=session_id,
        )

        if data.get("error"):
            error = data["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise ValueError(f"MCP tool '{tool.name}' failed: {message}")

        result = data.get("result", data)
        if isinstance(result, dict) and result.get("isError"):
            raise ValueError(f"MCP tool '{tool.name}' returned an error result: {_mcp_result_to_content(result)}")

        return ToolExecutionResult(
            tool_name=tool.name,
            content=_mcp_result_to_content(result),
            metadata={
                "source": "mcp",
                "server_name": server.server_name,
                "transport": server.transport,
                "endpoint": server.endpoint,
                "arguments": arguments,
                "execution_mode": "live_http",
                "request_id": request_id,
            },
        )

    return handler


def _build_unsupported_mcp_handler(server: MCPServerConfig, tool: MCPToolConfig):
    def handler(arguments: dict) -> ToolExecutionResult:
        raise ValueError(
            f"MCP transport '{server.transport}' is not supported for tool '{tool.name}'. "
            "Use transport 'mock', 'http', or 'streamable_http'."
        )

    return handler


def _default_template(server_name: str, tool_name: str) -> str:
    return (
        f"MCP tool '{tool_name}' was discovered from server '{server_name}'. "
        "Live server execution is not wired yet, so this is the adapter fallback result."
    )


def _render_template(template: str, arguments: dict) -> str:
    formatter = Formatter()
    field_names = {
        field_name
        for _, field_name, _, _ in formatter.parse(template)
        if field_name
    }
    safe_values = {name: json.dumps(arguments.get(name), ensure_ascii=False) if isinstance(arguments.get(name), (dict, list)) else arguments.get(name, "") for name in field_names}
    try:
        return template.format_map(_SafeDict(safe_values))
    except Exception:
        return f"{template}\n\nArguments: {json.dumps(arguments, ensure_ascii=False, default=str)}"


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _mcp_result_to_content(result) -> str:
    if isinstance(result, dict):
        content_items = result.get("content")
        if isinstance(content_items, list):
            rendered_items = [_render_mcp_content_item(item) for item in content_items]
            rendered = "\n".join(item for item in rendered_items if item)
            if rendered:
                return rendered
        if "structuredContent" in result:
            return json.dumps(result["structuredContent"], ensure_ascii=False, default=str)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)


def _render_mcp_content_item(item) -> str:
    if isinstance(item, dict):
        if item.get("type") == "text":
            return str(item.get("text", ""))
        return json.dumps(item, ensure_ascii=False, default=str)
    return str(item)


def _discover_http_tools(server: MCPServerConfig) -> list[MCPToolConfig]:
    transport = server.transport.strip().lower()
    if transport not in {"http", "streamable_http"}:
        raise ValueError(f"MCP tool discovery requires an HTTP transport for server '{server.server_name}'.")
    if not server.endpoint:
        raise ValueError(f"MCP server '{server.server_name}' is missing an HTTP endpoint.")

    session_id = _initialize_http_session(server, timeout_seconds=15) if server.initialize else None
    data = _post_json_rpc(
        server=server,
        method="tools/list",
        params={},
        timeout_seconds=15,
        session_id=session_id,
    )
    result = data.get("result", data)
    raw_tools = result.get("tools", []) if isinstance(result, dict) else []
    if not isinstance(raw_tools, list):
        raise ValueError(f"MCP server '{server.server_name}' returned an invalid tools/list response.")

    discovered_tools: list[MCPToolConfig] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict) or not raw_tool.get("name"):
            continue
        discovered_tools.append(
            MCPToolConfig(
                name=str(raw_tool["name"]),
                description=str(raw_tool.get("description", "")),
                input_schema=_normalize_input_schema(raw_tool.get("inputSchema") or raw_tool.get("input_schema")),
            )
        )
    return discovered_tools


def _initialize_http_session(server: MCPServerConfig, timeout_seconds: int) -> str | None:
    data, headers = _post_json_rpc_with_headers(
        server=server,
        method="initialize",
        params={
            "protocolVersion": server.protocol_version,
            "capabilities": {},
            "clientInfo": {
                "name": server.client_name,
                "version": server.client_version,
            },
        },
        timeout_seconds=timeout_seconds,
    )
    if data.get("error"):
        error = data["error"]
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise ValueError(f"MCP server '{server.server_name}' initialization failed: {message}")
    session_id = headers.get("mcp-session-id") or headers.get("Mcp-Session-Id")
    return str(session_id) if session_id else None


def _post_json_rpc(
    server: MCPServerConfig,
    method: str,
    params: dict,
    timeout_seconds: int,
    request_id: str | None = None,
    session_id: str | None = None,
) -> dict:
    data, _ = _post_json_rpc_with_headers(
        server=server,
        method=method,
        params=params,
        timeout_seconds=timeout_seconds,
        request_id=request_id,
        session_id=session_id,
    )
    if data.get("error"):
        error = data["error"]
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise ValueError(f"MCP method '{method}' failed on server '{server.server_name}': {message}")
    return data


def _post_json_rpc_with_headers(
    server: MCPServerConfig,
    method: str,
    params: dict,
    timeout_seconds: int,
    request_id: str | None = None,
    session_id: str | None = None,
) -> tuple[dict, dict]:
    if not server.endpoint:
        raise ValueError(f"MCP server '{server.server_name}' is missing an HTTP endpoint.")

    payload = {
        "jsonrpc": "2.0",
        "id": request_id or str(uuid4()),
        "method": method,
        "params": params,
    }
    headers = _build_http_headers(server, session_id=session_id)
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(server.endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json(), dict(getattr(response, "headers", {}) or {})


def _build_http_headers(server: MCPServerConfig, session_id: str | None = None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    headers.update(server.headers)
    if session_id:
        headers["mcp-session-id"] = session_id
    return headers


def _normalize_input_schema(raw_schema) -> dict:
    if isinstance(raw_schema, dict) and raw_schema:
        return raw_schema
    return {"type": "object", "properties": {}}

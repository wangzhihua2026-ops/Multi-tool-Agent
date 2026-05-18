import asyncio
from pathlib import Path

from app.tools.executor import ToolExecutor
from app.tools.mcp.adapter import register_mcp_tools
from app.tools.mcp.discovery import load_mcp_catalog
from app.tools.registry import ToolRegistry


def test_load_mcp_catalog_and_register_tools(tmp_path: Path) -> None:
    config_path = tmp_path / "mcp_servers.json"
    config_path.write_text(
        """
        {
          "servers": [
            {
              "server_name": "ops-runbooks",
              "enabled": true,
              "transport": "mock",
              "tools": [
                {
                  "name": "lookup_runbook",
                  "description": "Look up an operations runbook.",
                  "input_schema": {
                    "type": "object",
                    "properties": {
                      "service": {"type": "string"}
                    },
                    "required": ["service"]
                  },
                  "mock_result_template": "Runbook for {service}: inspect logs first."
                }
              ]
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    catalog = load_mcp_catalog(str(config_path))
    registry = ToolRegistry()
    register_mcp_tools(registry, catalog)
    executor = ToolExecutor(registry)

    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "lookup_runbook"
    assert tools[0].source == "mcp"
    assert tools[0].server_name == "ops-runbooks"
    assert tools[0].execution_mode == "mock"

    result = asyncio.run(executor.execute("lookup_runbook", {"service": "payments"}))
    assert "payments" in result.content
    assert result.metadata["source"] == "mcp"


def test_mcp_mock_handler_renders_result(tmp_path: Path) -> None:
    config_path = tmp_path / "mcp_servers.json"
    config_path.write_text(
        """
        {
          "servers": [
            {
              "server_name": "ops-runbooks",
              "enabled": true,
              "transport": "mock",
              "tools": [
                {
                  "name": "lookup_runbook",
                  "description": "Look up an operations runbook.",
                  "input_schema": {
                    "type": "object",
                    "properties": {
                      "service": {"type": "string"}
                    },
                    "required": ["service"]
                  },
                  "mock_result_template": "Runbook for {service}: inspect logs first."
                }
              ]
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    catalog = load_mcp_catalog(str(config_path))
    registry = ToolRegistry()
    register_mcp_tools(registry, catalog)
    handler = registry.get_handler("lookup_runbook")
    result = handler({"service": "payments"})
    assert "payments" in result.content
    assert result.metadata["server_name"] == "ops-runbooks"


def test_mcp_http_handler_calls_live_endpoint(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "jsonrpc": "2.0",
                "id": captured["json"]["id"],
                "result": {
                    "content": [
                        {"type": "text", "text": "Live runbook for payments."}
                    ]
                },
            }

    class FakeClient:
        def __init__(self, timeout: int) -> None:
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("app.tools.mcp.adapter.httpx.Client", FakeClient)
    config_path = tmp_path / "mcp_servers.json"
    config_path.write_text(
        """
        {
          "servers": [
            {
              "server_name": "ops-runbooks",
              "enabled": true,
              "transport": "http",
              "endpoint": "http://mcp.example.test",
              "tools": [
                {
                  "name": "lookup_runbook",
                  "description": "Look up an operations runbook.",
                  "input_schema": {
                    "type": "object",
                    "properties": {
                      "service": {"type": "string"}
                    },
                    "required": ["service"]
                  },
                  "timeout_seconds": 7
                }
              ]
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    catalog = load_mcp_catalog(str(config_path))
    registry = ToolRegistry()
    register_mcp_tools(registry, catalog)
    executor = ToolExecutor(registry)

    result = asyncio.run(executor.execute("lookup_runbook", {"service": "payments"}))

    assert result.content == "Live runbook for payments."
    assert result.metadata["execution_mode"] == "live_http"
    assert captured["timeout"] == 7
    assert captured["url"] == "http://mcp.example.test"
    assert captured["json"]["method"] == "tools/call"
    assert captured["json"]["params"] == {
        "name": "lookup_runbook",
        "arguments": {"service": "payments"},
    }


def test_mcp_http_discovery_initializes_and_registers_tools(monkeypatch, tmp_path: Path) -> None:
    requests = []

    class FakeResponse:
        def __init__(self, method: str) -> None:
            self.method = method
            self.headers = {"mcp-session-id": "session-123"} if method == "initialize" else {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            if self.method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": "init",
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "fake-mcp", "version": "1.0.0"},
                    },
                }
            if self.method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": "list",
                    "result": {
                        "tools": [
                            {
                                "name": "lookup_runbook",
                                "description": "Look up an operations runbook.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"service": {"type": "string"}},
                                    "required": ["service"],
                                },
                            }
                        ]
                    },
                }
            return {
                "jsonrpc": "2.0",
                "id": "call",
                "result": {"content": [{"type": "text", "text": "Discovered runbook."}]},
            }

    class FakeClient:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            requests.append({"url": url, "headers": headers, "json": json})
            return FakeResponse(json["method"])

    monkeypatch.setattr("app.tools.mcp.adapter.httpx.Client", FakeClient)
    config_path = tmp_path / "mcp_servers.json"
    config_path.write_text(
        """
        {
          "servers": [
            {
              "server_name": "ops-runbooks",
              "enabled": true,
              "transport": "http",
              "endpoint": "http://mcp.example.test",
              "initialize": true,
              "discover_tools": true,
              "headers": {
                "Authorization": "Bearer test-token"
              }
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    catalog = load_mcp_catalog(str(config_path))
    registry = ToolRegistry()
    register_mcp_tools(registry, catalog)
    executor = ToolExecutor(registry)

    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "lookup_runbook"
    assert tools[0].execution_mode == "http"

    result = asyncio.run(executor.execute("lookup_runbook", {"service": "payments"}))

    assert result.content == "Discovered runbook."
    assert [request["json"]["method"] for request in requests] == [
        "initialize",
        "tools/list",
        "initialize",
        "tools/call",
    ]
    assert requests[0]["headers"]["Authorization"] == "Bearer test-token"
    assert requests[1]["headers"]["mcp-session-id"] == "session-123"
    assert requests[3]["headers"]["mcp-session-id"] == "session-123"

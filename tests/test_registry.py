from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition, ToolExecutionResult


def test_registry_rejects_duplicate_tool_names() -> None:
    registry = ToolRegistry()
    definition = ToolDefinition(
        name="calculator",
        description="Evaluate a math expression.",
        input_schema={"type": "object"},
    )

    registry.register(
        definition,
        lambda arguments: ToolExecutionResult(tool_name="calculator", content=str(arguments)),
    )

    try:
        registry.register(
            definition.model_copy(update={"source": "mcp", "server_name": "demo-server"}),
            lambda arguments: ToolExecutionResult(tool_name="calculator", content="shadowed"),
        )
    except ValueError as exc:
        assert "already registered" in str(exc)
        assert "demo-server" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected duplicate tool registration to raise ValueError.")

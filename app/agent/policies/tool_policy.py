from app.tools.schemas import ToolDefinition


def is_high_risk(tool_definition: ToolDefinition) -> bool:
    return tool_definition.risk_level in {"high", "critical"}

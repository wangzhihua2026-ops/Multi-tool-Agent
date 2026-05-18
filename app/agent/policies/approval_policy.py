from app.tools.schemas import ToolDefinition


def requires_human_approval(tool_definition: ToolDefinition) -> bool:
    return tool_definition.approval_required

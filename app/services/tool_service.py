from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition


class ToolService:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def list_tools(self) -> list[ToolDefinition]:
        return self.registry.list_tools()

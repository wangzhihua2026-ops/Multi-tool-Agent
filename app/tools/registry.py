from collections.abc import Callable
from typing import Any

from app.tools.schemas import ToolDefinition, ToolExecutionResult

ToolHandler = Callable[[dict[str, Any]], ToolExecutionResult]


class ToolRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        existing = self._definitions.get(definition.name)
        if existing is not None:
            existing_source = existing.server_name or existing.source
            new_source = definition.server_name or definition.source
            raise ValueError(
                f"Tool '{definition.name}' is already registered from '{existing_source}' "
                f"and cannot be replaced by '{new_source}'."
            )
        self._definitions[definition.name] = definition
        self._handlers[definition.name] = handler

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._definitions.values())

    def get_definition(self, name: str) -> ToolDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def get_handler(self, name: str) -> ToolHandler:
        try:
            return self._handlers[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

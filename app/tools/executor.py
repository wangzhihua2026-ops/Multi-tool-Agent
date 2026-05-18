import asyncio
import inspect
from typing import Any

from app.core.exceptions import ToolExecutionError
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition, ToolExecutionResult


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        try:
            definition = self.registry.get_definition(name)
            handler = self.registry.get_handler(name)
            _validate_arguments(definition, arguments)
            if inspect.iscoroutinefunction(handler):
                result = handler(arguments)
            else:
                result = await asyncio.to_thread(handler, arguments)
            if inspect.isawaitable(result):
                result = await result
            return result
        except KeyError as exc:
            detail = exc.args[0] if exc.args else f"Unknown tool: {name}"
            raise ToolExecutionError(tool_name=name, detail=detail) from exc
        except ToolExecutionError:
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise ToolExecutionError(tool_name=name, detail=str(exc)) from exc


def _validate_arguments(definition: ToolDefinition, arguments: dict[str, Any]) -> None:
    schema = definition.input_schema or {}
    if schema.get("type") != "object":
        return

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object.")

    properties = schema.get("properties") or {}
    required_fields = schema.get("required") or []
    missing = [
        field_name
        for field_name in required_fields
        if field_name not in arguments or arguments[field_name] is None or arguments[field_name] == ""
    ]
    if missing:
        raise ValueError(f"Missing required tool argument(s): {', '.join(missing)}")

    for field_name, value in arguments.items():
        field_schema = properties.get(field_name)
        if isinstance(field_schema, dict):
            _validate_value(field_name, value, field_schema)


def _validate_value(field_name: str, value: Any, schema: dict[str, Any]) -> None:
    expected_type = schema.get("type")
    if expected_type is None:
        return

    expected_types = expected_type if isinstance(expected_type, list) else [expected_type]
    if any(_matches_json_type(value, item) for item in expected_types):
        return

    expected_label = " or ".join(str(item) for item in expected_types)
    raise ValueError(f"Tool argument '{field_name}' must be {expected_label}.")


def _matches_json_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None
    return True

from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry import trace


SENSITIVE_KEY_PARTS = ("api_key", "authorization", "token", "password", "secret")
tracer = trace.get_tracer("app.agent.platform")


def redact_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _redact_value(key, item) for key, item in value.items()}


def _redact_value(key: str, value: Any) -> Any:
    if any(part in key.lower() for part in SENSITIVE_KEY_PARTS):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact_value("", item) for item in value]
    return value


def step_span_name(step_type: str) -> str:
    return {
        "plan": "agent.plan",
        "tool_call": "agent.tool",
        "approval": "agent.approval",
        "answer": "agent.answer",
    }.get(step_type, "agent.step")

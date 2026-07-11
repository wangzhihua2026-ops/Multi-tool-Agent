import re

from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition, ToolExecutionResult, ToolExecutionSemantics

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def send_email_tool(arguments: dict) -> ToolExecutionResult:
    to = str(arguments.get("to", "")).strip()
    subject = str(arguments.get("subject", "")).strip()
    body = str(arguments.get("body", "")).strip()

    if not to or not EMAIL_PATTERN.fullmatch(to):
        raise ValueError("A valid recipient email address is required.")
    if not subject:
        raise ValueError("An email subject is required.")
    if not body:
        raise ValueError("An email body is required.")

    return ToolExecutionResult(
        tool_name="send_email",
        content=(
            f"Mock email queued to {to} with subject '{subject}'. "
            f"Body preview: {body}"
        ),
        metadata={
            "to": to,
            "subject": subject,
            "delivery_mode": "mock",
        },
    )


def register_send_email_tool(registry: ToolRegistry) -> None:
    registry.register(
        ToolDefinition(
            name="send_email",
            description="Send an email to a recipient. This is a high-risk action that requires human approval.",
            input_schema={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
            risk_level="high",
            approval_required=True,
            execution_semantics=ToolExecutionSemantics.NON_IDEMPOTENT_SIDE_EFFECT,
        ),
        send_email_tool,
    )

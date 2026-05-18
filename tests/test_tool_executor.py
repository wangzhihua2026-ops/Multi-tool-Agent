import asyncio
import threading

import pytest

from app.core.exceptions import ToolExecutionError
from app.tools.builtins.send_email import register_send_email_tool
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition, ToolExecutionResult


def test_executor_validates_required_tool_arguments() -> None:
    registry = ToolRegistry()
    register_send_email_tool(registry)
    executor = ToolExecutor(registry)

    with pytest.raises(ToolExecutionError, match="Missing required tool argument"):
        asyncio.run(
            executor.execute(
                "send_email",
                {"to": "ops@example.com", "subject": "Restart"},
            )
        )


def test_executor_validates_tool_argument_types() -> None:
    registry = ToolRegistry()
    register_send_email_tool(registry)
    executor = ToolExecutor(registry)

    with pytest.raises(ToolExecutionError, match="must be string"):
        asyncio.run(
            executor.execute(
                "send_email",
                {"to": "ops@example.com", "subject": "Restart", "body": ["not", "text"]},
            )
        )


def test_executor_runs_sync_handlers_in_worker_thread() -> None:
    main_thread_id = threading.get_ident()
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="thread_check",
            description="Return the executing thread id.",
            input_schema={"type": "object", "properties": {}},
        ),
        lambda arguments: ToolExecutionResult(
            tool_name="thread_check",
            content="ok",
            metadata={"thread_id": threading.get_ident()},
        ),
    )
    executor = ToolExecutor(registry)

    result = asyncio.run(executor.execute("thread_check", {}))

    assert result.metadata["thread_id"] != main_thread_id

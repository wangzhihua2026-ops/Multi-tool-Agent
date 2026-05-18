import asyncio

import pytest

from app.core.exceptions import ToolExecutionError
from app.tools.builtins.calculator import evaluate_arithmetic_expression, register_calculator_tool
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


def test_calculator_evaluates_basic_arithmetic() -> None:
    assert evaluate_arithmetic_expression("(2 + 3) * 4 - 5") == 15
    assert evaluate_arithmetic_expression("7 / 2") == 3.5


def test_calculator_rejects_non_arithmetic_expression() -> None:
    with pytest.raises(ValueError):
        evaluate_arithmetic_expression("__import__('os').system('echo unsafe')")


def test_calculator_tool_wraps_invalid_expression() -> None:
    registry = ToolRegistry()
    register_calculator_tool(registry)
    executor = ToolExecutor(registry)

    with pytest.raises(ToolExecutionError, match="Invalid expression"):
        asyncio.run(executor.execute("calculator", {"expression": "open('secret.txt')"}))

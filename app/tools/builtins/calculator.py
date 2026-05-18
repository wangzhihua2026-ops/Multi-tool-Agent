import ast
import math
import operator

from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition, ToolExecutionResult


MAX_EXPRESSION_LENGTH = 200
MAX_ABSOLUTE_RESULT = 1_000_000_000_000
MAX_POWER_EXPONENT = 12

BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculator_tool(arguments: dict) -> ToolExecutionResult:
    expression = str(arguments.get("expression", "")).strip()
    if not expression:
        return ToolExecutionResult(
            tool_name="calculator",
            content="No expression provided.",
        )

    try:
        result = evaluate_arithmetic_expression(expression)
    except Exception as exc:
        raise ValueError(f"Invalid expression: {expression}") from exc

    return ToolExecutionResult(
        tool_name="calculator",
        content=str(result),
    )


def evaluate_arithmetic_expression(expression: str) -> int | float:
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression is too long.")

    parsed = ast.parse(expression, mode="eval")
    result = _evaluate_node(parsed.body)
    _assert_safe_number(result)
    return int(result) if isinstance(result, float) and result.is_integer() else result


def _evaluate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric literals are allowed.")
        return node.value

    if isinstance(node, ast.UnaryOp):
        operator_fn = UNARY_OPS.get(type(node.op))
        if operator_fn is None:
            raise ValueError("Unsupported unary operator.")
        return _checked_result(operator_fn(_evaluate_node(node.operand)))

    if isinstance(node, ast.BinOp):
        operator_fn = BIN_OPS.get(type(node.op))
        if operator_fn is None:
            raise ValueError("Unsupported binary operator.")
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER_EXPONENT:
            raise ValueError("Exponent is too large.")
        return _checked_result(operator_fn(left, right))

    raise ValueError("Only arithmetic expressions are allowed.")


def _checked_result(value: int | float) -> int | float:
    _assert_safe_number(value)
    return value


def _assert_safe_number(value: int | float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Result is not numeric.")
    if not math.isfinite(float(value)):
        raise ValueError("Result is not finite.")
    if abs(float(value)) > MAX_ABSOLUTE_RESULT:
        raise ValueError("Result is too large.")


def register_calculator_tool(registry: ToolRegistry) -> None:
    registry.register(
        ToolDefinition(
            name="calculator",
            description="Evaluate a simple arithmetic expression.",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        ),
        calculator_tool,
    )

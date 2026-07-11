import random
from enum import StrEnum

import httpx

from app.agent.execution import ErrorClass
from app.tools.schemas import ToolExecutionSemantics


class RetryDecision(StrEnum):
    RETRY = "retry"
    FAIL = "fail"
    NEEDS_ATTENTION = "needs_attention"


def classify_exception(exc: Exception) -> ErrorClass:
    if isinstance(exc, (TimeoutError, httpx.TimeoutException, httpx.NetworkError)):
        return ErrorClass.TRANSIENT
    return ErrorClass.PERMANENT


def retry_decision(
    error_class: ErrorClass,
    attempt: int,
    max_attempts: int,
    semantics: ToolExecutionSemantics = ToolExecutionSemantics.READ_ONLY,
) -> RetryDecision:
    if semantics is ToolExecutionSemantics.NON_IDEMPOTENT_SIDE_EFFECT:
        return RetryDecision.NEEDS_ATTENTION
    if error_class is ErrorClass.TRANSIENT and attempt < max_attempts:
        return RetryDecision.RETRY
    return RetryDecision.FAIL


def retry_delay_seconds(attempt: int, random_value: float | None = None) -> float:
    jitter = random.random() if random_value is None else random_value
    return min(30.0, float(3 ** max(0, attempt - 1)) + jitter)

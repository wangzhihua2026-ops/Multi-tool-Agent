from app.agent.execution import ErrorClass
from app.agent.retry_policy import (
    RetryDecision,
    classify_exception,
    retry_decision,
    retry_delay_seconds,
)
from app.tools.schemas import ToolExecutionSemantics


def test_timeout_is_transient_and_retried() -> None:
    assert classify_exception(TimeoutError("provider timeout")) is ErrorClass.TRANSIENT
    assert retry_decision(ErrorClass.TRANSIENT, attempt=1, max_attempts=3) == RetryDecision.RETRY


def test_permanent_error_is_not_retried() -> None:
    assert retry_decision(ErrorClass.PERMANENT, attempt=1, max_attempts=3) == RetryDecision.FAIL


def test_non_idempotent_side_effect_needs_attention() -> None:
    semantics = ToolExecutionSemantics.NON_IDEMPOTENT_SIDE_EFFECT
    assert retry_decision(ErrorClass.TRANSIENT, 1, 3, semantics) == RetryDecision.NEEDS_ATTENTION


def test_retry_delay_uses_exponential_backoff_with_bounded_jitter() -> None:
    assert retry_delay_seconds(attempt=1, random_value=0.25) == 1.25
    assert retry_delay_seconds(attempt=2, random_value=0.25) == 3.25
    assert retry_delay_seconds(attempt=5, random_value=0.25) == 30.0

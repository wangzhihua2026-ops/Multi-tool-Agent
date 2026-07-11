import pytest

from app.agent.state import RunStatus
from app.agent.state_machine import InvalidRunTransition, transition_run


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunStatus.QUEUED, RunStatus.RUNNING),
        (RunStatus.RUNNING, RunStatus.WAITING_APPROVAL),
        (RunStatus.WAITING_APPROVAL, RunStatus.QUEUED),
        (RunStatus.RUNNING, RunStatus.RETRY_SCHEDULED),
        (RunStatus.RETRY_SCHEDULED, RunStatus.QUEUED),
        (RunStatus.RUNNING, RunStatus.COMPLETED),
        (RunStatus.RUNNING, RunStatus.FAILED),
    ],
)
def test_legal_run_transitions(current: RunStatus, target: RunStatus) -> None:
    assert transition_run(current, target) is target


def test_terminal_run_cannot_transition() -> None:
    with pytest.raises(InvalidRunTransition):
        transition_run(RunStatus.COMPLETED, RunStatus.RUNNING)


def test_waiting_approval_can_be_canceled() -> None:
    assert transition_run(RunStatus.WAITING_APPROVAL, RunStatus.CANCELED) is RunStatus.CANCELED

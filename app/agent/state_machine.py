from app.agent.state import RunStatus


class InvalidRunTransition(ValueError):
    pass


ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.QUEUED: frozenset({RunStatus.RUNNING, RunStatus.CANCELED}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.WAITING_APPROVAL,
            RunStatus.RETRY_SCHEDULED,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELED,
        }
    ),
    RunStatus.WAITING_APPROVAL: frozenset({RunStatus.QUEUED, RunStatus.CANCELED}),
    RunStatus.RETRY_SCHEDULED: frozenset({RunStatus.QUEUED, RunStatus.CANCELED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELED: frozenset(),
}


def transition_run(current: RunStatus, target: RunStatus) -> RunStatus:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidRunTransition(f"Illegal run transition: {current} -> {target}")
    return target

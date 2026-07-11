import asyncio
from datetime import datetime, timezone

from app.agent.execution import (
    CheckpointAction,
    ExecutionCheckpoint,
    StepOutcome,
    StepType,
)
from app.persistence.run_store import InMemoryRunStore, NewRun
from app.services.run_orchestrator import RunOrchestrator


def new_run(run_id: str) -> NewRun:
    return NewRun(
        run_id=run_id,
        session_id="worker-session",
        user_message="calculate",
        created_at=datetime.now(timezone.utc),
    )


def test_duplicate_worker_cannot_claim_running_run() -> None:
    async def scenario() -> None:
        store = InMemoryRunStore()
        await store.create_run_with_outbox(new_run("run-duplicate"))
        runtime = CountingRuntime()
        first = RunOrchestrator(store, runtime, worker_id="worker-a")
        second = RunOrchestrator(store, runtime, worker_id="worker-b")

        assert await first.claim("run-duplicate") is True
        assert await second.claim("run-duplicate") is False

    asyncio.run(scenario())


def test_persisted_plan_checkpoint_resumes_at_tool() -> None:
    async def scenario() -> None:
        store = InMemoryRunStore()
        await store.create_run_with_outbox(new_run("run-resume"))
        runtime = CountingRuntime()
        first = RunOrchestrator(store, runtime, worker_id="worker-a")
        assert await first.claim("run-resume") is True

        await first.execute_next_step("run-resume")
        second = RunOrchestrator(store, runtime, worker_id="worker-a")
        await second.execute_next_step("run-resume")

        assert runtime.plan_calls == 1
        assert runtime.tool_calls == 1
        saved = await store.get_run("run-resume")
        assert saved.checkpoint is not None
        assert saved.checkpoint.pending_action is CheckpointAction.ANSWER

    asyncio.run(scenario())


class CountingRuntime:
    def __init__(self) -> None:
        self.plan_calls = 0
        self.tool_calls = 0

    async def advance(self, checkpoint: ExecutionCheckpoint) -> StepOutcome:
        updated = checkpoint.model_copy(deep=True)
        if updated.pending_action is CheckpointAction.PLAN:
            self.plan_calls += 1
            updated.pending_action = CheckpointAction.TOOL
            updated.pending_tool_name = "calculator"
            updated.pending_tool_arguments = {"expression": "2 + 2"}
            return StepOutcome(step_type=StepType.PLAN, checkpoint=updated)
        self.tool_calls += 1
        updated.pending_action = CheckpointAction.ANSWER
        updated.next_step += 1
        return StepOutcome(step_type=StepType.TOOL_CALL, checkpoint=updated)

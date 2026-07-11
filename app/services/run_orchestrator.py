from typing import Protocol
from uuid import uuid4

from app.agent.execution import (
    CheckpointAction,
    ExecutionCheckpoint,
    StepOutcome,
    StepStatus,
    StepType,
)
from app.agent.state import RunStatus
from app.persistence.run_store import RunStore, StoredStep
from app.queue.run_queue import RunQueue


class ResumableRuntime(Protocol):
    async def advance(self, checkpoint: ExecutionCheckpoint) -> StepOutcome: ...


class RunOrchestrator:
    def __init__(
        self,
        store: RunStore,
        runtime: ResumableRuntime,
        worker_id: str,
        queue: RunQueue | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.worker_id = worker_id
        self.queue = queue
        self.lease_seconds = lease_seconds

    async def claim(self, run_id: str) -> bool:
        claimed = await self.store.claim_run(
            run_id,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        return claimed is not None

    async def execute_next_step(self, run_id: str) -> StepOutcome:
        run = await self.store.get_run(run_id)
        checkpoint = run.checkpoint or ExecutionCheckpoint(
            run_id=run.run_id,
            session_id=run.session_id,
            user_message=run.user_message,
        )
        expected_type = self._step_type_for(checkpoint.pending_action)
        idempotency_key = self.step_key(
            run_id,
            checkpoint.next_step,
            expected_type,
        )
        completed = await self.store.get_completed_step(run_id, idempotency_key)
        if completed is not None:
            return StepOutcome(
                step_type=completed.step_type,
                checkpoint=completed.checkpoint,
            )

        outcome = await self.runtime.advance(checkpoint)
        target_status = RunStatus.RUNNING
        if outcome.waiting_for_approval:
            target_status = RunStatus.WAITING_APPROVAL
        elif outcome.completed:
            target_status = RunStatus.COMPLETED

        step = StoredStep(
            step_id=str(uuid4()),
            run_id=run_id,
            sequence=checkpoint.next_step,
            step_type=outcome.step_type,
            status=StepStatus.COMPLETED,
            idempotency_key=idempotency_key,
            checkpoint=outcome.checkpoint,
            output={
                "waiting_for_approval": outcome.waiting_for_approval,
                "completed": outcome.completed,
                "final_response": outcome.final_response,
            },
        )
        await self.store.save_step(step, target_status)
        for payload in outcome.events:
            event = await self.store.append_event(
                run_id,
                str(payload["type"]),
                dict(payload.get("data", {})),
            )
            if self.queue is not None:
                await self.queue.publish_event(run_id, event.sequence)
        return outcome

    async def run_to_boundary(self, run_id: str) -> None:
        if not await self.claim(run_id):
            return
        while True:
            outcome = await self.execute_next_step(run_id)
            if outcome.waiting_for_approval or outcome.completed:
                return

    @staticmethod
    def step_key(run_id: str, sequence: int, step_type: StepType) -> str:
        return f"{run_id}:{sequence}:{step_type.value}"

    @staticmethod
    def _step_type_for(action: CheckpointAction) -> StepType:
        if action is CheckpointAction.PLAN:
            return StepType.PLAN
        if action is CheckpointAction.TOOL:
            return StepType.TOOL_CALL
        return StepType.ANSWER

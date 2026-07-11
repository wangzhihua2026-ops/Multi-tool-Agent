import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from app.agent.execution import ExecutionCheckpoint, StepStatus, StepType
from app.agent.state import RunStatus


class NewRun(BaseModel):
    run_id: str
    session_id: str
    user_message: str
    created_at: datetime
    config_snapshot: dict[str, Any] = Field(default_factory=dict)


class StoredRun(BaseModel):
    run_id: str
    session_id: str
    user_message: str
    status: RunStatus
    version: int = 0
    attempt_count: int = 0
    max_attempts: int = 3
    checkpoint: ExecutionCheckpoint | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    next_retry_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class StoredEvent(BaseModel):
    run_id: str
    sequence: int
    event_type: str
    data: dict[str, Any]
    created_at: datetime


class StoredStep(BaseModel):
    step_id: str
    run_id: str
    sequence: int
    step_type: StepType
    status: StepStatus
    idempotency_key: str
    checkpoint: ExecutionCheckpoint
    output: dict[str, Any] = Field(default_factory=dict)


class StoredApproval(BaseModel):
    approval_id: str
    run_id: str
    step_id: str
    status: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class OutboxRecord(BaseModel):
    outbox_id: str
    topic: str
    deduplication_key: str
    payload: dict[str, Any]
    attempt_count: int = 0
    published_at: datetime | None = None


class RunStore(Protocol):
    async def create_run_with_outbox(self, run: NewRun) -> StoredRun: ...

    async def get_run(self, run_id: str) -> StoredRun: ...

    async def claim_run(
        self, run_id: str, worker_id: str, lease_seconds: int
    ) -> StoredRun | None: ...

    async def renew_lease(self, run_id: str, worker_id: str, lease_seconds: int) -> bool: ...

    async def release_lease(self, run_id: str, worker_id: str) -> bool: ...

    async def get_completed_step(
        self, run_id: str, idempotency_key: str
    ) -> StoredStep | None: ...

    async def list_steps(self, run_id: str) -> list[StoredStep]: ...

    async def save_step(self, step: StoredStep, target_status: RunStatus) -> StoredRun: ...

    async def create_pending_approval(
        self,
        run_id: str,
        step_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> StoredApproval: ...

    async def decide_approval(self, approval_id: str, approved: bool, actor: str) -> bool: ...

    async def request_cancel(self, run_id: str) -> StoredRun: ...

    async def schedule_retry(
        self,
        run_id: str,
        next_retry_at: datetime,
        error_code: str,
        error_message: str,
    ) -> StoredRun: ...

    async def append_event(
        self, run_id: str, event_type: str, data: dict[str, Any]
    ) -> StoredEvent: ...

    async def list_events(
        self, run_id: str, after_sequence: int, limit: int
    ) -> list[StoredEvent]: ...

    async def list_unpublished_outbox(self, limit: int) -> list[OutboxRecord]: ...

    async def mark_outbox_published(self, outbox_id: str) -> bool: ...

    async def requeue_expired_leases_with_outbox(self, limit: int) -> int: ...

    async def requeue_due_retries_with_outbox(self, limit: int) -> int: ...


class InMemoryRunStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._runs: dict[str, StoredRun] = {}
        self._events: dict[str, list[StoredEvent]] = {}
        self._outbox: dict[str, OutboxRecord] = {}
        self._steps: dict[tuple[str, str], StoredStep] = {}
        self._approvals: dict[str, StoredApproval] = {}

    async def create_run_with_outbox(self, run: NewRun) -> StoredRun:
        async with self._lock:
            if run.run_id in self._runs:
                raise ValueError(f"Run already exists: {run.run_id}")
            stored = StoredRun(
                run_id=run.run_id,
                session_id=run.session_id,
                user_message=run.user_message,
                status=RunStatus.QUEUED,
                created_at=run.created_at,
                updated_at=run.created_at,
            )
            outbox = OutboxRecord(
                outbox_id=str(uuid4()),
                topic="agent.run.dispatch",
                deduplication_key=f"dispatch:{run.run_id}:0",
                payload={"run_id": run.run_id},
            )
            self._runs[run.run_id] = stored
            self._events[run.run_id] = []
            self._outbox[outbox.outbox_id] = outbox
            return stored.model_copy(deep=True)

    async def get_run(self, run_id: str) -> StoredRun:
        async with self._lock:
            try:
                return self._runs[run_id].model_copy(deep=True)
            except KeyError as exc:
                raise KeyError(f"Run not found: {run_id}") from exc

    async def claim_run(
        self, run_id: str, worker_id: str, lease_seconds: int
    ) -> StoredRun | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Run not found: {run_id}")
            now = datetime.now(timezone.utc)
            eligible = run.status in {RunStatus.QUEUED, RunStatus.RETRY_SCHEDULED}
            lease_available = run.lease_expires_at is None or run.lease_expires_at < now
            if not eligible or not lease_available:
                return None
            updated = run.model_copy(
                update={
                    "status": RunStatus.RUNNING,
                    "version": run.version + 1,
                    "lease_owner": worker_id,
                    "lease_expires_at": now + timedelta(seconds=lease_seconds),
                    "updated_at": now,
                },
                deep=True,
            )
            self._runs[run_id] = updated
            return updated.model_copy(deep=True)

    async def renew_lease(self, run_id: str, worker_id: str, lease_seconds: int) -> bool:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.lease_owner != worker_id or run.status is not RunStatus.RUNNING:
                return False
            now = datetime.now(timezone.utc)
            self._runs[run_id] = run.model_copy(
                update={
                    "lease_expires_at": now + timedelta(seconds=lease_seconds),
                    "updated_at": now,
                },
                deep=True,
            )
            return True

    async def release_lease(self, run_id: str, worker_id: str) -> bool:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.lease_owner != worker_id:
                return False
            self._runs[run_id] = run.model_copy(
                update={"lease_owner": None, "lease_expires_at": None},
                deep=True,
            )
            return True

    async def get_completed_step(
        self, run_id: str, idempotency_key: str
    ) -> StoredStep | None:
        async with self._lock:
            step = self._steps.get((run_id, idempotency_key))
            if step is None or step.status is not StepStatus.COMPLETED:
                return None
            return step.model_copy(deep=True)

    async def list_steps(self, run_id: str) -> list[StoredStep]:
        async with self._lock:
            return sorted(
                (
                    step.model_copy(deep=True)
                    for (stored_run_id, _), step in self._steps.items()
                    if stored_run_id == run_id
                ),
                key=lambda step: (step.sequence, step.step_type.value),
            )

    async def save_step(self, step: StoredStep, target_status: RunStatus) -> StoredRun:
        async with self._lock:
            run = self._runs.get(step.run_id)
            if run is None:
                raise KeyError(f"Run not found: {step.run_id}")
            key = (step.run_id, step.idempotency_key)
            if key not in self._steps:
                self._steps[key] = step.model_copy(deep=True)
                now = datetime.now(timezone.utc)
                clear_lease = target_status is not RunStatus.RUNNING
                run = run.model_copy(
                    update={
                        "status": target_status,
                        "checkpoint": step.checkpoint.model_copy(deep=True),
                        "version": run.version + 1,
                        "updated_at": now,
                        "lease_owner": None if clear_lease else run.lease_owner,
                        "lease_expires_at": None if clear_lease else run.lease_expires_at,
                    },
                    deep=True,
                )
                self._runs[step.run_id] = run
            return self._runs[step.run_id].model_copy(deep=True)

    async def create_pending_approval(
        self,
        run_id: str,
        step_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> StoredApproval:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Run not found: {run_id}")
            if run.status is not RunStatus.RUNNING:
                raise ValueError(f"Run is not running: {run_id}")
            approval = StoredApproval(
                approval_id=str(uuid4()),
                run_id=run_id,
                step_id=step_id,
                status="pending",
                tool_name=tool_name,
                arguments=arguments,
            )
            now = datetime.now(timezone.utc)
            self._approvals[approval.approval_id] = approval
            self._runs[run_id] = run.model_copy(
                update={
                    "status": RunStatus.WAITING_APPROVAL,
                    "version": run.version + 1,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                },
                deep=True,
            )
            return approval.model_copy(deep=True)

    async def decide_approval(self, approval_id: str, approved: bool, actor: str) -> bool:
        async with self._lock:
            approval = self._approvals.get(approval_id)
            if approval is None or approval.status != "pending":
                return False
            run = self._runs[approval.run_id]
            if run.status is not RunStatus.WAITING_APPROVAL:
                return False
            self._approvals[approval_id] = approval.model_copy(
                update={"status": "approved" if approved else "rejected"},
                deep=True,
            )
            now = datetime.now(timezone.utc)
            version = run.version + 1
            self._runs[run.run_id] = run.model_copy(
                update={
                    "status": RunStatus.QUEUED,
                    "version": version,
                    "updated_at": now,
                },
                deep=True,
            )
            self._add_dispatch_outbox(run.run_id, version, now)
            return True

    async def request_cancel(self, run_id: str) -> StoredRun:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Run not found: {run_id}")
            if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELED}:
                return run.model_copy(deep=True)
            now = datetime.now(timezone.utc)
            if run.status is RunStatus.RUNNING:
                updates: dict[str, Any] = {"cancel_requested_at": now, "updated_at": now}
            else:
                updates = {
                    "status": RunStatus.CANCELED,
                    "version": run.version + 1,
                    "cancel_requested_at": now,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            updated = run.model_copy(update=updates, deep=True)
            self._runs[run_id] = updated
            return updated.model_copy(deep=True)

    async def schedule_retry(
        self,
        run_id: str,
        next_retry_at: datetime,
        error_code: str,
        error_message: str,
    ) -> StoredRun:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Run not found: {run_id}")
            now = datetime.now(timezone.utc)
            updated = run.model_copy(
                update={
                    "status": RunStatus.RETRY_SCHEDULED,
                    "version": run.version + 1,
                    "next_retry_at": next_retry_at,
                    "error_code": error_code,
                    "error_message": error_message,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                },
                deep=True,
            )
            self._runs[run_id] = updated
            return updated.model_copy(deep=True)

    async def requeue_expired_leases_with_outbox(self, limit: int) -> int:
        async with self._lock:
            now = datetime.now(timezone.utc)
            candidates = [
                run
                for run in self._runs.values()
                if run.status is RunStatus.RUNNING
                and run.lease_expires_at is not None
                and run.lease_expires_at < now
            ][:limit]
            for run in candidates:
                version = run.version + 1
                self._runs[run.run_id] = run.model_copy(
                    update={
                        "status": RunStatus.QUEUED,
                        "version": version,
                        "attempt_count": run.attempt_count + 1,
                        "lease_owner": None,
                        "lease_expires_at": None,
                        "updated_at": now,
                    },
                    deep=True,
                )
                self._add_dispatch_outbox(run.run_id, version, now)
            return len(candidates)

    async def requeue_due_retries_with_outbox(self, limit: int) -> int:
        async with self._lock:
            now = datetime.now(timezone.utc)
            candidates = [
                run
                for run in self._runs.values()
                if run.status is RunStatus.RETRY_SCHEDULED
                and run.next_retry_at is not None
                and run.next_retry_at <= now
            ][:limit]
            for run in candidates:
                version = run.version + 1
                self._runs[run.run_id] = run.model_copy(
                    update={
                        "status": RunStatus.QUEUED,
                        "version": version,
                        "next_retry_at": None,
                        "updated_at": now,
                    },
                    deep=True,
                )
                self._add_dispatch_outbox(run.run_id, version, now)
            return len(candidates)

    async def append_event(
        self, run_id: str, event_type: str, data: dict[str, Any]
    ) -> StoredEvent:
        async with self._lock:
            if run_id not in self._runs:
                raise KeyError(f"Run not found: {run_id}")
            event = StoredEvent(
                run_id=run_id,
                sequence=len(self._events[run_id]),
                event_type=event_type,
                data=data,
                created_at=datetime.now(timezone.utc),
            )
            self._events[run_id].append(event)
            return event.model_copy(deep=True)

    async def list_events(
        self, run_id: str, after_sequence: int = -1, limit: int = 100
    ) -> list[StoredEvent]:
        async with self._lock:
            return [
                event.model_copy(deep=True)
                for event in self._events.get(run_id, [])
                if event.sequence > after_sequence
            ][:limit]

    async def list_unpublished_outbox(self, limit: int) -> list[OutboxRecord]:
        async with self._lock:
            return [
                record.model_copy(deep=True)
                for record in self._outbox.values()
                if record.published_at is None
            ][:limit]

    async def mark_outbox_published(self, outbox_id: str) -> bool:
        async with self._lock:
            record = self._outbox.get(outbox_id)
            if record is None or record.published_at is not None:
                return False
            self._outbox[outbox_id] = record.model_copy(
                update={"published_at": datetime.now(timezone.utc)}
            )
            return True

    def _add_dispatch_outbox(self, run_id: str, version: int, created_at: datetime) -> None:
        deduplication_key = f"dispatch:{run_id}:{version}"
        if any(
            record.deduplication_key == deduplication_key
            for record in self._outbox.values()
        ):
            return
        record = OutboxRecord(
            outbox_id=str(uuid4()),
            topic="agent.run.dispatch",
            deduplication_key=deduplication_key,
            payload={"run_id": run_id},
        )
        self._outbox[record.outbox_id] = record

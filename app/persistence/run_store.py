import asyncio
from datetime import datetime, timezone
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

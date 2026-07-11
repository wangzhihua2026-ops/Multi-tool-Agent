from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, insert, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.agent.execution import ExecutionCheckpoint, StepStatus, StepType
from app.agent.state import RunStatus
from app.persistence.db import create_database
from app.persistence.platform_tables import (
    agent_evaluations,
    agent_runs,
    outbox_events,
    run_approvals,
    run_events,
    run_steps,
)
from app.persistence.run_store import (
    NewRun,
    OutboxRecord,
    StoredApproval,
    StoredEvent,
    StoredRun,
    StoredStep,
)


class PostgresRunStore:
    def __init__(
        self,
        engine: AsyncEngine,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._engine = engine
        self._sessions = session_factory

    @classmethod
    def from_url(cls, database_url: str) -> "PostgresRunStore":
        engine, sessions = create_database(database_url)
        return cls(engine, sessions)

    async def close(self) -> None:
        await self._engine.dispose()

    async def clear_for_test(self) -> None:
        async with self._sessions.begin() as session:
            for table in (
                run_events,
                run_approvals,
                run_steps,
                outbox_events,
                agent_evaluations,
                agent_runs,
            ):
                await session.execute(delete(table))

    async def create_run_with_outbox(self, run: NewRun) -> StoredRun:
        outbox_id = str(uuid4())
        async with self._sessions.begin() as session:
            await session.execute(
                insert(agent_runs).values(
                    run_id=run.run_id,
                    session_id=run.session_id,
                    user_message=run.user_message,
                    status=RunStatus.QUEUED.value,
                    version=0,
                    attempt_count=0,
                    max_attempts=3,
                    config_snapshot_json=run.config_snapshot,
                    created_at=run.created_at,
                    updated_at=run.created_at,
                )
            )
            await session.execute(
                insert(outbox_events).values(
                    outbox_id=outbox_id,
                    topic="agent.run.dispatch",
                    deduplication_key=f"dispatch:{run.run_id}:0",
                    payload_json={"run_id": run.run_id},
                    attempt_count=0,
                    created_at=run.created_at,
                )
            )
        return await self.get_run(run.run_id)

    async def get_run(self, run_id: str) -> StoredRun:
        async with self._sessions() as session:
            row = (await session.execute(
                select(agent_runs).where(agent_runs.c.run_id == run_id)
            )).mappings().one()
        return self._run_from_row(row)

    async def claim_run(
        self, run_id: str, worker_id: str, lease_seconds: int
    ) -> StoredRun | None:
        now = datetime.now(timezone.utc)
        statement = (
            update(agent_runs)
            .where(agent_runs.c.run_id == run_id)
            .where(agent_runs.c.status.in_([RunStatus.QUEUED.value, RunStatus.RETRY_SCHEDULED.value]))
            .where(
                or_(
                    agent_runs.c.lease_expires_at.is_(None),
                    agent_runs.c.lease_expires_at < now,
                )
            )
            .values(
                status=RunStatus.RUNNING.value,
                lease_owner=worker_id,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                version=agent_runs.c.version + 1,
                started_at=func.coalesce(agent_runs.c.started_at, now),
                updated_at=now,
            )
            .returning(agent_runs)
        )
        async with self._sessions.begin() as session:
            row = (await session.execute(statement)).mappings().one_or_none()
        return self._run_from_row(row) if row is not None else None

    async def renew_lease(self, run_id: str, worker_id: str, lease_seconds: int) -> bool:
        now = datetime.now(timezone.utc)
        statement = (
            update(agent_runs)
            .where(agent_runs.c.run_id == run_id)
            .where(agent_runs.c.status == RunStatus.RUNNING.value)
            .where(agent_runs.c.lease_owner == worker_id)
            .values(
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                updated_at=now,
            )
        )
        async with self._sessions.begin() as session:
            result = await session.execute(statement)
        return result.rowcount == 1

    async def release_lease(self, run_id: str, worker_id: str) -> bool:
        statement = (
            update(agent_runs)
            .where(agent_runs.c.run_id == run_id)
            .where(agent_runs.c.lease_owner == worker_id)
            .values(lease_owner=None, lease_expires_at=None)
        )
        async with self._sessions.begin() as session:
            result = await session.execute(statement)
        return result.rowcount == 1

    async def get_completed_step(
        self, run_id: str, idempotency_key: str
    ) -> StoredStep | None:
        statement = (
            select(run_steps)
            .where(run_steps.c.run_id == run_id)
            .where(run_steps.c.idempotency_key == idempotency_key)
            .where(run_steps.c.status == StepStatus.COMPLETED.value)
        )
        async with self._sessions() as session:
            row = (await session.execute(statement)).mappings().one_or_none()
        if row is None:
            return None
        return StoredStep(
            step_id=row["step_id"],
            run_id=row["run_id"],
            sequence=row["sequence"],
            step_type=StepType(row["step_type"]),
            status=StepStatus(row["status"]),
            idempotency_key=row["idempotency_key"],
            checkpoint=ExecutionCheckpoint.model_validate(row["checkpoint_json"]),
            output=row["output_json"],
        )

    async def list_steps(self, run_id: str) -> list[StoredStep]:
        statement = (
            select(run_steps)
            .where(run_steps.c.run_id == run_id)
            .order_by(run_steps.c.sequence, run_steps.c.step_type)
        )
        async with self._sessions() as session:
            rows = (await session.execute(statement)).mappings().all()
        return [
            StoredStep(
                step_id=row["step_id"],
                run_id=row["run_id"],
                sequence=row["sequence"],
                step_type=StepType(row["step_type"]),
                status=StepStatus(row["status"]),
                idempotency_key=row["idempotency_key"],
                checkpoint=ExecutionCheckpoint.model_validate(row["checkpoint_json"]),
                output=row["output_json"],
            )
            for row in rows
        ]

    async def save_step(self, step: StoredStep, target_status: RunStatus) -> StoredRun:
        now = datetime.now(timezone.utc)
        statement = (
            pg_insert(run_steps)
            .values(
                step_id=step.step_id,
                run_id=step.run_id,
                sequence=step.sequence,
                step_type=step.step_type.value,
                status=step.status.value,
                idempotency_key=step.idempotency_key,
                checkpoint_json=step.checkpoint.model_dump(mode="json"),
                input_json={},
                output_json=step.output,
                attempt_count=0,
                completed_at=now,
            )
            .on_conflict_do_nothing(
                index_elements=[run_steps.c.run_id, run_steps.c.idempotency_key]
            )
        )
        async with self._sessions.begin() as session:
            result = await session.execute(statement)
            if result.rowcount == 1:
                clear_lease = target_status is not RunStatus.RUNNING
                await session.execute(
                    update(agent_runs)
                    .where(agent_runs.c.run_id == step.run_id)
                    .values(
                        status=target_status.value,
                        checkpoint_json=step.checkpoint.model_dump(mode="json"),
                        version=agent_runs.c.version + 1,
                        updated_at=now,
                        lease_owner=None if clear_lease else agent_runs.c.lease_owner,
                        lease_expires_at=None if clear_lease else agent_runs.c.lease_expires_at,
                        completed_at=now
                        if target_status in {
                            RunStatus.COMPLETED,
                            RunStatus.FAILED,
                            RunStatus.CANCELED,
                        }
                        else agent_runs.c.completed_at,
                    )
                )
        return await self.get_run(step.run_id)

    async def create_pending_approval(
        self,
        run_id: str,
        step_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> StoredApproval:
        approval_id = str(uuid4())
        now = datetime.now(timezone.utc)
        async with self._sessions.begin() as session:
            await session.execute(
                insert(run_approvals).values(
                    approval_id=approval_id,
                    run_id=run_id,
                    step_id=step_id,
                    status="pending",
                    tool_name=tool_name,
                    arguments_json=arguments,
                    risk_level="high",
                    requested_at=now,
                )
            )
            await session.execute(
                update(agent_runs)
                .where(agent_runs.c.run_id == run_id)
                .where(agent_runs.c.status == RunStatus.RUNNING.value)
                .values(
                    status=RunStatus.WAITING_APPROVAL.value,
                    lease_owner=None,
                    lease_expires_at=None,
                    version=agent_runs.c.version + 1,
                    updated_at=now,
                )
            )
        return StoredApproval(
            approval_id=approval_id,
            run_id=run_id,
            step_id=step_id,
            status="pending",
            tool_name=tool_name,
            arguments=arguments,
        )

    async def decide_approval(self, approval_id: str, approved: bool, actor: str) -> bool:
        now = datetime.now(timezone.utc)
        async with self._sessions.begin() as session:
            pending = (
                await session.execute(
                    select(
                        run_approvals.c.run_id,
                        agent_runs.c.version,
                    )
                    .select_from(
                        run_approvals.join(
                            agent_runs,
                            run_approvals.c.run_id == agent_runs.c.run_id,
                        )
                    )
                    .where(run_approvals.c.approval_id == approval_id)
                    .where(run_approvals.c.status == "pending")
                    .where(agent_runs.c.status == RunStatus.WAITING_APPROVAL.value)
                    .with_for_update()
                )
            ).mappings().one_or_none()
            if pending is None:
                return False
            await session.execute(
                update(run_approvals)
                .where(run_approvals.c.approval_id == approval_id)
                .values(
                    status="approved" if approved else "rejected",
                    decision_by=actor,
                    decided_at=now,
                )
            )
            version = int(pending["version"]) + 1
            await session.execute(
                update(agent_runs)
                .where(agent_runs.c.run_id == pending["run_id"])
                .values(
                    status=RunStatus.QUEUED.value,
                    version=version,
                    updated_at=now,
                )
            )
            await session.execute(
                pg_insert(outbox_events)
                .values(
                    outbox_id=str(uuid4()),
                    topic="agent.run.dispatch",
                    deduplication_key=f"dispatch:{pending['run_id']}:{version}",
                    payload_json={"run_id": pending["run_id"]},
                    attempt_count=0,
                    created_at=now,
                )
                .on_conflict_do_nothing(
                    index_elements=[outbox_events.c.deduplication_key]
                )
            )
        return True

    async def request_cancel(self, run_id: str) -> StoredRun:
        now = datetime.now(timezone.utc)
        async with self._sessions.begin() as session:
            row = (
                await session.execute(
                    select(agent_runs)
                    .where(agent_runs.c.run_id == run_id)
                    .with_for_update()
                )
            ).mappings().one()
            status = RunStatus(row["status"])
            if status not in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELED}:
                values: dict[str, Any] = {
                    "cancel_requested_at": now,
                    "updated_at": now,
                }
                if status is not RunStatus.RUNNING:
                    values.update(
                        status=RunStatus.CANCELED.value,
                        version=int(row["version"]) + 1,
                        lease_owner=None,
                        lease_expires_at=None,
                        completed_at=now,
                    )
                await session.execute(
                    update(agent_runs)
                    .where(agent_runs.c.run_id == run_id)
                    .values(**values)
                )
        return await self.get_run(run_id)

    async def schedule_retry(
        self,
        run_id: str,
        next_retry_at: datetime,
        error_code: str,
        error_message: str,
    ) -> StoredRun:
        now = datetime.now(timezone.utc)
        async with self._sessions.begin() as session:
            await session.execute(
                update(agent_runs)
                .where(agent_runs.c.run_id == run_id)
                .where(agent_runs.c.status == RunStatus.RUNNING.value)
                .values(
                    status=RunStatus.RETRY_SCHEDULED.value,
                    version=agent_runs.c.version + 1,
                    next_retry_at=next_retry_at,
                    error_code=error_code,
                    error_message=error_message,
                    lease_owner=None,
                    lease_expires_at=None,
                    updated_at=now,
                )
            )
        return await self.get_run(run_id)

    async def requeue_expired_leases_with_outbox(self, limit: int) -> int:
        now = datetime.now(timezone.utc)
        async with self._sessions.begin() as session:
            rows = (
                await session.execute(
                    select(agent_runs.c.run_id, agent_runs.c.version)
                    .where(agent_runs.c.status == RunStatus.RUNNING.value)
                    .where(agent_runs.c.lease_expires_at < now)
                    .order_by(agent_runs.c.lease_expires_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).mappings().all()
            for row in rows:
                version = int(row["version"]) + 1
                await session.execute(
                    update(agent_runs)
                    .where(agent_runs.c.run_id == row["run_id"])
                    .values(
                        status=RunStatus.QUEUED.value,
                        version=version,
                        attempt_count=agent_runs.c.attempt_count + 1,
                        lease_owner=None,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                await self._insert_dispatch_outbox(session, row["run_id"], version, now)
        return len(rows)

    async def requeue_due_retries_with_outbox(self, limit: int) -> int:
        now = datetime.now(timezone.utc)
        async with self._sessions.begin() as session:
            rows = (
                await session.execute(
                    select(agent_runs.c.run_id, agent_runs.c.version)
                    .where(agent_runs.c.status == RunStatus.RETRY_SCHEDULED.value)
                    .where(agent_runs.c.next_retry_at <= now)
                    .order_by(agent_runs.c.next_retry_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).mappings().all()
            for row in rows:
                version = int(row["version"]) + 1
                await session.execute(
                    update(agent_runs)
                    .where(agent_runs.c.run_id == row["run_id"])
                    .values(
                        status=RunStatus.QUEUED.value,
                        version=version,
                        next_retry_at=None,
                        updated_at=now,
                    )
                )
                await self._insert_dispatch_outbox(session, row["run_id"], version, now)
        return len(rows)

    async def append_event(
        self, run_id: str, event_type: str, data: dict[str, Any]
    ) -> StoredEvent:
        now = datetime.now(timezone.utc)
        async with self._sessions.begin() as session:
            await session.execute(
                select(agent_runs.c.run_id)
                .where(agent_runs.c.run_id == run_id)
                .with_for_update()
            )
            sequence = int((await session.execute(
                select(func.coalesce(func.max(run_events.c.sequence), -1)).where(
                    run_events.c.run_id == run_id
                )
            )).scalar_one()) + 1
            await session.execute(
                insert(run_events).values(
                    run_id=run_id,
                    sequence=sequence,
                    event_type=event_type,
                    data_json=data,
                    created_at=now,
                )
            )
        return StoredEvent(
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            data=data,
            created_at=now,
        )

    async def list_events(
        self, run_id: str, after_sequence: int = -1, limit: int = 100
    ) -> list[StoredEvent]:
        statement = (
            select(run_events)
            .where(run_events.c.run_id == run_id)
            .where(run_events.c.sequence > after_sequence)
            .order_by(run_events.c.sequence)
            .limit(limit)
        )
        async with self._sessions() as session:
            rows = (await session.execute(statement)).mappings().all()
        return [
            StoredEvent(
                run_id=row["run_id"],
                sequence=row["sequence"],
                event_type=row["event_type"],
                data=row["data_json"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def list_unpublished_outbox(self, limit: int) -> list[OutboxRecord]:
        statement = (
            select(outbox_events)
            .where(outbox_events.c.published_at.is_(None))
            .order_by(outbox_events.c.created_at)
            .limit(limit)
        )
        async with self._sessions() as session:
            rows = (await session.execute(statement)).mappings().all()
        return [
            OutboxRecord(
                outbox_id=row["outbox_id"],
                topic=row["topic"],
                deduplication_key=row["deduplication_key"],
                payload=row["payload_json"],
                attempt_count=row["attempt_count"],
                published_at=row["published_at"],
            )
            for row in rows
        ]

    async def mark_outbox_published(self, outbox_id: str) -> bool:
        statement = (
            update(outbox_events)
            .where(outbox_events.c.outbox_id == outbox_id)
            .where(outbox_events.c.published_at.is_(None))
            .values(published_at=datetime.now(timezone.utc))
        )
        async with self._sessions.begin() as session:
            result = await session.execute(statement)
        return result.rowcount == 1

    def _run_from_row(self, row: Any) -> StoredRun:
        checkpoint = row["checkpoint_json"]
        return StoredRun(
            run_id=row["run_id"],
            session_id=row["session_id"],
            user_message=row["user_message"],
            status=RunStatus(row["status"]),
            version=row["version"],
            attempt_count=row["attempt_count"],
            max_attempts=row["max_attempts"],
            checkpoint=ExecutionCheckpoint.model_validate(checkpoint) if checkpoint else None,
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            cancel_requested_at=row["cancel_requested_at"],
            next_retry_at=row["next_retry_at"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def _insert_dispatch_outbox(
        self,
        session: AsyncSession,
        run_id: str,
        version: int,
        created_at: datetime,
    ) -> None:
        await session.execute(
            pg_insert(outbox_events)
            .values(
                outbox_id=str(uuid4()),
                topic="agent.run.dispatch",
                deduplication_key=f"dispatch:{run_id}:{version}",
                payload_json={"run_id": run_id},
                attempt_count=0,
                created_at=created_at,
            )
            .on_conflict_do_nothing(
                index_elements=[outbox_events.c.deduplication_key]
            )
        )

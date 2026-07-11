import asyncio
import os

import pytest

from app.persistence.postgres_run_store import PostgresRunStore
from app.agent.execution import CheckpointAction, ExecutionCheckpoint, StepStatus, StepType
from app.agent.state import RunStatus
from app.persistence.run_store import NewRun, StoredStep


pytestmark = pytest.mark.skipif(
    "TEST_DATABASE_URL" not in os.environ,
    reason="TEST_DATABASE_URL is required for PostgreSQL integration tests",
)

CLAIM_RUN_ID = "00000000-0000-0000-0000-000000000001"
APPROVAL_RUN_ID = "00000000-0000-0000-0000-000000000002"
APPROVAL_STEP_ID = "00000000-0000-0000-0000-000000000003"


def test_postgres_create_claim_and_duplicate_claim() -> None:
    async def scenario() -> None:
        store = PostgresRunStore.from_url(os.environ["TEST_DATABASE_URL"])
        await store.clear_for_test()
        await store.create_run_with_outbox(NewRun.model_validate({
            "run_id": CLAIM_RUN_ID,
            "session_id": "integration",
            "user_message": "hello",
            "created_at": "2026-07-12T00:00:00Z",
        }))

        first = await store.claim_run(CLAIM_RUN_ID, worker_id="worker-a", lease_seconds=30)
        second = await store.claim_run(CLAIM_RUN_ID, worker_id="worker-b", lease_seconds=30)

        assert first is not None
        assert second is None
        await store.close()

    asyncio.run(scenario())


def test_postgres_approval_compare_and_set_is_single_use() -> None:
    async def scenario() -> None:
        store = PostgresRunStore.from_url(os.environ["TEST_DATABASE_URL"])
        await store.clear_for_test()
        await store.create_run_with_outbox(NewRun.model_validate({
            "run_id": APPROVAL_RUN_ID,
            "session_id": "integration",
            "user_message": "send email",
            "created_at": "2026-07-12T00:00:00Z",
        }))
        claimed = await store.claim_run(APPROVAL_RUN_ID, worker_id="worker-a", lease_seconds=30)
        assert claimed is not None
        approval = await store.create_pending_approval(
            APPROVAL_RUN_ID,
            APPROVAL_STEP_ID,
            "send_email",
            {"to": "reviewer@example.com"},
        )

        assert await store.decide_approval(
            approval.approval_id, approved=True, actor="tester"
        ) is True
        assert await store.decide_approval(
            approval.approval_id, approved=True, actor="tester"
        ) is False
        await store.close()

    asyncio.run(scenario())


def test_postgres_outbox_publish_compare_and_set() -> None:
    async def scenario() -> None:
        store = PostgresRunStore.from_url(os.environ["TEST_DATABASE_URL"])
        await store.clear_for_test()
        await store.create_run_with_outbox(NewRun.model_validate({
            "run_id": "00000000-0000-0000-0000-000000000004",
            "session_id": "integration",
            "user_message": "dispatch",
            "created_at": "2026-07-12T00:00:00Z",
        }))
        records = await store.list_unpublished_outbox(limit=10)

        assert await store.mark_outbox_published(records[0].outbox_id) is True
        assert await store.mark_outbox_published(records[0].outbox_id) is False
        assert await store.list_unpublished_outbox(limit=10) == []
        await store.close()

    asyncio.run(scenario())


def test_postgres_step_checkpoint_is_idempotent() -> None:
    async def scenario() -> None:
        run_id = "00000000-0000-0000-0000-000000000005"
        store = PostgresRunStore.from_url(os.environ["TEST_DATABASE_URL"])
        await store.clear_for_test()
        await store.create_run_with_outbox(NewRun.model_validate({
            "run_id": run_id,
            "session_id": "integration",
            "user_message": "checkpoint",
            "created_at": "2026-07-12T00:00:00Z",
        }))
        assert await store.claim_run(run_id, "worker-a", 30) is not None
        checkpoint = ExecutionCheckpoint(
            run_id=run_id,
            session_id="integration",
            user_message="checkpoint",
            pending_action=CheckpointAction.TOOL,
            pending_tool_name="calculator",
            pending_tool_arguments={"expression": "2 + 2"},
        )
        step = StoredStep(
            step_id="00000000-0000-0000-0000-000000000006",
            run_id=run_id,
            sequence=1,
            step_type=StepType.PLAN,
            status=StepStatus.COMPLETED,
            idempotency_key=f"{run_id}:1:plan",
            checkpoint=checkpoint,
        )

        first = await store.save_step(step, RunStatus.RUNNING)
        second = await store.save_step(step, RunStatus.RUNNING)
        completed = await store.get_completed_step(run_id, step.idempotency_key)

        assert first.checkpoint == checkpoint
        assert second.checkpoint == checkpoint
        assert completed is not None
        assert completed.step_id == step.step_id
        await store.close()

    asyncio.run(scenario())


def test_postgres_expired_lease_requeues_once() -> None:
    async def scenario() -> None:
        run_id = "00000000-0000-0000-0000-000000000007"
        store = PostgresRunStore.from_url(os.environ["TEST_DATABASE_URL"])
        await store.clear_for_test()
        await store.create_run_with_outbox(NewRun.model_validate({
            "run_id": run_id,
            "session_id": "integration",
            "user_message": "recover",
            "created_at": "2026-07-12T00:00:00Z",
        }))
        initial = (await store.list_unpublished_outbox(10))[0]
        await store.mark_outbox_published(initial.outbox_id)
        assert await store.claim_run(run_id, "dead-worker", -1) is not None

        assert await store.requeue_expired_leases_with_outbox(100) == 1
        assert await store.requeue_expired_leases_with_outbox(100) == 0
        assert (await store.get_run(run_id)).status is RunStatus.QUEUED
        assert len(await store.list_unpublished_outbox(10)) == 1
        await store.close()

    asyncio.run(scenario())

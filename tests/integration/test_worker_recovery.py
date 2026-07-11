import asyncio
import os

import pytest

from app.agent.state import RunStatus
from app.persistence.postgres_run_store import PostgresRunStore
from app.persistence.run_store import NewRun
from app.services.recovery_service import RecoveryService


pytestmark = pytest.mark.skipif(
    "TEST_DATABASE_URL" not in os.environ,
    reason="TEST_DATABASE_URL is required for PostgreSQL integration tests",
)


def test_expired_worker_lease_is_requeued_once() -> None:
    async def scenario() -> None:
        store = PostgresRunStore.from_url(os.environ["TEST_DATABASE_URL"])
        await store.clear_for_test()
        run_id = "00000000-0000-0000-0000-000000000101"
        await store.create_run_with_outbox(NewRun.model_validate({"run_id": run_id, "session_id": "recovery", "user_message": "resume", "created_at": "2026-07-12T00:00:00Z"}))
        initial = (await store.list_unpublished_outbox(10))[0]
        await store.mark_outbox_published(initial.outbox_id)
        assert await store.claim_run(run_id, "dead-worker", -1) is not None
        recovery = RecoveryService(store)
        assert await recovery.requeue_expired_leases() == 1
        assert await recovery.requeue_expired_leases() == 0
        assert (await store.get_run(run_id)).status is RunStatus.QUEUED
        assert len(await store.list_unpublished_outbox(10)) == 1
        await store.close()

    asyncio.run(scenario())


def test_cancel_waiting_approval_prevents_resume() -> None:
    async def scenario() -> None:
        store = PostgresRunStore.from_url(os.environ["TEST_DATABASE_URL"])
        await store.clear_for_test()
        run_id = "00000000-0000-0000-0000-000000000102"
        await store.create_run_with_outbox(NewRun.model_validate({"run_id": run_id, "session_id": "recovery", "user_message": "send", "created_at": "2026-07-12T00:00:00Z"}))
        assert await store.claim_run(run_id, "worker", 30) is not None
        approval = await store.create_pending_approval(run_id, "00000000-0000-0000-0000-000000000103", "send_email", {"to": "reviewer@example.com"})
        canceled = await store.request_cancel(run_id)
        assert canceled.status is RunStatus.CANCELED
        assert await store.decide_approval(approval.approval_id, True, "tester") is False
        await store.close()

    asyncio.run(scenario())

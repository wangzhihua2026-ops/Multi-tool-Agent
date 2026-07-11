import asyncio
import os

import pytest

from app.persistence.postgres_run_store import PostgresRunStore
from app.persistence.run_store import NewRun


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

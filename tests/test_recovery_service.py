import asyncio
from datetime import datetime, timedelta, timezone

from app.agent.state import RunStatus
from app.persistence.run_store import InMemoryRunStore, NewRun
from app.services.recovery_service import RecoveryService


def make_run(run_id: str) -> NewRun:
    return NewRun(
        run_id=run_id,
        session_id="recovery-session",
        user_message="recover",
        created_at=datetime.now(timezone.utc),
    )


def test_recovery_requeues_expired_lease_once() -> None:
    async def scenario() -> None:
        store = InMemoryRunStore()
        await store.create_run_with_outbox(make_run("expired-run"))
        initial = (await store.list_unpublished_outbox(10))[0]
        await store.mark_outbox_published(initial.outbox_id)
        assert await store.claim_run("expired-run", "dead-worker", -1) is not None
        service = RecoveryService(store)

        assert await service.requeue_expired_leases(limit=100) == 1
        assert await service.requeue_expired_leases(limit=100) == 0
        assert (await store.get_run("expired-run")).status is RunStatus.QUEUED
        assert len(await store.list_unpublished_outbox(10)) == 1

    asyncio.run(scenario())


def test_recovery_requeues_due_retry_once() -> None:
    async def scenario() -> None:
        store = InMemoryRunStore()
        await store.create_run_with_outbox(make_run("retry-run"))
        initial = (await store.list_unpublished_outbox(10))[0]
        await store.mark_outbox_published(initial.outbox_id)
        assert await store.claim_run("retry-run", "worker-a", 30) is not None
        await store.schedule_retry(
            "retry-run",
            next_retry_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            error_code="provider_timeout",
            error_message="temporary",
        )
        service = RecoveryService(store)

        assert await service.requeue_due_retries(limit=100) == 1
        assert await service.requeue_due_retries(limit=100) == 0
        assert (await store.get_run("retry-run")).status is RunStatus.QUEUED

    asyncio.run(scenario())

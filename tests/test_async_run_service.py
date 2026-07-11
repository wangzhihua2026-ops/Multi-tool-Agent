import asyncio
from datetime import datetime, timezone

from app.agent.state import RunStatus
from app.persistence.run_store import InMemoryRunStore, NewRun
from app.services.async_run_service import AsyncRunService


def new_run(run_id: str) -> NewRun:
    return NewRun(
        run_id=run_id,
        session_id="control-session",
        user_message="send email",
        created_at=datetime.now(timezone.utc),
    )


def test_duplicate_approval_creates_one_resume_dispatch() -> None:
    async def scenario() -> None:
        store = InMemoryRunStore()
        await store.create_run_with_outbox(new_run("approval-run"))
        initial = (await store.list_unpublished_outbox(10))[0]
        await store.mark_outbox_published(initial.outbox_id)
        assert await store.claim_run("approval-run", "worker-a", 30) is not None
        approval = await store.create_pending_approval(
            "approval-run", "step-1", "send_email", {"to": "reviewer@example.com"}
        )
        service = AsyncRunService(store)

        first = await service.decide_approval(
            "approval-run", approval.approval_id, approved=True, actor="reviewer"
        )
        second = await service.decide_approval(
            "approval-run", approval.approval_id, approved=True, actor="reviewer"
        )

        assert first.accepted is True
        assert second.accepted is False
        assert len(await store.list_unpublished_outbox(10)) == 1
        assert (await store.get_run("approval-run")).status is RunStatus.QUEUED

    asyncio.run(scenario())


def test_cancel_waiting_run_is_terminal() -> None:
    async def scenario() -> None:
        store = InMemoryRunStore()
        await store.create_run_with_outbox(new_run("cancel-run"))
        assert await store.claim_run("cancel-run", "worker-a", 30) is not None
        approval = await store.create_pending_approval(
            "cancel-run", "step-1", "send_email", {"to": "reviewer@example.com"}
        )
        service = AsyncRunService(store)

        canceled = await service.cancel_run("cancel-run")
        decision = await service.decide_approval(
            "cancel-run", approval.approval_id, approved=True, actor="reviewer"
        )

        assert canceled.status is RunStatus.CANCELED
        assert decision.accepted is False

    asyncio.run(scenario())

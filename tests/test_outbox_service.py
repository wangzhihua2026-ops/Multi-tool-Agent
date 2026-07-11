import asyncio
from datetime import datetime, timezone

from app.persistence.run_store import InMemoryRunStore, NewRun
from app.queue.run_queue import InMemoryRunQueue
from app.services.outbox_service import OutboxService


def make_run(run_id: str) -> NewRun:
    return NewRun(
        run_id=run_id,
        session_id="outbox-session",
        user_message="hello",
        created_at=datetime.now(timezone.utc),
    )


def test_outbox_dispatch_marks_record_after_enqueue() -> None:
    async def scenario() -> None:
        store = InMemoryRunStore()
        queue = InMemoryRunQueue()
        await store.create_run_with_outbox(make_run("run-outbox"))
        service = OutboxService(store, queue)

        assert await service.dispatch_once(limit=10) == 1
        assert await queue.pop() == "run-outbox"
        assert await service.dispatch_once(limit=10) == 0

    asyncio.run(scenario())


def test_queue_delivery_deduplicates_message_key() -> None:
    async def scenario() -> None:
        queue = InMemoryRunQueue()
        await queue.enqueue_run("run-repeat", message_key="dispatch:run-repeat:0")
        await queue.enqueue_run("run-repeat", message_key="dispatch:run-repeat:0")

        assert await queue.pop() == "run-repeat"
        assert await queue.pop() is None

    asyncio.run(scenario())

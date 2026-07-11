import asyncio
from datetime import datetime, timezone

from app.agent.state import RunStatus
from app.persistence.run_store import InMemoryRunStore, NewRun


def make_run(run_id: str) -> NewRun:
    return NewRun(
        run_id=run_id,
        session_id="session-1",
        user_message="hello",
        created_at=datetime.now(timezone.utc),
        config_snapshot={"model": "mock"},
    )


def test_create_run_also_creates_dispatch_outbox() -> None:
    async def scenario() -> None:
        store = InMemoryRunStore()
        await store.create_run_with_outbox(make_run("run-1"))

        saved = await store.get_run("run-1")
        outbox = await store.list_unpublished_outbox(limit=10)

        assert saved.status is RunStatus.QUEUED
        assert outbox[0].deduplication_key == "dispatch:run-1:0"

    asyncio.run(scenario())


def test_append_event_assigns_monotonic_sequence() -> None:
    async def scenario() -> None:
        store = InMemoryRunStore()
        await store.create_run_with_outbox(make_run("run-2"))

        first = await store.append_event("run-2", "run.queued", {"status": "queued"})
        second = await store.append_event("run-2", "worker.claimed", {"worker": "w1"})

        assert [first.sequence, second.sequence] == [0, 1]

    asyncio.run(scenario())

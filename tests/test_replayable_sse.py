import asyncio
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.api.dependencies import get_platform_run_store
from app.api.server import app
from app.persistence.run_store import InMemoryRunStore, NewRun


def test_sse_replay_starts_after_requested_sequence() -> None:
    async def seed(store: InMemoryRunStore) -> str:
        run_id = "sse-run"
        await store.create_run_with_outbox(
            NewRun(
                run_id=run_id,
                session_id="sse-session",
                user_message="hello",
                created_at=datetime.now(timezone.utc),
            )
        )
        await store.append_event(run_id, "run.queued", {"status": "queued"})
        await store.append_event(run_id, "worker.claimed", {"worker": "w1"})
        return run_id

    store = InMemoryRunStore()
    run_id = asyncio.run(seed(store))
    app.dependency_overrides[get_platform_run_store] = lambda: store
    try:
        with TestClient(app) as client:
            response = client.get(
                f"/api/runs/{run_id}/events?after_sequence=0&follow=false",
                headers={"accept": "text/event-stream"},
            )
    finally:
        app.dependency_overrides.pop(get_platform_run_store, None)

    assert response.status_code == 200
    assert "id: 1\n" in response.text
    assert "id: 0\n" not in response.text


def test_terminal_run_event_stream_closes_without_waiting() -> None:
    async def seed(store: InMemoryRunStore) -> str:
        run_id = "terminal-sse-run"
        await store.create_run_with_outbox(
            NewRun(
                run_id=run_id,
                session_id="sse-session",
                user_message="hello",
                created_at=datetime.now(timezone.utc),
            )
        )
        await store.append_event(run_id, "run.queued", {"status": "queued"})
        await store.request_cancel(run_id)
        return run_id

    store = InMemoryRunStore()
    run_id = asyncio.run(seed(store))
    app.dependency_overrides[get_platform_run_store] = lambda: store
    try:
        with TestClient(app) as client:
            response = client.get(f"/api/runs/{run_id}/events")
    finally:
        app.dependency_overrides.pop(get_platform_run_store, None)

    assert response.status_code == 200
    assert "event: run.queued" in response.text

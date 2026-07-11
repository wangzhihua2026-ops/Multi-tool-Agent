import asyncio
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.api.dependencies import get_optional_platform_run_store, get_platform_run_store
from app.api.server import app
from app.core.config import get_settings
from app.persistence.run_store import InMemoryRunStore, NewRun
from app.agent.execution import ExecutionCheckpoint, StepStatus, StepType
from app.agent.state import RunStatus
from app.persistence.run_store import StoredStep


def test_create_run_returns_202() -> None:
    store = InMemoryRunStore()
    with platform_client(store) as client:
        response = client.post(
            "/api/runs",
            json={"session_id": "s1", "message": "hello"},
        )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["run_id"]


def test_get_run_uses_durable_store_when_worker_enabled(monkeypatch) -> None:
    store = InMemoryRunStore()
    created = asyncio.run(
        store.create_run_with_outbox(
            NewRun(
                run_id="durable-detail-run",
                session_id="s1",
                user_message="hello",
                created_at=datetime.now(timezone.utc),
            )
        )
    )
    monkeypatch.setenv("AGENT_WORKER_ENABLED", "true")
    get_settings.cache_clear()
    try:
        with platform_client(store) as client:
            response = client.get(f"/api/runs/{created.run_id}")
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    assert response.json()["run_id"] == created.run_id
    assert response.json()["status"] == "queued"


def test_cancel_endpoint_is_idempotent() -> None:
    store = InMemoryRunStore()
    with platform_client(store) as client:
        created = client.post(
            "/api/runs",
            json={"session_id": "s1", "message": "hello"},
        ).json()
        first = client.post(f"/api/runs/{created['run_id']}/cancel")
        second = client.post(f"/api/runs/{created['run_id']}/cancel")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["status"] == "canceled"


def test_nested_approval_endpoint_is_single_use() -> None:
    async def seed(store: InMemoryRunStore):
        run_id = "approval-api-run"
        await store.create_run_with_outbox(
            NewRun(
                run_id=run_id,
                session_id="s1",
                user_message="send email",
                created_at=datetime.now(timezone.utc),
            )
        )
        assert await store.claim_run(run_id, "worker-a", 30) is not None
        approval = await store.create_pending_approval(
            run_id,
            "step-1",
            "send_email",
            {"to": "reviewer@example.com"},
        )
        return run_id, approval.approval_id

    store = InMemoryRunStore()
    run_id, approval_id = asyncio.run(seed(store))
    payload = {"approved": True, "actor": "reviewer"}
    with platform_client(store) as client:
        first = client.post(
            f"/api/runs/{run_id}/approvals/{approval_id}",
            json=payload,
        )
        second = client.post(
            f"/api/runs/{run_id}/approvals/{approval_id}",
            json=payload,
        )

    assert first.json() == {"accepted": True}
    assert second.json() == {"accepted": False}


def test_steps_endpoint_returns_persisted_steps() -> None:
    async def seed(store: InMemoryRunStore) -> str:
        run_id = "steps-api-run"
        await store.create_run_with_outbox(
            NewRun(
                run_id=run_id,
                session_id="s1",
                user_message="hello",
                created_at=datetime.now(timezone.utc),
            )
        )
        assert await store.claim_run(run_id, "worker-a", 30) is not None
        checkpoint = ExecutionCheckpoint(
            run_id=run_id,
            session_id="s1",
            user_message="hello",
        )
        await store.save_step(
            StoredStep(
                step_id="step-api-1",
                run_id=run_id,
                sequence=1,
                step_type=StepType.PLAN,
                status=StepStatus.COMPLETED,
                idempotency_key=f"{run_id}:1:plan",
                checkpoint=checkpoint,
            ),
            RunStatus.RUNNING,
        )
        return run_id

    store = InMemoryRunStore()
    run_id = asyncio.run(seed(store))
    with platform_client(store) as client:
        response = client.get(f"/api/runs/{run_id}/steps")

    assert response.status_code == 200
    assert response.json()[0]["step_type"] == "plan"


class platform_client:
    def __init__(self, store: InMemoryRunStore) -> None:
        self.store = store
        self.client: TestClient | None = None

    def __enter__(self) -> TestClient:
        app.dependency_overrides[get_platform_run_store] = lambda: self.store
        app.dependency_overrides[get_optional_platform_run_store] = lambda: self.store
        self.client = TestClient(app)
        return self.client

    def __exit__(self, exc_type, exc, traceback) -> None:
        app.dependency_overrides.pop(get_platform_run_store, None)
        app.dependency_overrides.pop(get_optional_platform_run_store, None)

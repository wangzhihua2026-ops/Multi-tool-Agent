import asyncio

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.worker import WorkerSettings, execute_run, shutdown, startup


def test_execute_run_delegates_to_orchestrator() -> None:
    async def scenario() -> None:
        orchestrator = RecordingOrchestrator()
        await execute_run({"orchestrator": orchestrator}, "run-worker")
        assert orchestrator.run_ids == ["run-worker"]

    asyncio.run(scenario())


def test_worker_heartbeat_must_be_shorter_than_lease() -> None:
    with pytest.raises(ValidationError, match="heartbeat"):
        Settings(
            agent_worker_enabled=True,
            agent_worker_lease_seconds=10,
            agent_worker_heartbeat_seconds=10,
        )


def test_worker_settings_register_lifecycle_and_job() -> None:
    assert WorkerSettings.functions == [execute_run]
    assert WorkerSettings.on_startup is startup
    assert WorkerSettings.on_shutdown is shutdown


def test_worker_startup_requires_database_url(monkeypatch) -> None:
    monkeypatch.delenv("PLATFORM_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="PLATFORM_DATABASE_URL"):
            asyncio.run(startup({}))
    finally:
        get_settings.cache_clear()


def test_worker_shutdown_closes_resources() -> None:
    async def scenario() -> None:
        store = CloseRecorder()
        redis = CloseRecorder()
        await shutdown({"store": store, "redis": redis})
        assert store.closed is True
        assert redis.closed is True

    asyncio.run(scenario())


class RecordingOrchestrator:
    def __init__(self) -> None:
        self.run_ids: list[str] = []

    async def run_to_boundary(self, run_id: str) -> None:
        self.run_ids.append(run_id)


class CloseRecorder:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True

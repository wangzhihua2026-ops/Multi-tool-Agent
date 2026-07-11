from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.api.dependencies import get_durable_chat_streamer
from app.api.server import app
from app.core.config import get_settings
from app.persistence.run_store import StoredEvent


def test_chat_stream_uses_durable_facade_when_worker_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_WORKER_ENABLED", "true")
    get_settings.cache_clear()
    app.dependency_overrides[get_durable_chat_streamer] = lambda: FakeDurableChatStreamer()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/chat/stream",
                json={"session_id": "durable-session", "message": "hello"},
            )
    finally:
        app.dependency_overrides.pop(get_durable_chat_streamer, None)
        get_settings.cache_clear()

    assert response.status_code == 200
    assert "assistant.message" in response.text
    assert "durable response" in response.text


class FakeDurableChatStreamer:
    async def stream(self, request):
        yield StoredEvent(
            run_id="durable-run",
            sequence=0,
            event_type="assistant.message",
            data={"content": "durable response"},
            created_at=datetime.now(timezone.utc),
        )

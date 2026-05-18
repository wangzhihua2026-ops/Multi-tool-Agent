from fastapi.testclient import TestClient

from app.api.dependencies import get_message_repository, get_run_repository
from app.api.server import app


client = TestClient(app)


def test_session_messages_api_returns_persisted_conversation() -> None:
    get_run_repository().clear()
    get_message_repository().clear()

    client.post(
        "/api/chat/stream",
        json={"session_id": "session-99", "message": "First turn"},
    )
    second_response = client.post(
        "/api/chat/stream",
        json={"session_id": "session-99", "message": "Second turn"},
    )
    assert second_response.status_code == 200
    assert "Recent context" in second_response.text

    messages_response = client.get("/api/sessions/session-99/messages", params={"limit": 10})
    assert messages_response.status_code == 200
    messages = messages_response.json()
    assert len(messages) == 4
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]

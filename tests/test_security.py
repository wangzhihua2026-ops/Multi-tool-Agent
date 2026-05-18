from fastapi.testclient import TestClient

from app.api.server import app
from app.core.config import get_settings


def test_remote_api_request_is_blocked_without_token() -> None:
    client = TestClient(app, client=("203.0.113.10", 50000))

    response = client.get("/api/documents")

    assert response.status_code == 403


def test_remote_api_request_accepts_configured_token(monkeypatch) -> None:
    monkeypatch.setenv("API_AUTH_TOKEN", "secret-token")
    get_settings.cache_clear()
    client = TestClient(app, client=("203.0.113.10", 50000))

    rejected = client.get("/api/documents", headers={"x-api-key": "wrong"})
    accepted = client.get("/api/documents", headers={"x-api-key": "secret-token"})

    assert rejected.status_code == 401
    assert accepted.status_code == 200


def test_request_id_header_is_returned() -> None:
    client = TestClient(app)

    response = client.get("/api/health", headers={"x-request-id": "request-123"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "request-123"

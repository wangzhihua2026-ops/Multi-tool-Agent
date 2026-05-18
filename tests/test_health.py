from fastapi.testclient import TestClient

from app.api.server import app


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_deep_health_check_reports_components() -> None:
    response = client.get("/api/health/deep")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert set(payload["checks"]) == {
        "knowledge_store",
        "vector_store",
        "embedding_provider",
        "llm_provider",
    }
    assert payload["checks"]["vector_store"]["backend"] == "memory"

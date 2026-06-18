from fastapi.testclient import TestClient

from app.api.server import app


client = TestClient(app)


def test_root_redirects_to_web_app() -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/app/"


def test_app_config_exposes_runtime_prefix() -> None:
    response = client.get("/app-config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["apiPrefix"] == "/api"
    assert payload["appName"]


def test_web_app_shell_is_served() -> None:
    response = client.get("/app/")
    assert response.status_code == 200
    assert "Multi Tool Agent Console" in response.text
    assert 'id="document-file"' in response.text
    assert 'id="document-upload-panel"' in response.text
    assert "Add or Update Knowledge" in response.text
    assert "PDF, DOCX, or image files" in response.text
    assert "./app.js" in response.text


def test_web_app_javascript_allows_image_uploads() -> None:
    response = client.get("/app/app.js")
    assert response.status_code == 200
    assert '".png"' in response.text
    assert '"image/png"' in response.text

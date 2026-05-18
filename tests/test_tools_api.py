from fastapi.testclient import TestClient

from app.api.server import app


client = TestClient(app)


def test_tools_api_lists_builtin_tools() -> None:
    response = client.get("/api/tools")
    assert response.status_code == 200
    tools = response.json()
    tool_names = {tool["name"] for tool in tools}
    assert "calculator" in tool_names
    assert "extract_document_items" in tool_names
    assert "extract_ccf_c_journals" in tool_names
    assert "search_knowledge_base" in tool_names

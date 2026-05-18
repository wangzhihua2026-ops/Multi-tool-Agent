import asyncio

from fastapi.testclient import TestClient

from app.api.dependencies import get_knowledge_store, get_tool_executor
from app.api.server import app
from app.rag.models import DocumentRecord


client = TestClient(app)


def test_export_file_can_be_downloaded() -> None:
    store = get_knowledge_store()
    store.clear()
    store.add_document(
        DocumentRecord(
            title="project-list",
            content="Alpha project uses Transformer.\nBeta project uses Transformer.",
        ),
        [],
    )

    result = asyncio.run(
        get_tool_executor().execute(
            "extract_document_items",
            {"query": "list all items"},
        )
    )
    csv_export = next(item for item in result.metadata["exports"] if item["format"] == "csv")

    response = client.get(csv_export["url"])

    assert response.status_code == 200
    assert "Alpha project uses Transformer" in response.text


def test_export_download_rejects_path_traversal() -> None:
    response = client.get("/api/exports/bad$name.csv")

    assert response.status_code == 400

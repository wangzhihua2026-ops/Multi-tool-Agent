import base64
from io import BytesIO
from zipfile import ZipFile

from fastapi.testclient import TestClient

from app.api.dependencies import get_knowledge_store, get_reindex_job_service, get_vector_store
from app.api.server import app
from app.core.config import get_settings
from app.services.ocr import OcrResult


client = TestClient(app)


def test_document_upload_list_and_search() -> None:
    get_knowledge_store().clear()
    get_vector_store().clear()

    create_response = client.post(
        "/api/documents",
        json={
            "title": "deployment-guide",
            "content": "Deployment steps: configure environment variables, start the FastAPI service, then check the health endpoint.",
        },
    )
    assert create_response.status_code == 200
    created = create_response.json()
    assert created["title"] == "deployment-guide"
    assert created["chunk_count"] >= 1
    assert created["index_status"] == "ready"
    assert created["index_error"] is None

    list_response = client.get("/api/documents")
    assert list_response.status_code == 200
    listed = list_response.json()
    assert len(listed) == 1
    assert listed[0]["document_id"] == created["document_id"]
    assert listed[0]["index_status"] == "ready"

    detail_response = client.get(f"/api/documents/{created['document_id']}")
    assert detail_response.status_code == 200
    assert "start the FastAPI service" in detail_response.json()["content"]
    assert detail_response.json()["index_status"] == "ready"

    search_response = client.get("/api/documents/search", params={"query": "deploy FastAPI"})
    assert search_response.status_code == 200
    search_hits = search_response.json()
    assert len(search_hits) == 1
    assert search_hits[0]["document_title"] == "deployment-guide"
    assert "health endpoint" in search_hits[0]["content"]
    assert search_hits[0]["retrieval_mode"] in {"hybrid", "vector", "lexical"}


def test_document_file_upload_accepts_text_file() -> None:
    get_knowledge_store().clear()
    get_vector_store().clear()

    create_response = client.post(
        "/api/documents/upload",
        json={
            "file_name": "release-notes.md",
            "content_type": "text/markdown",
            "content_base64": _as_base64(b"# Release notes\nThe upload flow supports Markdown files."),
            "metadata": {"team": "docs"},
        },
    )

    assert create_response.status_code == 200
    created = create_response.json()
    assert created["title"] == "release-notes"
    assert created["metadata"]["file_name"] == "release-notes.md"
    assert created["metadata"]["file_parser"] == "text"

    detail_response = client.get(f"/api/documents/{created['document_id']}")
    assert detail_response.status_code == 200
    assert "supports Markdown files" in detail_response.json()["content"]


def test_document_file_upload_accepts_docx_file() -> None:
    get_knowledge_store().clear()
    get_vector_store().clear()

    create_response = client.post(
        "/api/documents/upload",
        json={
            "title": "word-import",
            "file_name": "word-import.docx",
            "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "content_base64": _as_base64(
                _sample_docx_bytes("DOCX upload keeps Word paragraphs searchable.")
            ),
        },
    )

    assert create_response.status_code == 200
    created = create_response.json()
    assert created["title"] == "word-import"
    assert created["metadata"]["file_parser"] == "docx"

    search_response = client.get("/api/documents/search", params={"query": "Word paragraphs"})
    assert search_response.status_code == 200
    assert search_response.json()[0]["document_title"] == "word-import"


def test_document_file_upload_accepts_pdf_file() -> None:
    get_knowledge_store().clear()
    get_vector_store().clear()

    create_response = client.post(
        "/api/documents/upload",
        json={
            "file_name": "pdf-import.pdf",
            "content_type": "application/pdf",
            "content_base64": _as_base64(
                _sample_pdf_bytes("PDF upload exposes searchable vector knowledge.")
            ),
        },
    )

    assert create_response.status_code == 200
    created = create_response.json()
    assert created["title"] == "pdf-import"
    assert created["metadata"]["file_parser"] in {"pdfplumber", "pdf"}
    assert created["metadata"]["ocr_used"] == "false"
    assert "page_count" in created["metadata"]
    assert "table_count" in created["metadata"]

    detail_response = client.get(f"/api/documents/{created['document_id']}")
    assert detail_response.status_code == 200
    assert "searchable vector knowledge" in detail_response.json()["content"]


def test_document_file_upload_accepts_image_file_with_mocked_ocr(monkeypatch) -> None:
    class FakeOcrEngine:
        def extract_text_from_image(self, image_bytes: bytes) -> OcrResult:
            return OcrResult(lines=["Image upload OCR text is searchable."], warnings=[])

    get_knowledge_store().clear()
    get_vector_store().clear()
    monkeypatch.setattr(
        "app.services.document_file_parser.get_default_ocr_engine",
        lambda: FakeOcrEngine(),
    )

    create_response = client.post(
        "/api/documents/upload",
        json={
            "file_name": "scan.png",
            "content_type": "image/png",
            "content_base64": _as_base64(b"fake-png-bytes"),
        },
    )

    assert create_response.status_code == 200
    created = create_response.json()
    assert created["title"] == "scan"
    assert created["metadata"]["file_parser"] == "ocr"
    assert created["metadata"]["ocr_used"] == "true"
    assert created["metadata"]["page_count"] == "1"

    search_response = client.get("/api/documents/search", params={"query": "OCR text"})
    assert search_response.status_code == 200
    assert search_response.json()[0]["document_title"] == "scan"


def test_document_file_upload_respects_disabled_ocr_setting(monkeypatch) -> None:
    monkeypatch.setenv("DOCUMENT_OCR_ENABLED", "false")
    get_settings.cache_clear()
    get_knowledge_store().clear()
    get_vector_store().clear()
    try:
        create_response = client.post(
            "/api/documents/upload",
            json={
                "file_name": "scan.png",
                "content_type": "image/png",
                "content_base64": _as_base64(b"fake-image-bytes"),
            },
        )
    finally:
        get_settings.cache_clear()

    assert create_response.status_code == 400
    assert "OCR support is disabled" in create_response.json()["detail"]


def test_document_file_upload_rejects_unsupported_file() -> None:
    get_knowledge_store().clear()
    get_vector_store().clear()

    create_response = client.post(
        "/api/documents/upload",
        json={
            "file_name": "installer.exe",
            "content_type": "application/octet-stream",
            "content_base64": _as_base64(b"binary"),
        },
    )

    assert create_response.status_code == 400
    assert "Unsupported file type" in create_response.json()["detail"]


def test_document_file_upload_rejects_oversized_file(monkeypatch) -> None:
    monkeypatch.setenv("DOCUMENT_UPLOAD_MAX_BYTES", "4")
    get_settings.cache_clear()
    try:
        create_response = client.post(
            "/api/documents/upload",
            json={
                "file_name": "too-large.txt",
                "content_type": "text/plain",
                "content_base64": _as_base64(b"12345"),
            },
        )
    finally:
        get_settings.cache_clear()

    assert create_response.status_code == 400
    assert "too large" in create_response.json()["detail"]


def test_chat_stream_uses_uploaded_document_context() -> None:
    get_knowledge_store().clear()
    get_vector_store().clear()
    client.post(
        "/api/documents",
        json={
            "title": "operations-notes",
            "content": "When startup fails, first check whether OPENAI_API_KEY exists, then confirm uvicorn started successfully.",
        },
    )

    response = client.post(
        "/api/chat/stream",
        json={
            "session_id": "session-1",
            "message": "Please search the knowledge base and tell me what to check first when startup fails.",
        },
    )
    assert response.status_code == 200
    assert "OPENAI_API_KEY" in response.text


def test_document_reindex_restores_vector_index() -> None:
    get_knowledge_store().clear()
    get_vector_store().clear()
    get_reindex_job_service().clear()

    create_response = client.post(
        "/api/documents",
        json={
            "title": "reindex-guide",
            "content": "Reindex the knowledge base after changing the embedding model.",
        },
    )
    assert create_response.status_code == 200
    created = create_response.json()

    vector_store = get_vector_store()
    vector_store.clear()
    assert vector_store.count() == 0

    reindex_response = client.post(
        "/api/documents/reindex",
        json={"clear_vector_store": True},
    )
    assert reindex_response.status_code == 202
    job = reindex_response.json()
    assert job["status"] in {"queued", "running"}

    job_response = client.get(f"/api/documents/reindex/{job['job_id']}")
    assert job_response.status_code == 200
    payload = job_response.json()["summary"]

    assert payload["document_count"] == 1
    assert payload["chunk_count"] == created["chunk_count"]
    assert payload["cleared_vector_store"] is True
    assert vector_store.count() == created["chunk_count"]

    search_response = client.get("/api/documents/search", params={"query": "embedding model", "top_k": 3})
    assert search_response.status_code == 200
    search_hits = search_response.json()
    assert len(search_hits) == 1
    assert search_hits[0]["document_title"] == "reindex-guide"


def _as_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _sample_docx_bytes(text: str) -> bytes:
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>{text}</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


def _sample_pdf_bytes(text: str) -> bytes:
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT\n/F1 12 Tf\n72 720 Td\n({escaped_text}) Tj\nET".encode("latin-1")
    return b"\n".join(
        [
            b"%PDF-1.4",
            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >> endobj",
            f"4 0 obj << /Length {len(stream)} >>".encode("ascii"),
            b"stream",
            stream,
            b"endstream",
            b"endobj",
            b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
            b"trailer << /Root 1 0 R >>",
            b"%%EOF",
        ]
    )

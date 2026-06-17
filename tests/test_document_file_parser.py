import base64

from app.rag.models import DocumentFileUploadRequest
from app.services.document_file_parser import build_document_request_from_file_upload, parse_document_file


def test_text_parser_returns_structured_metadata() -> None:
    parsed = parse_document_file(
        file_name="notes.md",
        content_type="text/markdown",
        file_bytes=b"# Notes\nParser metadata should be visible.",
    )

    assert parsed.content == "# Notes\nParser metadata should be visible."
    assert parsed.parser == "text"
    assert parsed.metadata == {
        "ocr_used": "false",
        "page_count": "0",
        "table_count": "0",
        "parse_warnings": "",
        "structured_blocks_count": "1",
    }


def test_file_upload_merges_parser_metadata() -> None:
    file_bytes = b"# Notes\nMetadata merge works."
    request = DocumentFileUploadRequest(
        file_name="notes.md",
        content_type="text/markdown",
        content_base64=base64.b64encode(file_bytes).decode("ascii"),
        metadata={"team": "docs"},
    )

    document = build_document_request_from_file_upload(request)

    assert document.metadata["team"] == "docs"
    assert document.metadata["source"] == "file_upload"
    assert document.metadata["file_name"] == "notes.md"
    assert document.metadata["file_type"] == "text/markdown"
    assert document.metadata["file_size"] == str(len(file_bytes))
    assert document.metadata["file_parser"] == "text"
    assert document.metadata["ocr_used"] == "false"
    assert document.metadata["page_count"] == "0"
    assert document.metadata["table_count"] == "0"
    assert document.metadata["parse_warnings"] == ""
    assert document.metadata["structured_blocks_count"] == "1"


def test_file_upload_generated_metadata_takes_expected_precedence() -> None:
    file_bytes = b"# Notes\nPrecedence works."
    request = DocumentFileUploadRequest(
        file_name="notes.md",
        content_type="text/markdown",
        content_base64=base64.b64encode(file_bytes).decode("ascii"),
        metadata={
            "source": "manual",
            "file_name": "spoofed.md",
            "file_parser": "spoofed",
            "ocr_used": "true",
        },
    )

    document = build_document_request_from_file_upload(request)

    assert document.metadata["source"] == "manual"
    assert document.metadata["file_name"] == "notes.md"
    assert document.metadata["file_parser"] == "text"
    assert document.metadata["ocr_used"] == "false"
    assert document.metadata["file_size"] == str(len(file_bytes))

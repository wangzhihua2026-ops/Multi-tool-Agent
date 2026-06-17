import base64
from dataclasses import dataclass

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


def test_pdf_table_serialization_formats_markdown_like_block() -> None:
    from app.services.document_file_parser import _serialize_pdf_table

    table = [
        ["Name", "Value"],
        ["Latency", "14 ms"],
        ["Recall", "100%"],
    ]

    block = _serialize_pdf_table(table, page_number=3, table_index=1)

    assert "[Page 3 Table 1]" in block
    assert "| Name | Value |" in block
    assert "| --- | --- |" in block
    assert "| Latency | 14 ms |" in block
    assert "| Recall | 100% |" in block

    ragged_block = _serialize_pdf_table(
        [
            ["Header", "Pipe | Header", "Nullable"],
            [None, "Line\nbreak", "A | B"],
            [],
            [None, " ", ""],
            ["Only one cell"],
        ],
        page_number=4,
        table_index=2,
    )

    assert "| Header | Pipe \\| Header | Nullable |" in ragged_block
    assert "|  | Line break | A \\| B |" in ragged_block
    assert "| Only one cell |  |  |" in ragged_block
    assert "|  |   |  |" not in ragged_block


@dataclass
class _FakePdfPage:
    text: str
    tables: list[list[list[str | None]]]

    def extract_text(self) -> str:
        return self.text

    def extract_tables(self) -> list[list[list[str | None]]]:
        return self.tables


class _FakePdf:
    def __init__(self, pages: list[_FakePdfPage]) -> None:
        self.pages = pages

    def __enter__(self) -> "_FakePdf":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class _TableFailurePdfPage:
    def extract_text(self) -> str:
        return "Text survives table extraction failure."

    def extract_tables(self) -> list[list[list[str | None]]]:
        raise RuntimeError("table extraction exploded")


class _TextFailurePdfPage:
    def extract_text(self) -> str:
        raise RuntimeError("text extraction exploded")

    def extract_tables(self) -> list[list[list[str | None]]]:
        return [[["Metric", "Value"], ["Recall", "100%"]]]


def test_pdf_parser_uses_pdfplumber_text_and_tables(monkeypatch) -> None:
    from app.services import document_file_parser

    fake_pdf = _FakePdf(
        [
            _FakePdfPage(
                text="Native PDF text.",
                tables=[
                    [[None, ""], []],
                    [["Metric", "Value"], ["Recall", "100%"]],
                ],
            )
        ]
    )

    class _FakePdfPlumber:
        @staticmethod
        def open(file_object):
            return fake_pdf

    monkeypatch.setattr(document_file_parser, "_load_pdfplumber", lambda: _FakePdfPlumber)

    parsed = parse_document_file(
        file_name="metrics.pdf",
        content_type="application/pdf",
        file_bytes=b"%PDF fake",
    )

    assert parsed.parser == "pdfplumber"
    assert "Native PDF text." in parsed.content
    assert "[Page 1 Table 1]" in parsed.content
    assert "| Metric | Value |" in parsed.content
    assert parsed.metadata["page_count"] == "1"
    assert parsed.metadata["table_count"] == "1"
    assert parsed.metadata["ocr_used"] == "false"


def test_pdf_fallback_preserves_pdfplumber_page_count(monkeypatch) -> None:
    from app.services import document_file_parser

    fake_pdf = _FakePdf(
        [
            _FakePdfPage(text="", tables=[]),
            _FakePdfPage(text="", tables=[]),
        ]
    )

    class _FakePdfPlumber:
        @staticmethod
        def open(file_object):
            return fake_pdf

    monkeypatch.setattr(document_file_parser, "_load_pdfplumber", lambda: _FakePdfPlumber)
    monkeypatch.setattr(document_file_parser, "_extract_pdf_text", lambda file_bytes, max_pdf_pages: "Fallback text.")
    monkeypatch.setattr(document_file_parser, "_estimate_basic_pdf_page_count", lambda file_bytes: 0)

    parsed = parse_document_file(
        file_name="fallback.pdf",
        content_type="application/pdf",
        file_bytes=b"%PDF fake",
    )

    assert parsed.content == "Fallback text."
    assert parsed.parser == "pdf"
    assert parsed.metadata["page_count"] == "2"


def test_pdf_parser_keeps_page_text_when_table_extraction_fails(monkeypatch) -> None:
    from app.services import document_file_parser

    fake_pdf = _FakePdf([_TableFailurePdfPage()])

    class _FakePdfPlumber:
        @staticmethod
        def open(file_object):
            return fake_pdf

    monkeypatch.setattr(document_file_parser, "_load_pdfplumber", lambda: _FakePdfPlumber)
    monkeypatch.setattr(document_file_parser, "_extract_pdf_text", lambda file_bytes, max_pdf_pages: "Fallback text.")

    parsed = parse_document_file(
        file_name="partial.pdf",
        content_type="application/pdf",
        file_bytes=b"%PDF fake",
    )

    assert parsed.parser == "pdfplumber"
    assert "Text survives table extraction failure." in parsed.content
    assert "table extraction failed on page 1: table extraction exploded" in parsed.metadata["parse_warnings"]


def test_pdf_parser_keeps_table_when_text_extraction_fails(monkeypatch) -> None:
    from app.services import document_file_parser

    fake_pdf = _FakePdf([_TextFailurePdfPage()])

    class _FakePdfPlumber:
        @staticmethod
        def open(file_object):
            return fake_pdf

    monkeypatch.setattr(document_file_parser, "_load_pdfplumber", lambda: _FakePdfPlumber)
    monkeypatch.setattr(document_file_parser, "_extract_pdf_text", lambda file_bytes, max_pdf_pages: "Fallback text.")

    parsed = parse_document_file(
        file_name="partial.pdf",
        content_type="application/pdf",
        file_bytes=b"%PDF fake",
    )

    assert parsed.parser == "pdfplumber"
    assert "[Page 1 Table 1]" in parsed.content
    assert "| Metric | Value |" in parsed.content
    assert parsed.metadata["table_count"] == "1"
    assert "text extraction failed on page 1: text extraction exploded" in parsed.metadata["parse_warnings"]

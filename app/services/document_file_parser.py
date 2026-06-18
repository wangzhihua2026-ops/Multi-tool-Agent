import base64
import binascii
import re
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from app.rag.models import DocumentCreateRequest, DocumentFileUploadRequest
from app.services.ocr import MissingOcrDependencyError, OcrEngine, get_default_ocr_engine
from app.services.pdf_rendering import render_pdf_pages_to_png_bytes


class DocumentFileParseError(Exception):
    pass


@dataclass(frozen=True)
class ParsedDocumentFile:
    content: str
    parser: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PdfExtractionResult:
    content: str
    parser: str
    page_count: int
    table_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PdfOcrExtractionResult:
    content: str
    page_count: int
    warnings: list[str] = field(default_factory=list)


TEXT_EXTENSIONS = {
    ".txt",
    ".text",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    ".xml",
    ".log",
}
TEXT_CONTENT_TYPES = {
    "application/json",
    "application/ld+json",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
    "application/x-ndjson",
}
PDF_CONTENT_TYPES = {"application/pdf"}
DOCX_CONTENT_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
IMAGE_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


def build_document_request_from_file_upload(
    request: DocumentFileUploadRequest,
    max_file_bytes: int = 20 * 1024 * 1024,
    max_extracted_chars: int = 2_000_000,
    max_pdf_pages: int = 300,
    ocr_enabled: bool = True,
    ocr_max_pages: int = 50,
    ocr_min_native_chars: int = 50,
    ocr_engine: OcrEngine | None = None,
    pdf_page_renderer: Callable[[bytes, int], list[bytes]] | None = None,
) -> DocumentCreateRequest:
    file_bytes = _decode_base64(request.content_base64, max_file_bytes=max_file_bytes)
    file_name = _clean_file_name(request.file_name)
    parsed = parse_document_file(
        file_name=file_name,
        content_type=request.content_type,
        file_bytes=file_bytes,
        max_extracted_chars=max_extracted_chars,
        max_pdf_pages=max_pdf_pages,
        ocr_enabled=ocr_enabled,
        ocr_max_pages=ocr_max_pages,
        ocr_min_native_chars=ocr_min_native_chars,
        ocr_engine=ocr_engine,
        pdf_page_renderer=pdf_page_renderer,
    )
    title = _clean_title(request.title) or _title_from_file_name(file_name)
    metadata = _stringify_metadata(request.metadata)
    metadata.setdefault("source", "file_upload")
    metadata.update(parsed.metadata)
    metadata["file_name"] = file_name
    metadata["file_type"] = request.content_type or _extension(file_name).lstrip(".") or "unknown"
    metadata["file_size"] = str(len(file_bytes))
    metadata["file_parser"] = parsed.parser
    return DocumentCreateRequest(
        title=title,
        content=parsed.content,
        metadata=metadata,
    )


def parse_document_file(
    *,
    file_name: str,
    content_type: str | None,
    file_bytes: bytes,
    max_extracted_chars: int = 2_000_000,
    max_pdf_pages: int = 300,
    ocr_enabled: bool = True,
    ocr_max_pages: int = 50,
    ocr_min_native_chars: int = 50,
    ocr_engine: OcrEngine | None = None,
    pdf_page_renderer: Callable[[bytes, int], list[bytes]] | None = None,
) -> ParsedDocumentFile:
    extension = _extension(file_name)
    normalized_type = (content_type or "").split(";")[0].strip().lower()

    if extension == ".docx" or normalized_type in DOCX_CONTENT_TYPES:
        return ParsedDocumentFile(
            content=_ensure_extracted_text_size(_extract_docx_text(file_bytes), max_extracted_chars),
            parser="docx",
            metadata=_parser_metadata(),
        )

    if extension == ".pdf" or normalized_type in PDF_CONTENT_TYPES:
        pdf_result = _extract_pdf_with_pdfplumber(file_bytes, max_pdf_pages=max_pdf_pages)
        ocr_warnings: list[str] = []
        fallback_error: DocumentFileParseError | None = None
        if pdf_result is not None and not pdf_result.content.strip() and pdf_result.warnings:
            try:
                fallback_content = _extract_pdf_text(file_bytes, max_pdf_pages=max_pdf_pages)
            except DocumentFileParseError as exc:
                fallback_error = exc
            else:
                return ParsedDocumentFile(
                    content=_ensure_extracted_text_size(fallback_content, max_extracted_chars),
                    parser="pdf",
                    metadata=_parser_metadata(
                        page_count=pdf_result.page_count,
                        parse_warnings=pdf_result.warnings,
                    ),
                )
        if pdf_result is not None and len(pdf_result.content.strip()) < ocr_min_native_chars:
            try:
                ocr_result = _extract_pdf_text_with_ocr(
                    file_bytes=file_bytes,
                    ocr_enabled=ocr_enabled,
                    ocr_max_pages=ocr_max_pages,
                    ocr_engine=ocr_engine,
                    pdf_page_renderer=pdf_page_renderer,
                )
            except DocumentFileParseError as exc:
                if not pdf_result.content.strip():
                    raise
                ocr_warnings.append(str(exc))
            else:
                if ocr_result.content:
                    content_blocks = [ocr_result.content]
                    if pdf_result.content.strip():
                        content_blocks.insert(0, pdf_result.content.strip())
                    content = "\n\n".join(content_blocks)
                    return ParsedDocumentFile(
                        content=_ensure_extracted_text_size(content, max_extracted_chars),
                        parser="pdfplumber+ocr",
                        metadata=_parser_metadata(
                            ocr_used=True,
                            page_count=ocr_result.page_count,
                            table_count=pdf_result.table_count,
                            parse_warnings=pdf_result.warnings + ocr_result.warnings,
                            structured_blocks_count=max(1, ocr_result.page_count + pdf_result.table_count),
                        ),
                    )
                ocr_warnings.extend(ocr_result.warnings)
                if not pdf_result.content.strip():
                    ocr_warnings.append("OCR did not find readable text in the PDF.")
        if pdf_result is not None and pdf_result.content:
            return ParsedDocumentFile(
                content=_ensure_extracted_text_size(pdf_result.content, max_extracted_chars),
                parser=pdf_result.parser,
                metadata=_parser_metadata(
                    page_count=pdf_result.page_count,
                    table_count=pdf_result.table_count,
                    parse_warnings=pdf_result.warnings + ocr_warnings,
                    structured_blocks_count=max(1, pdf_result.page_count + pdf_result.table_count),
                ),
            )

        if fallback_error is not None:
            raise fallback_error
        fallback_content = _extract_pdf_text(file_bytes, max_pdf_pages=max_pdf_pages)
        return ParsedDocumentFile(
            content=_ensure_extracted_text_size(fallback_content, max_extracted_chars),
            parser="pdf",
            metadata=_parser_metadata(
                page_count=(
                    pdf_result.page_count
                    if pdf_result is not None
                    else _estimate_basic_pdf_page_count(file_bytes)
                ),
                parse_warnings=pdf_result.warnings if pdf_result is not None else [],
            ),
        )

    if _is_image_file(extension, normalized_type):
        content = _extract_image_text_with_ocr(
            file_bytes=file_bytes,
            ocr_enabled=ocr_enabled,
            ocr_engine=ocr_engine,
        )
        return ParsedDocumentFile(
            content=_ensure_extracted_text_size(content, max_extracted_chars),
            parser="ocr",
            metadata=_parser_metadata(ocr_used=True, page_count=1),
        )

    if _is_text_file(extension, normalized_type):
        return ParsedDocumentFile(
            content=_ensure_extracted_text_size(_decode_text_file(file_bytes), max_extracted_chars),
            parser="text",
            metadata=_parser_metadata(),
        )

    raise DocumentFileParseError(
        "Unsupported file type. Upload TXT, Markdown, CSV, JSON, HTML, XML, PDF, DOCX, or image files."
    )


def _decode_base64(value: str, max_file_bytes: int) -> bytes:
    max_encoded_length = ((max_file_bytes + 2) // 3) * 4
    if len(value) > max_encoded_length:
        raise DocumentFileParseError(
            f"Uploaded file is too large. Maximum size is {_format_bytes(max_file_bytes)}."
        )
    try:
        file_bytes = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise DocumentFileParseError("Uploaded file content is not valid base64.") from exc
    if len(file_bytes) > max_file_bytes:
        raise DocumentFileParseError(
            f"Uploaded file is too large. Maximum size is {_format_bytes(max_file_bytes)}."
        )
    return file_bytes


def _is_text_file(extension: str, content_type: str) -> bool:
    return extension in TEXT_EXTENSIONS or content_type.startswith("text/") or content_type in TEXT_CONTENT_TYPES


def _is_image_file(extension: str, content_type: str) -> bool:
    return extension in IMAGE_EXTENSIONS or content_type in IMAGE_CONTENT_TYPES


def _extract_image_text_with_ocr(
    *,
    file_bytes: bytes,
    ocr_enabled: bool,
    ocr_engine: OcrEngine | None,
) -> str:
    if not ocr_enabled:
        raise DocumentFileParseError("OCR support is disabled for this server.")
    if ocr_engine is not None:
        engine = ocr_engine
    else:
        try:
            engine = get_default_ocr_engine()
        except MissingOcrDependencyError as exc:
            raise DocumentFileParseError(str(exc)) from exc
    try:
        result = engine.extract_text_from_image(file_bytes)
    except MissingOcrDependencyError as exc:
        raise DocumentFileParseError(str(exc)) from exc
    except Exception as exc:
        raise DocumentFileParseError(f"Image OCR failed: {exc}") from exc
    content = result.text.strip()
    if not content:
        raise DocumentFileParseError("OCR did not find readable text in the uploaded image.")
    return content


def _extract_pdf_text_with_ocr(
    *,
    file_bytes: bytes,
    ocr_enabled: bool,
    ocr_max_pages: int,
    ocr_engine: OcrEngine | None,
    pdf_page_renderer: Callable[[bytes, int], list[bytes]] | None,
) -> PdfOcrExtractionResult:
    if not ocr_enabled:
        raise DocumentFileParseError("The PDF did not contain extractable text and OCR support is disabled.")
    renderer = pdf_page_renderer or render_pdf_pages_to_png_bytes
    try:
        page_images = renderer(file_bytes, ocr_max_pages)
    except MissingOcrDependencyError as exc:
        raise DocumentFileParseError(str(exc)) from exc
    except ValueError as exc:
        raise DocumentFileParseError(f"OCR page limit exceeded. {exc}") from exc
    except Exception as exc:
        raise DocumentFileParseError(f"PDF OCR rendering failed: {exc}") from exc

    if len(page_images) > ocr_max_pages:
        raise DocumentFileParseError(
            f"OCR page limit exceeded. PDF OCR page limit is {ocr_max_pages}, got {len(page_images)} rendered pages."
        )

    if ocr_engine is not None:
        engine = ocr_engine
    else:
        try:
            engine = get_default_ocr_engine()
        except MissingOcrDependencyError as exc:
            raise DocumentFileParseError(str(exc)) from exc
    blocks: list[str] = []
    warnings: list[str] = []
    for page_index, image_bytes in enumerate(page_images, start=1):
        try:
            result = engine.extract_text_from_image(image_bytes)
        except MissingOcrDependencyError as exc:
            raise DocumentFileParseError(str(exc)) from exc
        except Exception as exc:
            raise DocumentFileParseError(f"PDF OCR failed on page {page_index}: {exc}") from exc
        warnings.extend(result.warnings)
        if result.text:
            blocks.append(f"[Page {page_index} OCR]\n{result.text}")

    return PdfOcrExtractionResult(
        content="\n\n".join(blocks).strip(),
        page_count=len(page_images),
        warnings=warnings,
    )


def _decode_text_file(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "gb18030", "latin-1"):
        try:
            content = file_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 can decode any byte sequence
        raise DocumentFileParseError("Unable to decode the text file.")

    content = content.replace("\x00", "").strip()
    if not content:
        raise DocumentFileParseError("The uploaded text file did not contain readable text.")
    return content


def _extract_docx_text(file_bytes: bytes) -> str:
    try:
        with ZipFile(BytesIO(file_bytes)) as archive:
            xml_names = [
                "word/document.xml",
                *sorted(
                    name
                    for name in archive.namelist()
                    if re.fullmatch(r"word/(?:header|footer)\d+\.xml", name)
                ),
            ]
            paragraphs: list[str] = []
            for xml_name in xml_names:
                if xml_name in archive.namelist():
                    paragraphs.extend(_extract_word_xml_paragraphs(archive.read(xml_name)))
    except BadZipFile as exc:
        raise DocumentFileParseError("The DOCX file is not a valid Word document.") from exc

    content = "\n".join(paragraph for paragraph in paragraphs if paragraph).strip()
    if not content:
        raise DocumentFileParseError("The DOCX file did not contain readable text.")
    return content


def _extract_word_xml_paragraphs(xml_bytes: bytes) -> list[str]:
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError as exc:
        raise DocumentFileParseError("The DOCX document XML could not be parsed.") from exc

    paragraphs: list[str] = []
    for paragraph in root.iter():
        if not _tag_is(paragraph.tag, "p"):
            continue
        parts: list[str] = []
        for node in paragraph.iter():
            if _tag_is(node.tag, "t") and node.text:
                parts.append(node.text)
            elif _tag_is(node.tag, "tab"):
                parts.append("\t")
            elif _tag_is(node.tag, "br") or _tag_is(node.tag, "cr"):
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _extract_pdf_with_pdfplumber(file_bytes: bytes, max_pdf_pages: int) -> PdfExtractionResult | None:
    pdfplumber = _load_pdfplumber()
    if pdfplumber is None:
        return None

    warnings: list[str] = []
    text_blocks: list[str] = []
    table_count = 0
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            page_count = len(pdf.pages)
            if page_count > max_pdf_pages:
                raise DocumentFileParseError(
                    f"The PDF has {page_count} pages, which exceeds the configured limit of {max_pdf_pages} pages."
                )
            for page_index, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = (page.extract_text() or "").strip()
                except Exception as exc:
                    warnings.append(f"pdfplumber text extraction failed on page {page_index}: {exc}")
                    page_text = ""
                if page_text:
                    text_blocks.append(f"[Page {page_index}]\n{page_text}")
                try:
                    page_tables = page.extract_tables() or []
                except Exception as exc:
                    warnings.append(f"pdfplumber table extraction failed on page {page_index}: {exc}")
                    page_tables = []
                for table in page_tables:
                    table_block = _serialize_pdf_table(
                        table,
                        page_number=page_index,
                        table_index=table_count + 1,
                    )
                    if table_block:
                        table_count += 1
                        text_blocks.append(table_block)
    except DocumentFileParseError:
        raise
    except Exception as exc:
        warnings.append(f"pdfplumber extraction failed: {exc}")
        return PdfExtractionResult(
            content="",
            parser="pdfplumber",
            page_count=0,
            table_count=0,
            warnings=warnings,
        )

    return PdfExtractionResult(
        content="\n\n".join(block for block in text_blocks if block).strip(),
        parser="pdfplumber",
        page_count=page_count,
        table_count=table_count,
        warnings=warnings,
    )


def _load_pdfplumber():
    try:
        import pdfplumber  # type: ignore[import-not-found]
    except ImportError:
        return None
    return pdfplumber


def _serialize_pdf_table(table: list[list[str | None]], page_number: int, table_index: int) -> str:
    rows = [
        [_serialize_pdf_table_cell(cell) for cell in row]
        for row in table
        if row and any(cell is not None and str(cell).strip() for cell in row)
    ]
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    header = normalized_rows[0]
    body = normalized_rows[1:]
    lines = [
        f"[Page {page_number} Table {table_index}]",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _serialize_pdf_table_cell(cell: str | None) -> str:
    if cell is None:
        return ""
    return str(cell).replace("\n", " ").strip().replace("|", "\\|")


def _extract_pdf_text(file_bytes: bytes, max_pdf_pages: int) -> str:
    external_text = _extract_pdf_text_with_optional_library(file_bytes, max_pdf_pages=max_pdf_pages)
    if external_text:
        return external_text

    page_count = _estimate_basic_pdf_page_count(file_bytes)
    if page_count > max_pdf_pages:
        raise DocumentFileParseError(
            f"The PDF has {page_count} pages, which exceeds the configured limit of {max_pdf_pages} pages."
        )

    fallback_text = _extract_pdf_text_basic(file_bytes)
    if fallback_text:
        return fallback_text

    raise DocumentFileParseError(
        "The PDF file did not contain extractable text and OCR support is unavailable."
    )


def _extract_pdf_text_with_optional_library(file_bytes: bytes, max_pdf_pages: int) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        return ""

    try:
        reader = PdfReader(BytesIO(file_bytes))
        page_count = len(reader.pages)
        if page_count > max_pdf_pages:
            raise DocumentFileParseError(
                f"The PDF has {page_count} pages, which exceeds the configured limit of {max_pdf_pages} pages."
            )
        pages = [page.extract_text() or "" for page in reader.pages]
    except DocumentFileParseError:
        raise
    except Exception:
        return ""

    return "\n".join(page.strip() for page in pages if page.strip()).strip()


def _estimate_basic_pdf_page_count(file_bytes: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", file_bytes))


def _extract_pdf_text_basic(file_bytes: bytes) -> str:
    text_parts: list[str] = []
    for stream_data in _iter_pdf_streams(file_bytes):
        decoded_stream = _decode_pdf_stream(stream_data)
        if decoded_stream is None:
            continue
        text_parts.extend(_extract_pdf_text_from_stream(decoded_stream))

    content = "\n".join(part for part in text_parts if part).strip()
    return re.sub(r"[ \t]{2,}", " ", content)


def _iter_pdf_streams(file_bytes: bytes) -> list[bytes]:
    streams: list[bytes] = []
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", file_bytes, flags=re.DOTALL):
        streams.append(match.group(1))
    return streams


def _decode_pdf_stream(stream_data: bytes) -> bytes | None:
    try:
        return zlib.decompress(stream_data)
    except zlib.error:
        return stream_data


def _extract_pdf_text_from_stream(stream_data: bytes) -> list[str]:
    stream = stream_data.decode("latin-1", errors="ignore")
    text_parts: list[str] = []
    for array_match in re.finditer(r"\[(.*?)\]\s*TJ", stream, flags=re.DOTALL):
        text = "".join(_decode_pdf_literal_string(match.group(1)) for match in re.finditer(r"\((.*?)\)", array_match.group(1), flags=re.DOTALL))
        if text.strip():
            text_parts.append(text.strip())

    for literal_match in re.finditer(r"\((.*?)\)\s*(?:Tj|'|\")", stream, flags=re.DOTALL):
        text = _decode_pdf_literal_string(literal_match.group(1)).strip()
        if text:
            text_parts.append(text)

    for hex_match in re.finditer(r"<([0-9A-Fa-f\s]+)>\s*Tj", stream):
        hex_value = re.sub(r"\s+", "", hex_match.group(1))
        if len(hex_value) % 2:
            hex_value += "0"
        try:
            text = bytes.fromhex(hex_value).decode("utf-16-be", errors="ignore").strip()
            if not text:
                text = bytes.fromhex(hex_value).decode("latin-1", errors="ignore").strip()
        except ValueError:
            continue
        if text:
            text_parts.append(text)

    return text_parts


def _decode_pdf_literal_string(value: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character != "\\":
            result.append(character)
            index += 1
            continue

        index += 1
        if index >= len(value):
            break
        escaped = value[index]
        if escaped in "nrtbf":
            result.append({"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}[escaped])
            index += 1
        elif escaped in "\\()":
            result.append(escaped)
            index += 1
        elif escaped in "\r\n":
            if escaped == "\r" and index + 1 < len(value) and value[index + 1] == "\n":
                index += 2
            else:
                index += 1
        elif escaped in "01234567":
            octal = escaped
            index += 1
            for _ in range(2):
                if index < len(value) and value[index] in "01234567":
                    octal += value[index]
                    index += 1
            result.append(chr(int(octal, 8)))
        else:
            result.append(escaped)
            index += 1
    return "".join(result)


def _extension(file_name: str) -> str:
    clean_name = _clean_file_name(file_name)
    if "." not in clean_name:
        return ""
    return f".{clean_name.rsplit('.', 1)[1].lower()}"


def _clean_file_name(file_name: str) -> str:
    normalized = file_name.replace("\\", "/")
    return PurePosixPath(normalized).name.strip() or "uploaded-document"


def _title_from_file_name(file_name: str) -> str:
    clean_name = _clean_file_name(file_name)
    stem = clean_name.rsplit(".", 1)[0].strip() if "." in clean_name else clean_name
    return stem or "uploaded-document"


def _clean_title(value: str | None) -> str:
    return (value or "").strip()


def _stringify_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            result[key] = ""
        elif isinstance(value, str):
            result[key] = value
        else:
            result[key] = str(value)
    return result


def _parser_metadata(
    *,
    ocr_used: bool = False,
    page_count: int = 0,
    table_count: int = 0,
    parse_warnings: list[str] | None = None,
    structured_blocks_count: int = 1,
) -> dict[str, str]:
    warnings = parse_warnings or []
    return {
        "ocr_used": str(ocr_used).lower(),
        "page_count": str(page_count),
        "table_count": str(table_count),
        "parse_warnings": "; ".join(warning for warning in warnings if warning),
        "structured_blocks_count": str(structured_blocks_count),
    }


def _tag_is(tag: str, local_name: str) -> bool:
    return tag == local_name or tag.endswith(f"}}{local_name}")


def _ensure_extracted_text_size(content: str, max_extracted_chars: int) -> str:
    if len(content) > max_extracted_chars:
        raise DocumentFileParseError(
            "Extracted document text is too large. "
            f"Maximum extracted length is {max_extracted_chars} characters."
        )
    return content


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} bytes"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value / (1024 * 1024):.1f} MiB"

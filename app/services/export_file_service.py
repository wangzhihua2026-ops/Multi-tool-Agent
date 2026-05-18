import csv
import re
import zipfile
from pathlib import Path
from uuid import uuid4
from xml.sax.saxutils import escape as xml_escape


EXPORT_FILE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def create_tabular_exports(
    *,
    export_directory: str,
    base_name: str,
    rows: list[list[object]],
) -> list[dict[str, str]]:
    export_dir = Path(export_directory)
    export_dir.mkdir(parents=True, exist_ok=True)
    safe_base = _safe_base_name(base_name)
    export_id = uuid4().hex
    csv_name = f"{safe_base}-{export_id}.csv"
    xlsx_name = f"{safe_base}-{export_id}.xlsx"

    csv_path = safe_export_path(export_directory, csv_name)
    xlsx_path = safe_export_path(export_directory, xlsx_name)
    _write_csv(csv_path, rows)
    _write_xlsx(xlsx_path, rows)
    return [
        {"format": "csv", "file_name": csv_name, "url": f"/api/exports/{csv_name}"},
        {"format": "xlsx", "file_name": xlsx_name, "url": f"/api/exports/{xlsx_name}"},
    ]


def safe_export_path(export_directory: str, file_name: str) -> Path:
    if not EXPORT_FILE_PATTERN.fullmatch(file_name):
        raise ValueError("Invalid export file name.")
    export_dir = Path(export_directory).resolve()
    candidate = (export_dir / file_name).resolve()
    if export_dir != candidate.parent:
        raise ValueError("Export file path escapes the export directory.")
    return candidate


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _write_xlsx(path: Path, rows: list[list[object]]) -> None:
    sheet_xml = _worksheet_xml(rows)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml())
        archive.writestr("_rels/.rels", _root_relationships_xml())
        archive.writestr("xl/workbook.xml", _workbook_xml())
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_relationships_xml())
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def _worksheet_xml(rows: list[list[object]]) -> str:
    row_xml = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for column_index, value in enumerate(row, start=1):
            cell_ref = f"{_column_name(column_index)}{row_index}"
            text = xml_escape("" if value is None else str(value))
            cells.append(
                f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'
            )
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        "</worksheet>"
    )


def _content_types_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )


def _root_relationships_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def _workbook_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Extraction" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )


def _workbook_relationships_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def _safe_base_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-").lower()
    return normalized[:48] or "export"

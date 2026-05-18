import re
from dataclasses import dataclass
from typing import Any

from app.core.exceptions import DocumentNotFoundError
from app.rag.models import DocumentRecord
from app.rag.store import KnowledgeStore
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition, ToolExecutionResult


JOURNAL_HEADING = "中国计算机学会推荐国际学术期刊"
CONFERENCE_HEADING = "中国计算机学会推荐国际学术会议"
C_CLASS_HEADING_RE = re.compile(r"三、\s*C\s*类", re.IGNORECASE)
AB_HEADING_RE = re.compile(r"[一二]、\s*[AB]\s*类", re.IGNORECASE)
AREA_RE = re.compile(r"^[（(](.+?)[）)]$")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
HEADER_RE = re.compile(r"^.*?序号\s+期刊简称\s+期刊全称\s+出版社\s+网址\s*", re.DOTALL)
PUBLISHERS = [
    "Cambridge University Press",
    "Oxford University Press",
    "Taylor & Francis",
    "Higher Education Press",
    "BioMed Central",
    "Science in China Press/Springer",
    "Springer Nature",
    "CCF/Springer",
    "IOS Press",
    "MIT Press",
    "AAAI",
    "ACM",
    "IEEE",
    "Elsevier",
    "Springer",
    "Wiley",
    "SIAM",
    "IET",
]


@dataclass(frozen=True)
class CcfCatalogEntry:
    area: str
    sequence: int
    name: str
    publisher: str
    url: str


@dataclass(frozen=True)
class CcfCatalogBlock:
    area: str
    lines: list[str]


def build_ccf_c_journals_tool(store: KnowledgeStore):
    def ccf_c_journals_tool(arguments: dict[str, Any]) -> ToolExecutionResult:
        document = _select_document(
            store=store,
            document_id=str(arguments.get("document_id", "")).strip(),
            document_title=str(arguments.get("document_title", "")).strip(),
        )
        entries = extract_ccf_c_journal_entries(document.content)
        if not entries:
            return ToolExecutionResult(
                tool_name="extract_ccf_c_journals",
                content=(
                    f'Document "{document.title}" was scanned, but no C-class journal table '
                    "could be extracted. The document may not be a CCF catalog or the PDF text "
                    "may need better table extraction."
                ),
                metadata={
                    "document_id": document.document_id,
                    "document_title": document.title,
                    "entries": [],
                    "direct_answer": True,
                },
            )

        content = _format_entries(document=document, entries=entries)
        return ToolExecutionResult(
            tool_name="extract_ccf_c_journals",
            content=content,
            metadata={
                "document_id": document.document_id,
                "document_title": document.title,
                "entry_count": len(entries),
                "entries": [entry.__dict__ for entry in entries],
                "direct_answer": True,
            },
        )

    return ccf_c_journals_tool


def register_ccf_c_journals_tool(registry: ToolRegistry, store: KnowledgeStore) -> None:
    registry.register(
        ToolDefinition(
            name="extract_ccf_c_journals",
            description=(
                "Exhaustively scan a CCF catalog document and list C-class journals only. "
                "Use this instead of search_knowledge_base when the user asks for all CCF C类期刊 "
                "or asks to exclude conferences."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "document_title": {"type": "string"},
                },
            },
        ),
        build_ccf_c_journals_tool(store),
    )


def extract_ccf_c_journal_entries(content: str) -> list[CcfCatalogEntry]:
    blocks = _extract_c_journal_blocks(content)
    entries: list[CcfCatalogEntry] = []
    seen: set[tuple[str, str, str]] = set()
    for block in blocks:
        for entry in _parse_c_journal_block(block):
            key = (_normalize_key(entry.name), _normalize_key(entry.publisher), entry.url.lower())
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
    return entries


def _select_document(
    *,
    store: KnowledgeStore,
    document_id: str = "",
    document_title: str = "",
) -> DocumentRecord:
    if document_id:
        return store.get_document(document_id)

    candidates = store.list_documents()
    if document_title:
        lowered_title = document_title.lower()
        for summary in candidates:
            if lowered_title in summary.title.lower():
                return store.get_document(summary.document_id)
        raise DocumentNotFoundError(document_title)

    scored_candidates = sorted(
        candidates,
        key=lambda summary: _ccf_document_score(summary.title, summary.metadata),
        reverse=True,
    )
    if scored_candidates and _ccf_document_score(scored_candidates[0].title, scored_candidates[0].metadata) > 0:
        return store.get_document(scored_candidates[0].document_id)
    if candidates:
        return store.get_document(candidates[0].document_id)
    raise DocumentNotFoundError("latest")


def _ccf_document_score(title: str, metadata: dict[str, str]) -> int:
    haystack = " ".join([title, metadata.get("file_name", ""), metadata.get("source", "")]).lower()
    score = 0
    if "ccf" in haystack:
        score += 4
    if "中国计算机学会" in haystack:
        score += 4
    if "推荐" in haystack:
        score += 2
    if "期刊" in haystack:
        score += 2
    return score


def _extract_c_journal_blocks(content: str) -> list[CcfCatalogBlock]:
    lines = [_clean_line(line) for line in content.replace("\r\n", "\n").split("\n")]
    blocks: list[CcfCatalogBlock] = []
    current_kind: str | None = None
    current_area = ""
    active_lines: list[str] | None = None
    active_area = ""

    def flush_active() -> None:
        nonlocal active_lines, active_area
        if active_lines:
            blocks.append(CcfCatalogBlock(area=active_area or "未标注领域", lines=list(active_lines)))
        active_lines = None
        active_area = ""

    for line in lines:
        if not line:
            continue

        area_match = AREA_RE.fullmatch(line)
        if active_lines is None and area_match and _looks_like_area_heading(area_match.group(1)):
            current_area = area_match.group(1).strip()

        if JOURNAL_HEADING in line:
            flush_active()
            current_kind = "journal"
            continue
        if CONFERENCE_HEADING in line:
            flush_active()
            current_kind = "conference"
            continue

        if AB_HEADING_RE.search(line):
            flush_active()

        if C_CLASS_HEADING_RE.search(line):
            flush_active()
            if current_kind == "journal":
                active_lines = [line]
                active_area = current_area
            continue

        if active_lines is not None:
            if "会议简称" in line:
                flush_active()
                continue
            active_lines.append(line)

    flush_active()
    return blocks


def _parse_c_journal_block(block: CcfCatalogBlock) -> list[CcfCatalogEntry]:
    text = _normalize_table_text("\n".join(block.lines))
    text = HEADER_RE.sub("", text).strip()
    if not text:
        return []

    entries: list[CcfCatalogEntry] = []
    position = 0
    for match in URL_RE.finditer(text):
        prefix = text[position:match.start()].strip(" ;,")
        url = _clean_url(match.group(0))
        position = match.end()
        entry = _parse_entry_prefix(area=block.area, prefix=prefix, url=url)
        if entry is not None:
            entries.append(entry)
    return entries


def _parse_entry_prefix(area: str, prefix: str, url: str) -> CcfCatalogEntry | None:
    prefix = HEADER_RE.sub("", prefix).strip()
    row_match = re.search(r"(\d{1,3})\s+(.+)$", prefix, flags=re.DOTALL)
    if row_match is None:
        return None

    sequence = int(row_match.group(1))
    raw_name = _collapse_spaces(row_match.group(2))
    publisher = _extract_publisher(raw_name)
    name = raw_name
    if publisher:
        name = raw_name[: -len(publisher)].strip(" ,-/")
    return CcfCatalogEntry(
        area=area or "未标注领域",
        sequence=sequence,
        name=name or raw_name,
        publisher=publisher or "未知出版社",
        url=url,
    )


def _extract_publisher(value: str) -> str:
    normalized = _collapse_spaces(value)
    for publisher in sorted(PUBLISHERS, key=len, reverse=True):
        if normalized.endswith(publisher):
            return publisher
    return ""


def _looks_like_area_heading(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    lowered = normalized.lower()
    if normalized.startswith("原") or "university" in lowered or "springer" in lowered:
        return False
    return any(separator in normalized for separator in ("/", "与", "和")) or len(normalized) <= 16


def _format_entries(document: DocumentRecord, entries: list[CcfCatalogEntry]) -> str:
    lines = [
        f'已从文档《{document.title}》全文抽取到 {len(entries)} 条 CCF C 类期刊记录（已排除会议表）。',
        "",
        "| 序号 | 领域 | 期刊 | 出版社 | URL |",
        "|---:|---|---|---|---|",
    ]
    for index, entry in enumerate(entries, start=1):
        lines.append(
            "| "
            f"{index} | "
            f"{_escape_table_cell(entry.area)} | "
            f"{_escape_table_cell(entry.name)} | "
            f"{_escape_table_cell(entry.publisher)} | "
            f"{_escape_table_cell(entry.url)} |"
        )
    return "\n".join(lines)


def _normalize_table_text(value: str) -> str:
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "-", value)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"(?<=\.)\s+(?=html\b)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"inde\s+x\.html", "index.html", text, flags=re.IGNORECASE)
    return _collapse_spaces(text)


def _clean_url(value: str) -> str:
    return value.rstrip("。；;,.，")


def _clean_line(value: str) -> str:
    return _collapse_spaces(value.strip())


def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_key(value: str) -> str:
    return re.sub(r"\W+", "", value, flags=re.UNICODE).lower()


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.exceptions import DocumentNotFoundError
from app.rag.models import DocumentRecord
from app.rag.store import KnowledgeStore
from app.services.export_file_service import create_tabular_exports
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition, ToolExecutionResult


JOURNAL_HEADING = "中国计算机学会推荐国际学术期刊"
CONFERENCE_HEADING = "中国计算机学会推荐国际学术会议"
AREA_RE = re.compile(r"^[（(](.+?)[）)]$")
CLASS_HEADING_RE = re.compile(r"^[一二三]、\s*([ABC])\s*类", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
TABLE_HEADER_RE = re.compile(
    r"^.*?序号\s+(?:期刊简称|会议简称)\s+(?:期刊全称|会议全称)\s+出版社\s+网址\s*",
    re.DOTALL,
)
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
DEFAULT_EXTRACTION_LIMIT = 200
MAX_EXTRACTION_LIMIT = 500


@dataclass(frozen=True)
class ExtractionSpec:
    query: str
    class_label: str | None = None
    item_kind: str | None = None
    excluded_kinds: set[str] = field(default_factory=set)
    include_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractedItem:
    area: str
    kind: str
    class_label: str | None
    sequence: int | None
    text: str
    publisher: str = ""
    url: str = ""


@dataclass(frozen=True)
class CcfTableBlock:
    area: str
    kind: str
    class_label: str
    lines: list[str]


def build_document_items_tool(store: KnowledgeStore, export_directory: str | None = None):
    def document_items_tool(arguments: dict[str, Any]) -> ToolExecutionResult:
        query = str(arguments.get("query", "")).strip()
        limit = _bounded_int(arguments.get("limit"), default=DEFAULT_EXTRACTION_LIMIT, minimum=1, maximum=MAX_EXTRACTION_LIMIT)
        offset = _bounded_int(arguments.get("offset"), default=0, minimum=0, maximum=1_000_000)
        spec = parse_extraction_spec(query)
        document = _select_document(
            store=store,
            query=query,
            document_id=str(arguments.get("document_id", "")).strip(),
            document_title=str(arguments.get("document_title", "")).strip(),
        )

        all_items = extract_document_items(document.content, spec)
        total_count = len(all_items)
        items = all_items[offset : offset + limit]
        truncated = offset + len(items) < total_count
        exports = _build_exports(
            export_directory=export_directory,
            document=document,
            items=all_items,
        ) if all_items else []
        if not items:
            return ToolExecutionResult(
                tool_name="extract_document_items",
                content=(
                    f'已全文扫描文档《{document.title}》。'
                    if total_count
                    else f'已全文扫描文档《{document.title}》，但没有找到满足条件的条目。'
                )
                + (
                    f"共找到 {total_count} 条匹配记录，但 offset={offset} 超出了结果范围。"
                    if total_count
                    else "可以尝试补充更明确的类别、关键词或排除条件。"
                ),
                metadata={
                    "document_id": document.document_id,
                    "document_title": document.title,
                    "entry_count": total_count,
                    "returned_count": 0,
                    "offset": offset,
                    "limit": limit,
                    "truncated": False,
                    "items": [],
                    "exports": exports,
                    "direct_answer": True,
                    "strategy": "full_document_extract",
                },
            )

        return ToolExecutionResult(
            tool_name="extract_document_items",
            content=_format_items(
                document=document,
                items=items,
                spec=spec,
                total_count=total_count,
                offset=offset,
                limit=limit,
                truncated=truncated,
                exports=exports,
            ),
            metadata={
                "document_id": document.document_id,
                "document_title": document.title,
                "entry_count": total_count,
                "returned_count": len(items),
                "offset": offset,
                "limit": limit,
                "truncated": truncated,
                "items": [item.__dict__ for item in items],
                "exports": exports,
                "direct_answer": True,
                "strategy": "full_document_extract",
            },
        )

    return document_items_tool


def register_document_items_tool(
    registry: ToolRegistry,
    store: KnowledgeStore,
    export_directory: str | None = None,
) -> None:
    registry.register(
        ToolDefinition(
            name="extract_document_items",
            description=(
                "Exhaustively scan a document and extract every item matching the user's filters. "
                "Use this instead of semantic search when the user asks for all/complete/list/extract "
                "items from a file, especially with include or exclude conditions such as class, type, "
                "journal, conference, or not/exclude."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "document_id": {"type": "string"},
                    "document_title": {"type": "string"},
                    "limit": {"type": "integer", "default": DEFAULT_EXTRACTION_LIMIT},
                    "offset": {"type": "integer", "default": 0},
                },
                "required": ["query"],
            },
        ),
        build_document_items_tool(store, export_directory=export_directory),
    )


def parse_extraction_spec(query: str) -> ExtractionSpec:
    lowered = query.lower()
    compact = re.sub(r"\s+", "", lowered)
    class_match = re.search(r"([abc])(?:类|-class|class)", compact, re.IGNORECASE)
    class_label = class_match.group(1).upper() if class_match else None

    excludes_conference = any(term in lowered for term in ["不要会议", "不是会议", "排除会议", "not conference", "exclude conference"])
    excludes_journal = any(term in lowered for term in ["不要期刊", "不是期刊", "排除期刊", "not journal", "exclude journal"])

    item_kind: str | None = None
    if ("期刊" in query or "journal" in lowered) and not excludes_journal:
        item_kind = "journal"
    if ("会议" in query or "conference" in lowered) and not excludes_conference and item_kind is None:
        item_kind = "conference"

    excluded_kinds: set[str] = set()
    if excludes_conference:
        excluded_kinds.add("conference")
    if excludes_journal:
        excluded_kinds.add("journal")

    include_terms = _extract_quoted_terms(query)
    include_terms.extend(_extract_after_markers(query, ["包含", "含有", "包括"]))
    exclude_terms = _extract_after_markers(query, ["不要", "排除", "不包含", "不是"])
    exclude_terms = [term for term in exclude_terms if term not in {"会议", "期刊"}]

    return ExtractionSpec(
        query=query,
        class_label=class_label,
        item_kind=item_kind,
        excluded_kinds=excluded_kinds,
        include_terms=_dedupe_terms(include_terms),
        exclude_terms=_dedupe_terms(exclude_terms),
    )


def extract_document_items(content: str, spec: ExtractionSpec) -> list[ExtractedItem]:
    ccf_items = _extract_ccf_items(content, spec)
    if ccf_items:
        return ccf_items
    return _extract_generic_filtered_items(content, spec)


def _select_document(
    *,
    store: KnowledgeStore,
    query: str,
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

    if not candidates:
        raise DocumentNotFoundError("latest")

    scored_documents = [
        (
            _document_score(document.title, document.metadata, query, document.content),
            document,
        )
        for document in (store.get_document(summary.document_id) for summary in candidates)
    ]
    scored_documents.sort(key=lambda item: item[0], reverse=True)
    return scored_documents[0][1]


def _document_score(title: str, metadata: dict[str, str], query: str, content: str = "") -> int:
    haystack = " ".join([title, metadata.get("file_name", ""), metadata.get("source", "")]).lower()
    lowered_query = query.lower()
    lowered_content = content.lower()
    score = 0
    for token in re.findall(r"[a-zA-Z0-9]+", lowered_query):
        if len(token) >= 3 and token in haystack:
            score += 3
        if len(token) >= 3 and token in lowered_content[:5000]:
            score += 2
    for term in ["ccf", "中国计算机学会", "推荐", "期刊", "会议"]:
        if term.lower() in lowered_query and term.lower() in haystack:
            score += 4
        if term.lower() in lowered_query and term.lower() in lowered_content[:5000]:
            score += 3
    if JOURNAL_HEADING in content:
        score += 6
    if CONFERENCE_HEADING in content:
        score += 4
    if len(content) > 20_000:
        score += 3
    return score


def _extract_ccf_items(content: str, spec: ExtractionSpec) -> list[ExtractedItem]:
    if JOURNAL_HEADING not in content and CONFERENCE_HEADING not in content:
        return []
    blocks = _extract_ccf_blocks(content, spec)
    items: list[ExtractedItem] = []
    seen: set[tuple[str, str, str, str]] = set()
    for block in blocks:
        for item in _parse_ccf_block(block):
            key = (_normalize_key(item.text), item.publisher.lower(), item.url.lower(), item.kind)
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
    return items


def _extract_ccf_blocks(content: str, spec: ExtractionSpec) -> list[CcfTableBlock]:
    lines = [_clean_line(line) for line in content.replace("\r\n", "\n").split("\n")]
    blocks: list[CcfTableBlock] = []
    current_kind: str | None = None
    current_area = ""
    active_lines: list[str] | None = None
    active_area = ""
    active_kind = ""
    active_class = ""

    def flush_active() -> None:
        nonlocal active_lines, active_area, active_kind, active_class
        if active_lines:
            blocks.append(
                CcfTableBlock(
                    area=active_area or "未标注领域",
                    kind=active_kind,
                    class_label=active_class,
                    lines=list(active_lines),
                )
            )
        active_lines = None
        active_area = ""
        active_kind = ""
        active_class = ""

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

        class_match = CLASS_HEADING_RE.search(line)
        if class_match:
            flush_active()
            class_label = class_match.group(1).upper()
            if _ccf_block_matches_spec(kind=current_kind, class_label=class_label, spec=spec):
                active_lines = [line]
                active_area = current_area
                active_kind = current_kind or "unknown"
                active_class = class_label
            continue

        if active_lines is not None:
            active_lines.append(line)

    flush_active()
    return blocks


def _ccf_block_matches_spec(kind: str | None, class_label: str, spec: ExtractionSpec) -> bool:
    if kind is None:
        return False
    if kind in spec.excluded_kinds:
        return False
    if spec.item_kind and kind != spec.item_kind:
        return False
    if spec.class_label and class_label != spec.class_label:
        return False
    return True


def _parse_ccf_block(block: CcfTableBlock) -> list[ExtractedItem]:
    text = _normalize_table_text("\n".join(block.lines))
    text = TABLE_HEADER_RE.sub("", text).strip()
    if not text:
        return []

    items: list[ExtractedItem] = []
    position = 0
    for match in URL_RE.finditer(text):
        prefix = text[position:match.start()].strip(" ;,")
        url = _clean_url(match.group(0))
        position = match.end()
        item = _parse_ccf_entry_prefix(block=block, prefix=prefix, url=url)
        if item is not None:
            items.append(item)
    return items


def _parse_ccf_entry_prefix(block: CcfTableBlock, prefix: str, url: str) -> ExtractedItem | None:
    prefix = TABLE_HEADER_RE.sub("", prefix).strip()
    row_match = re.search(r"(\d{1,3})\s+(.+)$", prefix, flags=re.DOTALL)
    if row_match is None:
        return None

    sequence = int(row_match.group(1))
    raw_text = _collapse_spaces(row_match.group(2))
    publisher = _extract_publisher(raw_text)
    text = raw_text
    if publisher:
        text = raw_text[: -len(publisher)].strip(" ,-/")
    return ExtractedItem(
        area=block.area or "未标注领域",
        kind=block.kind,
        class_label=block.class_label,
        sequence=sequence,
        text=text or raw_text,
        publisher=publisher or "未知出版社",
        url=url,
    )


def _extract_generic_filtered_items(content: str, spec: ExtractionSpec) -> list[ExtractedItem]:
    candidates = _split_generic_candidates(content)
    include_terms = [term.lower() for term in spec.include_terms]
    exclude_terms = [term.lower() for term in spec.exclude_terms]
    items: list[ExtractedItem] = []
    seen: set[str] = set()

    for candidate in candidates:
        normalized = _collapse_spaces(candidate)
        lowered = normalized.lower()
        if include_terms and not all(term in lowered for term in include_terms):
            continue
        if exclude_terms and any(term in lowered for term in exclude_terms):
            continue
        if spec.item_kind == "journal" and any(term in lowered for term in ["conference", "会议"]):
            continue
        if spec.item_kind == "conference" and any(term in lowered for term in ["journal", "期刊"]):
            continue
        key = _normalize_key(normalized)
        if not key or key in seen:
            continue
        seen.add(key)
        items.append(
            ExtractedItem(
                area="全文匹配",
                kind=spec.item_kind or "item",
                class_label=spec.class_label,
                sequence=len(items) + 1,
                text=normalized,
            )
        )
    return items


def _split_generic_candidates(content: str) -> list[str]:
    normalized = content.replace("\r\n", "\n")
    lines = [_clean_line(line) for line in normalized.split("\n")]
    useful_lines = [line for line in lines if len(line) >= 2]
    if len(useful_lines) >= 2:
        return useful_lines
    return [part.strip() for part in re.split(r"\n\s*\n|[。；;]", normalized) if part.strip()]


def _extract_quoted_terms(query: str) -> list[str]:
    terms: list[str] = []
    for pattern in [r'"([^"]+)"', r"'([^']+)'", r"“([^”]+)”", r"‘([^’]+)’"]:
        terms.extend(match.strip() for match in re.findall(pattern, query) if match.strip())
    return terms


def _extract_after_markers(query: str, markers: list[str]) -> list[str]:
    terms: list[str] = []
    for marker in markers:
        if marker not in query:
            continue
        tail = query.split(marker, 1)[1]
        tail = re.split(r"[，。；;,.]|而不是|但|并且|以及|输出|列出", tail, maxsplit=1)[0]
        tail = _clean_marker_term(tail)
        if 1 <= len(tail) <= 30:
            terms.append(tail)
    return terms


def _clean_marker_term(value: str) -> str:
    term = value.strip(" ：:\t")
    term = re.sub(r"^(所有|全部|完整|这些|那些)\s*", "", term)
    term = re.sub(r"\s*(的)?(条目|记录|项目|内容|结果|items?)$", "", term, flags=re.IGNORECASE)
    return term.strip()


def _dedupe_terms(terms: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = _collapse_spaces(term)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


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
    if normalized.startswith("原 ") or "university" in lowered or "springer" in lowered:
        return False
    return any(separator in normalized for separator in ("/", "与", "和")) or len(normalized) <= 16


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)


def _build_exports(
    *,
    export_directory: str | None,
    document: DocumentRecord,
    items: list[ExtractedItem],
) -> list[dict[str, str]]:
    if not export_directory or not items:
        return []
    return create_tabular_exports(
        export_directory=export_directory,
        base_name=f"extraction-{document.title}",
        rows=_items_to_export_rows(items),
    )


def _items_to_export_rows(items: list[ExtractedItem]) -> list[list[object]]:
    rows: list[list[object]] = [["序号", "类型", "类别", "领域/位置", "条目", "出版社", "URL"]]
    for index, item in enumerate(items, start=1):
        rows.append(
            [
                index,
                _kind_label(item.kind),
                item.class_label or "",
                item.area,
                item.text,
                item.publisher,
                item.url,
            ]
        )
    return rows


def _format_items(
    document: DocumentRecord,
    items: list[ExtractedItem],
    spec: ExtractionSpec,
    total_count: int,
    offset: int,
    limit: int,
    truncated: bool,
    exports: list[dict[str, str]] | None = None,
) -> str:
    filter_summary = _format_filter_summary(spec)
    if offset == 0 and len(items) == total_count:
        lead = f"已从文档《{document.title}》全文抽取到 {total_count} 条匹配记录{filter_summary}。"
    else:
        start = offset + 1
        end = offset + len(items)
        lead = (
            f"已从文档《{document.title}》全文抽取到 {total_count} 条匹配记录{filter_summary}，"
            f"当前返回第 {start}-{end} 条。"
        )
    lines = [
        lead,
        "",
        "| 序号 | 类型 | 类别 | 领域/位置 | 条目 | 出版社 | URL |",
        "|---:|---|---|---|---|---|---|",
    ]
    for index, item in enumerate(items, start=1):
        lines.append(
            "| "
            f"{index} | "
            f"{_escape_table_cell(_kind_label(item.kind))} | "
            f"{_escape_table_cell(item.class_label or '')} | "
            f"{_escape_table_cell(item.area)} | "
            f"{_escape_table_cell(item.text)} | "
            f"{_escape_table_cell(item.publisher)} | "
            f"{_escape_table_cell(item.url)} |"
        )
    if truncated:
        lines.extend(
            [
                "",
                f"结果已分页：本次 limit={limit}, offset={offset}。继续请求时把 offset 设为 {offset + len(items)}。",
            ]
        )
    if exports:
        rendered_exports = "，".join(f"{item['format'].upper()}: {item['url']}" for item in exports)
        lines.extend(["", f"导出文件：{rendered_exports}"])
    return "\n".join(lines)


def _format_filter_summary(spec: ExtractionSpec) -> str:
    parts: list[str] = []
    if spec.class_label:
        parts.append(f"{spec.class_label} 类")
    if spec.item_kind:
        parts.append(_kind_label(spec.item_kind))
    if spec.excluded_kinds:
        parts.append("排除" + "、".join(_kind_label(kind) for kind in sorted(spec.excluded_kinds)))
    if not parts:
        return ""
    return "（" + "，".join(parts) + "）"


def _kind_label(kind: str) -> str:
    return {
        "journal": "期刊",
        "conference": "会议",
        "item": "条目",
    }.get(kind, kind)


def _normalize_table_text(value: str) -> str:
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "-", value)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"(?<=\.)\s+(?=html\b)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"inde\s+x\.html", "index.html", text, flags=re.IGNORECASE)
    return _collapse_spaces(text)


def _clean_url(value: str) -> str:
    return value.rstrip("。；;,.）)")


def _clean_line(value: str) -> str:
    return _collapse_spaces(value.strip())


def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_key(value: str) -> str:
    return re.sub(r"\W+", "", value, flags=re.UNICODE).lower()


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")

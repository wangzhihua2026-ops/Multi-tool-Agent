import asyncio
from zipfile import ZipFile

from app.agent.runtime import AgentRuntime
from app.agent.state import ChatRequest
from app.core.config import Settings
from app.core.llm_gateway import MockLLMGateway
from app.rag.models import DocumentRecord
from app.rag.store import InMemoryKnowledgeStore
from app.tools.builtins.ccf_catalog import register_ccf_c_journals_tool
from app.tools.builtins.document_extractor import (
    extract_document_items,
    parse_extraction_spec,
    register_document_items_tool,
)
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


CCF_SAMPLE = """
中国计算机学会推荐国际学术期刊
（数据库/数据挖掘/内容检索）
一、A 类
序号 期刊简称 期刊全称 出版社 网址
1 TODS ACM Transactions on Database Systems ACM http://dblp.uni-trier.de/db/journals/tods/
二、B 类
序号 期刊简称 期刊全称 出版社 网址
1 TKDD ACM Transactions on Knowledge Discovery from Data ACM http://dblp.uni-trier.de/db/journals/tkdd/
三、C 类
序号 期刊简称 期刊全称 出版社 网址
1 DPD Distributed and Parallel Databases Springer http://dblp.uni-trier.de/db/journals/dpd/
2 I&M Information & Management Elsevier http://dblp.uni-trier.de/db/journals/iam/
中国计算机学会推荐国际学术会议
（数据库/数据挖掘/内容检索）
三、C 类
序号 会议简称 会议全称 出版社 网址
1 APWeb Asia Pacific Web Conference Springer http://dblp.uni-trier.de/db/conf/apweb/
中国计算机学会推荐国际学术期刊
（人工智能）
三、C 类
序号 期刊简称 期刊全称 出版社 网址
1 TALLIP ACM Transactions on Asian and Low-Resource Language
Information Processing ACM http://dblp.uni-trier.de/db/journals/talip/
2 Applied Intelligence Springer http://dblp.uni-trier.de/db/journals/apin/
"""


def test_generic_extractor_handles_ccf_c_journals_without_conferences() -> None:
    spec = parse_extraction_spec("请找出这个文件里的所有C类期刊而不是会议")

    items = extract_document_items(CCF_SAMPLE, spec)

    assert len(items) == 4
    assert {item.kind for item in items} == {"journal"}
    assert {item.class_label for item in items} == {"C"}
    assert any("Distributed and Parallel Databases" in item.text for item in items)
    assert all("APWeb" not in item.text for item in items)


def test_generic_extractor_can_switch_to_conferences() -> None:
    spec = parse_extraction_spec("列出所有 C 类会议")

    items = extract_document_items(CCF_SAMPLE, spec)

    assert len(items) == 1
    assert items[0].kind == "conference"
    assert items[0].class_label == "C"
    assert "APWeb" in items[0].text


def test_generic_extractor_falls_back_to_full_text_filtering() -> None:
    content = """
Alpha project uses Transformer and is active.
Beta project uses CNN and is active.
Gamma project uses Transformer and is archived.
"""
    spec = parse_extraction_spec("列出所有包含 Transformer 的条目，排除 archived")

    items = extract_document_items(content, spec)

    assert len(items) == 1
    assert "Alpha project" in items[0].text


def test_runtime_prefers_generic_document_extractor_when_available() -> None:
    store = InMemoryKnowledgeStore()
    store.add_document(DocumentRecord(title="CCF catalog", content=CCF_SAMPLE), [])
    registry = ToolRegistry()
    register_document_items_tool(registry, store)
    register_ccf_c_journals_tool(registry, store)
    runtime = AgentRuntime(
        settings=Settings(max_tool_steps=3),
        registry=registry,
        executor=ToolExecutor(registry),
        llm_gateway=MockLLMGateway(),
    )

    events = asyncio.run(_collect_events(runtime))
    completed_tools = [event.data.get("tool_name") for event in events if event.type == "tool.completed"]
    answer = next(event.data["content"] for event in events if event.type == "assistant.message")

    assert completed_tools == ["extract_document_items"]
    assert "4 条匹配记录" in answer
    assert "APWeb" not in answer


def test_document_extractor_prefers_structured_full_catalog_over_title_match() -> None:
    store = InMemoryKnowledgeStore()
    store.add_document(DocumentRecord(title="CCF推荐期刊", content="2026年最新CCF推荐期刊"), [])
    store.add_document(DocumentRecord(title="第七版中国计算机学会推荐国际学术会议和期刊目录", content=CCF_SAMPLE), [])
    registry = ToolRegistry()
    register_document_items_tool(registry, store)
    executor = ToolExecutor(registry)

    result = asyncio.run(
        executor.execute(
            "extract_document_items",
            {"query": "请从我上传的ccf推荐投稿期刊文件里找出所有C类期刊，不要会议，输出完整列表"},
        )
    )

    assert result.metadata["document_title"] == "第七版中国计算机学会推荐国际学术会议和期刊目录"
    assert result.metadata["entry_count"] == 4
    assert "APWeb" not in result.content


def test_document_extractor_reports_paginated_truncation() -> None:
    store = InMemoryKnowledgeStore()
    content = "\n".join(f"Item {index} uses Transformer." for index in range(5))
    store.add_document(DocumentRecord(title="long-list", content=content), [])
    registry = ToolRegistry()
    register_document_items_tool(registry, store)
    executor = ToolExecutor(registry)

    result = asyncio.run(
        executor.execute(
            "extract_document_items",
            {"query": "list all items", "limit": 2},
        )
    )

    assert result.metadata["entry_count"] == 5
    assert result.metadata["returned_count"] == 2
    assert result.metadata["truncated"] is True
    assert "offset" in result.content


def test_document_extractor_writes_csv_and_xlsx_exports(tmp_path) -> None:
    store = InMemoryKnowledgeStore()
    store.add_document(DocumentRecord(title="CCF catalog", content=CCF_SAMPLE), [])
    registry = ToolRegistry()
    register_document_items_tool(registry, store, str(tmp_path))
    executor = ToolExecutor(registry)

    result = asyncio.run(
        executor.execute(
            "extract_document_items",
            {"query": "请找出这个文件里的所有C类期刊而不是会议"},
        )
    )

    exports = result.metadata["exports"]
    assert {item["format"] for item in exports} == {"csv", "xlsx"}
    exported_files = {item["format"]: tmp_path / item["file_name"] for item in exports}
    assert "Distributed and Parallel Databases" in exported_files["csv"].read_text(encoding="utf-8-sig")
    with ZipFile(exported_files["xlsx"]) as archive:
        assert "xl/worksheets/sheet1.xml" in archive.namelist()


async def _collect_events(runtime: AgentRuntime) -> list:
    events = []
    request = ChatRequest(
        session_id="generic-extract-session",
        message="请从我上传的ccf推荐投稿期刊文件里找出所有C类期刊，不要会议，输出完整列表",
    )
    async for event in runtime.stream(request):
        events.append(event)
    return events

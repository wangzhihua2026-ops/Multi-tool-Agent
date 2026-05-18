import asyncio

from app.agent.runtime import AgentRuntime
from app.agent.state import ChatRequest
from app.core.config import Settings
from app.core.llm_gateway import FallbackLLMGateway, LLMGateway, MockLLMGateway, OpenAICompatibleGateway
from app.rag.models import DocumentRecord
from app.rag.store import InMemoryKnowledgeStore
from app.tools.builtins.ccf_catalog import (
    extract_ccf_c_journal_entries,
    register_ccf_c_journals_tool,
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


def test_extract_ccf_c_journal_entries_excludes_conferences() -> None:
    entries = extract_ccf_c_journal_entries(CCF_SAMPLE)

    names = [entry.name for entry in entries]
    assert len(entries) == 4
    assert any("Distributed and Parallel Databases" in name for name in names)
    assert any("Information & Management" in name for name in names)
    assert any("TALLIP" in name for name in names)
    assert any("Applied Intelligence" in name for name in names)
    assert all("APWeb" not in name for name in names)


def test_ccf_c_journals_tool_returns_direct_complete_answer() -> None:
    store = InMemoryKnowledgeStore()
    store.add_document(
        DocumentRecord(title="CCF catalog", content=CCF_SAMPLE),
        [],
    )
    registry = ToolRegistry()
    register_ccf_c_journals_tool(registry, store)
    executor = ToolExecutor(registry)

    result = asyncio.run(executor.execute("extract_ccf_c_journals", {}))

    assert result.metadata["direct_answer"] is True
    assert result.metadata["entry_count"] == 4
    assert "APWeb" not in result.content
    assert "Distributed and Parallel Databases" in result.content


def test_runtime_routes_mock_ccf_request_to_extractor() -> None:
    store = InMemoryKnowledgeStore()
    document = DocumentRecord(title="CCF catalog", content=CCF_SAMPLE)
    store.add_document(document, [])
    registry = ToolRegistry()
    register_ccf_c_journals_tool(registry, store)
    executor = ToolExecutor(registry)
    runtime = AgentRuntime(
        settings=Settings(max_tool_steps=3),
        registry=registry,
        executor=executor,
        llm_gateway=MockLLMGateway(),
    )

    events = asyncio.run(_collect_events(runtime))
    completed_tools = [
        event.data.get("tool_name")
        for event in events
        if event.type == "tool.completed"
    ]
    answer = next(event.data["content"] for event in events if event.type == "assistant.message")

    assert completed_tools == ["extract_ccf_c_journals"]
    assert "Distributed and Parallel Databases" in answer
    assert "APWeb" not in answer


def test_openai_gateway_routes_ccf_request_without_network_call() -> None:
    gateway = OpenAICompatibleGateway(
        Settings(
            llm_provider="openai",
            llm_api_key="test-key",
            default_model="test-model",
        )
    )
    registry = ToolRegistry()
    register_ccf_c_journals_tool(registry, InMemoryKnowledgeStore())

    plan = asyncio.run(
        gateway.plan(
            user_message="请找出这个文件里的所有C类期刊而不是会议",
            tools=registry.list_tools(),
        )
    )

    assert plan.action == "call_tool"
    assert plan.tool_call is not None
    assert plan.tool_call.name == "extract_ccf_c_journals"


def test_fallback_gateway_routes_ccf_request_before_primary_network_call() -> None:
    registry = ToolRegistry()
    register_ccf_c_journals_tool(registry, InMemoryKnowledgeStore())
    gateway = FallbackLLMGateway(
        primary=_FailingGateway(),
        fallback=MockLLMGateway(),
    )

    plan = asyncio.run(
        gateway.plan(
            user_message="请从我上传的ccf推荐投稿期刊文件里找出所有C类期刊，不要会议，输出完整列表",
            tools=registry.list_tools(),
        )
    )

    assert plan.action == "call_tool"
    assert plan.provider == "deterministic"
    assert plan.tool_call is not None
    assert plan.tool_call.name == "extract_ccf_c_journals"


class _FailingGateway(LLMGateway):
    async def plan(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("primary gateway should not be called")

    async def answer(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("primary gateway should not be called")


async def _collect_events(runtime: AgentRuntime) -> list:
    events = []
    request = ChatRequest(
        session_id="ccf-session",
        message="请找出这个文件里的所有 C 类期刊而不是会议",
    )
    async for event in runtime.stream(request):
        events.append(event)
    return events

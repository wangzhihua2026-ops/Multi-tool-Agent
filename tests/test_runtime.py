import asyncio

from app.agent.runtime import AgentRuntime
from app.agent.state import ChatRequest
from app.core.config import Settings
from app.core.llm_gateway import MockLLMGateway, PlanAction, PlanResult, ToolInvocation
from app.rag.embeddings import HashEmbeddingProvider
from app.rag.models import DocumentCreateRequest
from app.rag.retriever import KnowledgeRetriever
from app.rag.store import InMemoryKnowledgeStore
from app.rag.vector_store import InMemoryVectorStore
from app.services.document_service import DocumentService
from app.tools.builtins.calculator import register_calculator_tool
from app.tools.builtins.knowledge_base import register_knowledge_base_tool
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


def test_runtime_executes_tool_and_returns_grounded_response() -> None:
    store = InMemoryKnowledgeStore()
    vector_store = InMemoryVectorStore()
    embedding_provider = HashEmbeddingProvider()
    retriever = KnowledgeRetriever(store, vector_store, embedding_provider)
    service = DocumentService(
        store=store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        retriever=retriever,
    )
    service.create_document(
        DocumentCreateRequest(
            title="ops-guide",
            content="Check OPENAI_API_KEY before restarting the service.",
            metadata={},
        )
    )

    registry = ToolRegistry()
    register_calculator_tool(registry)
    register_knowledge_base_tool(registry, retriever)
    executor = ToolExecutor(registry)
    runtime = AgentRuntime(
        settings=Settings(),
        registry=registry,
        executor=executor,
        llm_gateway=MockLLMGateway(),
    )

    events = asyncio.run(_collect_events(runtime))
    event_types = [event.type for event in events]
    assert "planner.completed" in event_types
    assert "tool.completed" in event_types
    assert any("OPENAI_API_KEY" in event.data.get("content", "") for event in events if event.type == "assistant.message")


def test_runtime_handles_unknown_tool_from_planner() -> None:
    registry = ToolRegistry()
    register_calculator_tool(registry)
    executor = ToolExecutor(registry)
    runtime = AgentRuntime(
        settings=Settings(),
        registry=registry,
        executor=executor,
        llm_gateway=UnknownToolGateway(),
    )

    events = asyncio.run(_collect_events(runtime, message="Please do the impossible."))
    event_types = [event.type for event in events]

    assert "tool.failed" in event_types
    assert "run.completed" in event_types
    assert "run.failed" not in event_types
    assert any(
        "Unknown tool: missing_tool" in event.data.get("content", "")
        for event in events
        if event.type == "assistant.message"
    )


def test_runtime_can_execute_multiple_tool_steps_before_answering() -> None:
    store = InMemoryKnowledgeStore()
    vector_store = InMemoryVectorStore()
    embedding_provider = HashEmbeddingProvider()
    retriever = KnowledgeRetriever(store, vector_store, embedding_provider)
    service = DocumentService(
        store=store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        retriever=retriever,
    )
    service.create_document(
        DocumentCreateRequest(
            title="ops-guide",
            content="Check OPENAI_API_KEY before restarting the service.",
            metadata={},
        )
    )

    registry = ToolRegistry()
    register_calculator_tool(registry)
    register_knowledge_base_tool(registry, retriever)
    executor = ToolExecutor(registry)
    runtime = AgentRuntime(
        settings=Settings(max_tool_steps=3),
        registry=registry,
        executor=executor,
        llm_gateway=TwoStepGateway(),
    )

    events = asyncio.run(_collect_events(runtime, message="Calculate and then search."))
    completed_tools = [
        event.data.get("tool_name")
        for event in events
        if event.type == "tool.completed"
    ]

    assert completed_tools == ["calculator", "search_knowledge_base"]
    assert len([event for event in events if event.type == "planner.completed"]) == 3
    assert any(
        "calculator=4" in event.data.get("content", "") and "OPENAI_API_KEY" in event.data.get("content", "")
        for event in events
        if event.type == "assistant.message"
    )


async def _collect_events(
    runtime: AgentRuntime,
    message: str = "Please search the knowledge base for the first startup check.",
) -> list:
    events = []
    request = ChatRequest(session_id="runtime-session", message=message)
    async for event in runtime.stream(request):
        events.append(event)
    return events


class UnknownToolGateway(MockLLMGateway):
    async def plan(self, user_message, tools, history=None, tool_results=None, tool_errors=None) -> PlanResult:
        return PlanResult(
            action=PlanAction.CALL_TOOL,
            provider="test",
            tool_call=ToolInvocation(name="missing_tool", arguments={"query": user_message}),
        )


class TwoStepGateway(MockLLMGateway):
    async def plan(self, user_message, tools, history=None, tool_results=None, tool_errors=None) -> PlanResult:
        result_count = len(tool_results or [])
        if result_count == 0:
            return PlanResult(
                action=PlanAction.CALL_TOOL,
                provider="test",
                tool_call=ToolInvocation(name="calculator", arguments={"expression": "2 + 2"}),
            )
        if result_count == 1:
            return PlanResult(
                action=PlanAction.CALL_TOOL,
                provider="test",
                tool_call=ToolInvocation(
                    name="search_knowledge_base",
                    arguments={"query": "startup check", "top_k": 1},
                ),
            )
        return PlanResult(action=PlanAction.RESPOND, provider="test")

    async def answer(
        self,
        user_message,
        tool_result=None,
        tool_error=None,
        history=None,
        tool_results=None,
        tool_errors=None,
    ) -> str:
        results = tool_results or []
        calculator_result = results[0].content if results else ""
        knowledge_result = results[1].content if len(results) > 1 else ""
        return f"calculator={calculator_result}; knowledge={knowledge_result}"

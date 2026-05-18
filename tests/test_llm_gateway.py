import asyncio

from app.agent.state import ConversationMessage
from app.core.llm_gateway import MockLLMGateway, PlanAction
from app.tools.schemas import ToolDefinition, ToolExecutionResult


def test_mock_gateway_calls_knowledge_base_tool() -> None:
    gateway = MockLLMGateway()
    tools = [
        ToolDefinition(
            name="search_knowledge_base",
            description="Search internal documents.",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
    ]

    plan = asyncio.run(gateway.plan("Please search the knowledge base for deployment steps.", tools))
    assert plan.action == PlanAction.CALL_TOOL
    assert plan.tool_call is not None
    assert plan.tool_call.name == "search_knowledge_base"


def test_mock_gateway_formats_tool_output() -> None:
    gateway = MockLLMGateway()
    answer = asyncio.run(
        gateway.answer(
            user_message="What should I check first?",
            tool_result=ToolExecutionResult(
                tool_name="search_knowledge_base",
                content="Check OPENAI_API_KEY before restarting the service.",
            ),
        )
    )
    assert "OPENAI_API_KEY" in answer


def test_mock_gateway_routes_continue_to_next_extraction_page() -> None:
    gateway = MockLLMGateway()
    tools = [
        ToolDefinition(
            name="extract_document_items",
            description="Extract document items.",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
    ]
    history = [
        ConversationMessage(role="user", content="列出所有包含 Transformer 的条目"),
        ConversationMessage(
            role="assistant",
            content="结果已分页：本次 limit=2, offset=0。继续请求时把 offset 设为 2。",
        ),
    ]

    plan = asyncio.run(gateway.plan("继续", tools, history=history))

    assert plan.action == PlanAction.CALL_TOOL
    assert plan.tool_call is not None
    assert plan.tool_call.name == "extract_document_items"
    assert plan.tool_call.arguments == {"query": "列出所有包含 Transformer 的条目", "offset": 2}

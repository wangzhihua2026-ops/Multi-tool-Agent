import asyncio

from app.agent.execution import ExecutionCheckpoint, StepType
from app.agent.runtime import AgentRuntime
from app.core.config import Settings
from app.core.llm_gateway import MockLLMGateway, PlanAction, PlanResult, ToolInvocation
from app.tools.builtins.calculator import register_calculator_tool
from app.tools.builtins.send_email import register_send_email_tool
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


def build_runtime(gateway: MockLLMGateway, *, include_email: bool = False) -> AgentRuntime:
    registry = ToolRegistry()
    register_calculator_tool(registry)
    if include_email:
        register_send_email_tool(registry)
    return AgentRuntime(
        settings=Settings(max_tool_steps=3),
        registry=registry,
        executor=ToolExecutor(registry),
        llm_gateway=gateway,
    )


def test_advance_plans_then_executes_tool() -> None:
    async def scenario() -> None:
        runtime = build_runtime(MockLLMGateway())
        checkpoint = ExecutionCheckpoint(
            run_id="00000000-0000-0000-0000-000000000101",
            session_id="resume-session",
            user_message="2 + 2",
        )

        planned = await runtime.advance(checkpoint)
        executed = await runtime.advance(planned.checkpoint)

        assert planned.step_type is StepType.PLAN
        assert planned.completed is False
        assert executed.step_type is StepType.TOOL_CALL
        assert executed.checkpoint.next_step == 2
        assert executed.checkpoint.tool_results[0].content == "4"

    asyncio.run(scenario())


def test_approval_checkpoint_does_not_replan() -> None:
    async def scenario() -> None:
        gateway = CountingEmailGateway()
        runtime = build_runtime(gateway, include_email=True)
        checkpoint = ExecutionCheckpoint(
            run_id="00000000-0000-0000-0000-000000000102",
            session_id="resume-session",
            user_message="send an email",
        )

        planned = await runtime.advance(checkpoint)
        waiting = await runtime.advance(planned.checkpoint)
        waiting_again = await runtime.advance(waiting.checkpoint)

        assert waiting.waiting_for_approval is True
        assert waiting.checkpoint.pending_tool_name == "send_email"
        assert waiting_again.waiting_for_approval is True
        assert gateway.plan_calls == 1

    asyncio.run(scenario())


class CountingEmailGateway(MockLLMGateway):
    def __init__(self) -> None:
        self.plan_calls = 0

    async def plan(self, user_message, tools, history=None, tool_results=None, tool_errors=None):
        self.plan_calls += 1
        return PlanResult(
            action=PlanAction.CALL_TOOL,
            provider="test",
            tool_call=ToolInvocation(
                name="send_email",
                arguments={
                    "to": "reviewer@example.com",
                    "subject": "Review",
                    "body": "Please review.",
                },
            ),
        )

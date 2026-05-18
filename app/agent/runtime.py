from collections.abc import AsyncIterator

from app.agent.events import AgentEvent
from app.agent.policies.approval_policy import requires_human_approval
from app.agent.state import ChatRequest, ConversationMessage, RunState, RunStatus
from app.core.config import Settings
from app.core.llm_gateway import LLMGateway, PlanAction
from app.core.exceptions import ToolExecutionError
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition, ToolExecutionResult


class AgentRuntime:
    def __init__(
        self,
        settings: Settings,
        registry: ToolRegistry,
        executor: ToolExecutor,
        llm_gateway: LLMGateway,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.executor = executor
        self.llm_gateway = llm_gateway

    async def stream(
        self,
        request: ChatRequest,
        history: list[ConversationMessage] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        state = RunState.from_request(request)
        yield AgentEvent(type="run.started", run_id=state.run_id, data={"status": state.status})

        response_text = ""
        tool_results: list[ToolExecutionResult] = []
        tool_errors: list[str] = []
        tools = self.registry.list_tools()
        max_tool_steps = max(1, self.settings.max_tool_steps)

        for step_index in range(max_tool_steps):
            step_number = step_index + 1
            plan = await self.llm_gateway.plan(
                user_message=request.message,
                tools=tools,
                history=history,
                tool_results=tool_results,
                tool_errors=tool_errors,
            )
            yield AgentEvent(
                type="planner.completed",
                run_id=state.run_id,
                data={
                    "action": plan.action,
                    "provider": plan.provider,
                    "model": plan.model,
                    "tool_name": plan.tool_call.name if plan.tool_call else None,
                    "step": step_number,
                },
            )

            response_text = plan.answer or ""
            if plan.action != PlanAction.CALL_TOOL:
                break

            if plan.tool_call is None:
                tool_errors.append("Planner requested a tool call without a tool invocation.")
                break

            tool_definition, tool_error = self._resolve_tool_definition(plan.tool_call.name)
            if tool_definition is None:
                tool_errors.append(tool_error or f"Unknown tool: {plan.tool_call.name}")
                yield AgentEvent(
                    type="tool.failed",
                    run_id=state.run_id,
                    data={"tool_name": plan.tool_call.name, "error": tool_errors[-1], "step": step_number},
                )
                break

            if requires_human_approval(tool_definition):
                state.status = RunStatus.WAITING_APPROVAL
                yield AgentEvent(
                    type="approval.required",
                    run_id=state.run_id,
                    data={
                        "tool_name": plan.tool_call.name,
                        "arguments": plan.tool_call.arguments,
                        "risk_level": tool_definition.risk_level,
                        "step": step_number,
                    },
                )
                yield AgentEvent(
                    type="run.waiting_approval",
                    run_id=state.run_id,
                    data={"status": state.status},
                )
                return

            tool_result, tool_error, tool_events = await self._execute_tool(
                run_id=state.run_id,
                tool_name=plan.tool_call.name,
                arguments=plan.tool_call.arguments,
                step=step_number,
            )
            for event in tool_events:
                yield event

            if tool_result is not None:
                tool_results.append(tool_result)
                if tool_result.metadata.get("direct_answer") is True:
                    response_text = tool_result.content
                    break
            if tool_error is not None:
                tool_errors.append(tool_error)
                break
        else:
            tool_errors.append(f"Reached maximum tool steps ({max_tool_steps}) before the planner produced a final answer.")

        if not response_text:
            response_text = await self.llm_gateway.answer(
                user_message=request.message,
                history=history,
                tool_results=tool_results,
                tool_errors=tool_errors,
            )

        state.status = RunStatus.COMPLETED
        yield AgentEvent(
            type="assistant.message",
            run_id=state.run_id,
            data={"content": response_text},
        )
        yield AgentEvent(type="run.completed", run_id=state.run_id, data={"status": state.status})

    async def resume(
        self,
        run_id: str,
        user_message: str,
        tool_name: str,
        arguments: dict,
        approved: bool,
        history: list[ConversationMessage] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        yield AgentEvent(
            type="run.resumed",
            run_id=run_id,
            data={"status": RunStatus.RUNNING},
        )

        if not approved:
            response_text = await self.llm_gateway.answer(
                user_message=user_message,
                tool_error="The requested tool call was rejected during human approval.",
                history=history,
            )
            yield AgentEvent(
                type="assistant.message",
                run_id=run_id,
                data={"content": response_text},
            )
            yield AgentEvent(
                type="run.completed",
                run_id=run_id,
                data={"status": RunStatus.COMPLETED},
            )
            return

        tool_result, tool_error, tool_events = await self._execute_tool(
            run_id=run_id,
            tool_name=tool_name,
            arguments=arguments,
            step=1,
        )
        for event in tool_events:
            yield event

        response_text = await self.llm_gateway.answer(
            user_message=user_message,
            history=history,
            tool_results=[tool_result] if tool_result is not None else [],
            tool_errors=[tool_error] if tool_error is not None else [],
        )
        yield AgentEvent(
            type="assistant.message",
            run_id=run_id,
            data={"content": response_text},
        )
        yield AgentEvent(
            type="run.completed",
            run_id=run_id,
            data={"status": RunStatus.COMPLETED},
        )

    def _resolve_tool_definition(self, tool_name: str) -> tuple[ToolDefinition | None, str | None]:
        try:
            return self.registry.get_definition(tool_name), None
        except KeyError as exc:
            detail = exc.args[0] if exc.args else f"Unknown tool: {tool_name}"
            return None, detail

    async def _execute_tool(
        self,
        run_id: str,
        tool_name: str,
        arguments: dict,
        step: int | None = None,
    ) -> tuple[ToolExecutionResult | None, str | None, list[AgentEvent]]:
        events = [
            AgentEvent(
                type="tool.requested",
                run_id=run_id,
                data={"tool_name": tool_name, "arguments": arguments, "step": step},
            )
        ]
        try:
            tool_result = await self.executor.execute(
                name=tool_name,
                arguments=arguments,
            )
            events.append(
                AgentEvent(
                    type="tool.completed",
                    run_id=run_id,
                    data={"tool_name": tool_name, "step": step},
                )
            )
            return tool_result, None, events
        except ToolExecutionError as exc:
            tool_error = str(exc)
            events.append(
                AgentEvent(
                    type="tool.failed",
                    run_id=run_id,
                    data={"tool_name": exc.tool_name, "error": tool_error, "step": step},
                )
            )
            return None, tool_error, events

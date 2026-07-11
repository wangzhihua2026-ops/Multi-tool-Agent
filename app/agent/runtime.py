from collections.abc import AsyncIterator
from uuid import uuid4

from app.agent.events import AgentEvent
from app.agent.execution import (
    CheckpointAction,
    ExecutionCheckpoint,
    StepOutcome,
    StepType,
)
from app.agent.policies.approval_policy import requires_human_approval
from app.agent.state import ChatRequest, ConversationMessage, RunStatus
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
        checkpoint = ExecutionCheckpoint(
            run_id=str(uuid4()),
            session_id=request.session_id,
            user_message=request.message,
            history=history or [],
        )
        yield AgentEvent(
            type="run.started",
            run_id=checkpoint.run_id,
            data={"status": RunStatus.RUNNING},
        )

        while True:
            outcome = await self.advance(checkpoint)
            checkpoint = outcome.checkpoint
            for payload in outcome.events:
                yield AgentEvent(
                    type=str(payload["type"]),
                    run_id=checkpoint.run_id,
                    data=dict(payload.get("data", {})),
                )
            if outcome.waiting_for_approval or outcome.completed:
                return

    async def advance(self, checkpoint: ExecutionCheckpoint) -> StepOutcome:
        if checkpoint.pending_action is CheckpointAction.PLAN:
            return await self._advance_plan(checkpoint)
        if checkpoint.pending_action is CheckpointAction.TOOL:
            return await self._advance_tool(checkpoint)
        if checkpoint.pending_action is CheckpointAction.ANSWER:
            return await self._advance_answer(checkpoint)
        raise ValueError(f"Unknown pending action: {checkpoint.pending_action}")

    async def _advance_plan(self, checkpoint: ExecutionCheckpoint) -> StepOutcome:
        updated = checkpoint.model_copy(deep=True)
        plan = await self.llm_gateway.plan(
            user_message=updated.user_message,
            tools=self.registry.list_tools(),
            history=updated.history,
            tool_results=updated.tool_results,
            tool_errors=updated.tool_errors,
        )
        events = [
            self._event(
                "planner.completed",
                {
                    "action": plan.action,
                    "provider": plan.provider,
                    "model": plan.model,
                    "tool_name": plan.tool_call.name if plan.tool_call else None,
                    "step": updated.next_step,
                },
            )
        ]

        if plan.action is PlanAction.CALL_TOOL:
            if plan.tool_call is None:
                updated.tool_errors.append(
                    "Planner requested a tool call without a tool invocation."
                )
                updated.pending_action = CheckpointAction.ANSWER
            else:
                updated.pending_tool_name = plan.tool_call.name
                updated.pending_tool_arguments = plan.tool_call.arguments
                updated.pending_action = CheckpointAction.TOOL
        else:
            updated.draft_answer = plan.answer or ""
            updated.pending_action = CheckpointAction.ANSWER

        return StepOutcome(
            step_type=StepType.PLAN,
            checkpoint=updated,
            events=events,
        )

    async def _advance_tool(self, checkpoint: ExecutionCheckpoint) -> StepOutcome:
        updated = checkpoint.model_copy(deep=True)
        tool_name = updated.pending_tool_name
        if not tool_name:
            updated.tool_errors.append("No pending tool call was stored in the checkpoint.")
            updated.pending_action = CheckpointAction.ANSWER
            return StepOutcome(step_type=StepType.TOOL_CALL, checkpoint=updated)

        tool_definition, definition_error = self._resolve_tool_definition(tool_name)
        if tool_definition is None:
            error = definition_error or f"Unknown tool: {tool_name}"
            updated.tool_errors.append(error)
            updated.pending_action = CheckpointAction.ANSWER
            return StepOutcome(
                step_type=StepType.TOOL_CALL,
                checkpoint=updated,
                events=[
                    self._event(
                        "tool.failed",
                        {"tool_name": tool_name, "error": error, "step": updated.next_step},
                    )
                ],
            )

        if requires_human_approval(tool_definition) and updated.approval_decision is None:
            return StepOutcome(
                step_type=StepType.APPROVAL,
                checkpoint=updated,
                events=[
                    self._event(
                        "approval.required",
                        {
                            "tool_name": tool_name,
                            "arguments": updated.pending_tool_arguments,
                            "risk_level": tool_definition.risk_level,
                            "step": updated.next_step,
                        },
                    ),
                    self._event(
                        "run.waiting_approval",
                        {"status": RunStatus.WAITING_APPROVAL},
                    ),
                ],
                waiting_for_approval=True,
            )

        if updated.approval_decision is False:
            updated.tool_errors.append(
                "The requested tool call was rejected during human approval."
            )
            self._clear_pending_tool(updated)
            updated.pending_action = CheckpointAction.ANSWER
            return StepOutcome(step_type=StepType.APPROVAL, checkpoint=updated)

        tool_result, tool_error, tool_events = await self._execute_tool(
            run_id=updated.run_id,
            tool_name=tool_name,
            arguments=updated.pending_tool_arguments,
            step=updated.next_step,
        )
        self._clear_pending_tool(updated)

        if tool_result is not None:
            updated.tool_results.append(tool_result)
        if tool_error is not None:
            updated.tool_errors.append(tool_error)

        updated.next_step += 1
        if tool_result is not None and tool_result.metadata.get("direct_answer") is True:
            updated.draft_answer = tool_result.content
            updated.pending_action = CheckpointAction.ANSWER
        elif tool_error is not None:
            updated.pending_action = CheckpointAction.ANSWER
        elif updated.next_step > max(1, self.settings.max_tool_steps):
            updated.tool_errors.append(
                f"Reached maximum tool steps ({max(1, self.settings.max_tool_steps)}) "
                "before the planner produced a final answer."
            )
            updated.pending_action = CheckpointAction.ANSWER
        else:
            updated.pending_action = CheckpointAction.PLAN

        return StepOutcome(
            step_type=StepType.TOOL_CALL,
            checkpoint=updated,
            events=[
                {"type": event.type, "data": event.data}
                for event in tool_events
            ],
        )

    async def _advance_answer(self, checkpoint: ExecutionCheckpoint) -> StepOutcome:
        updated = checkpoint.model_copy(deep=True)
        response_text = updated.draft_answer
        if not response_text:
            response_text = await self.llm_gateway.answer(
                user_message=updated.user_message,
                history=updated.history,
                tool_results=updated.tool_results,
                tool_errors=updated.tool_errors,
            )
        return StepOutcome(
            step_type=StepType.ANSWER,
            checkpoint=updated,
            events=[
                self._event("assistant.message", {"content": response_text}),
                self._event("run.completed", {"status": RunStatus.COMPLETED}),
            ],
            final_response=response_text,
            completed=True,
        )

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

        checkpoint = ExecutionCheckpoint(
            run_id=run_id,
            session_id="approval-resume",
            user_message=user_message,
            history=history or [],
            pending_tool_name=tool_name,
            pending_tool_arguments=arguments,
            approval_decision=approved,
            pending_action=CheckpointAction.TOOL,
        )
        while True:
            outcome = await self.advance(checkpoint)
            checkpoint = outcome.checkpoint
            for payload in outcome.events:
                yield AgentEvent(
                    type=str(payload["type"]),
                    run_id=run_id,
                    data=dict(payload.get("data", {})),
                )
            if outcome.completed:
                return

    @staticmethod
    def _event(event_type: str, data: dict) -> dict:
        return {"type": event_type, "data": data}

    @staticmethod
    def _clear_pending_tool(checkpoint: ExecutionCheckpoint) -> None:
        checkpoint.pending_tool_name = None
        checkpoint.pending_tool_arguments = {}
        checkpoint.approval_decision = None

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

import json
import logging
import re
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.agent.state import ConversationMessage
from app.agent.nodes.plan import should_search_knowledge_base
from app.agent.prompts import build_answer_system_prompt, build_planning_system_prompt
from app.core.config import Settings
from app.tools.schemas import ToolDefinition, ToolExecutionResult

logger = logging.getLogger(__name__)


class PlanAction(StrEnum):
    RESPOND = "respond"
    CALL_TOOL = "call_tool"


class ToolInvocation(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class PlanResult(BaseModel):
    action: PlanAction
    provider: str
    model: str | None = None
    answer: str | None = None
    tool_call: ToolInvocation | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)


class LLMGateway(ABC):
    @abstractmethod
    async def plan(
        self,
        user_message: str,
        tools: list[ToolDefinition],
        history: list[ConversationMessage] | None = None,
        tool_results: list[ToolExecutionResult] | None = None,
        tool_errors: list[str] | None = None,
    ) -> PlanResult:
        raise NotImplementedError

    @abstractmethod
    async def answer(
        self,
        user_message: str,
        tool_result: ToolExecutionResult | None = None,
        tool_error: str | None = None,
        history: list[ConversationMessage] | None = None,
        tool_results: list[ToolExecutionResult] | None = None,
        tool_errors: list[str] | None = None,
    ) -> str:
        raise NotImplementedError


class MockLLMGateway(LLMGateway):
    email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    math_pattern = re.compile(r"^[\d\s\+\-\*\/\(\)\.]+$")

    async def plan(
        self,
        user_message: str,
        tools: list[ToolDefinition],
        history: list[ConversationMessage] | None = None,
        tool_results: list[ToolExecutionResult] | None = None,
        tool_errors: list[str] | None = None,
    ) -> PlanResult:
        tool_names = {tool.name for tool in tools}

        if tool_results or tool_errors:
            return PlanResult(
                action=PlanAction.RESPOND,
                provider="mock",
            )

        if "send_email" in tool_names:
            email_arguments = self._extract_email_request(user_message)
            if email_arguments is not None:
                return PlanResult(
                    action=PlanAction.CALL_TOOL,
                    provider="mock",
                    tool_call=ToolInvocation(
                        name="send_email",
                        arguments=email_arguments,
                    ),
                )

        if "calculator" in tool_names:
            expression = self._extract_expression(user_message)
            if expression:
                return PlanResult(
                    action=PlanAction.CALL_TOOL,
                    provider="mock",
                    tool_call=ToolInvocation(
                        name="calculator",
                        arguments={"expression": expression},
                    ),
                )

        extraction_plan = _build_deterministic_extraction_plan(
            user_message=user_message,
            tools=tools,
            provider="mock",
            history=history,
        )
        if extraction_plan is not None:
            return extraction_plan

        if "search_knowledge_base" in tool_names and should_search_knowledge_base(user_message):
            return PlanResult(
                action=PlanAction.CALL_TOOL,
                provider="mock",
                tool_call=ToolInvocation(
                    name="search_knowledge_base",
                    arguments={"query": user_message, "top_k": 3},
                ),
            )

        if history:
            recent_context = history[-1].content
            return PlanResult(
                action=PlanAction.RESPOND,
                provider="mock",
                answer=(
                    "Current runtime is using the mock LLM mode with recent session history.\n\n"
                    f"User request: {user_message}\n"
                    f"Recent context: {recent_context}"
                ),
            )

        return PlanResult(
            action=PlanAction.RESPOND,
            provider="mock",
            answer=(
                "Current runtime is using the mock LLM mode. "
                "Configure a real provider to enable open-ended reasoning and summarization."
            ),
        )

    async def answer(
        self,
        user_message: str,
        tool_result: ToolExecutionResult | None = None,
        tool_error: str | None = None,
        history: list[ConversationMessage] | None = None,
        tool_results: list[ToolExecutionResult] | None = None,
        tool_errors: list[str] | None = None,
    ) -> str:
        all_tool_errors = [item for item in (tool_errors or []) if item]
        if tool_error:
            all_tool_errors.append(tool_error)

        all_tool_results = list(tool_results or [])
        if tool_result is not None:
            all_tool_results.append(tool_result)

        if all_tool_errors:
            return (
                "The tool call did not complete successfully.\n\n"
                f"User request: {user_message}\n"
                f"Tool error: {all_tool_errors[-1]}"
            )

        if all_tool_results:
            tool_lines = [
                f"{index}. {result.tool_name}: {result.content}"
                for index, result in enumerate(all_tool_results, start=1)
            ]
            return (
                "This answer is grounded in tool output.\n\n"
                f"User request: {user_message}\n"
                f"Tool output:\n{chr(10).join(tool_lines)}"
            )

        if history:
            recent_context = history[-1].content
            return (
                "Current runtime is using the mock LLM mode, but recent session history was loaded.\n\n"
                f"User request: {user_message}\n"
                f"Recent context: {recent_context}"
            )

        return (
            "Current runtime is using the mock LLM mode.\n\n"
            f"User request: {user_message}\n\n"
            "No tool was triggered and no external model call was made."
        )

    def _extract_expression(self, user_message: str) -> str | None:
        candidate = user_message.strip()
        if self.math_pattern.fullmatch(candidate):
            return candidate

        inline_match = re.search(r"([\d\s\+\-\*\/\(\)\.]{3,})", candidate)
        if inline_match:
            expression = inline_match.group(1).strip()
            if self.math_pattern.fullmatch(expression):
                return expression

        return None

    def _extract_email_request(self, user_message: str) -> dict[str, str] | None:
        lowered = user_message.lower()
        if "email" not in lowered and "mail" not in lowered:
            return None

        recipient_match = self.email_pattern.search(user_message)
        if recipient_match is None:
            return None

        subject_match = re.search(r"subject\s*[:=]\s*(.+?)(?:\s+body\s*[:=]|$)", user_message, re.IGNORECASE)
        body_match = re.search(r"body\s*[:=]\s*(.+)$", user_message, re.IGNORECASE)

        subject = subject_match.group(1).strip() if subject_match else "Agent message"
        body = body_match.group(1).strip() if body_match else user_message.strip()

        return {
            "to": recipient_match.group(0),
            "subject": subject,
            "body": body,
        }


class OpenAICompatibleGateway(LLMGateway):
    def __init__(self, settings: Settings) -> None:
        if not settings.resolved_llm_api_key:
            raise ValueError("A resolved LLM API key is required for the OpenAI-compatible gateway.")
        self.settings = settings

    async def plan(
        self,
        user_message: str,
        tools: list[ToolDefinition],
        history: list[ConversationMessage] | None = None,
        tool_results: list[ToolExecutionResult] | None = None,
        tool_errors: list[str] | None = None,
    ) -> PlanResult:
        if not tool_results and not tool_errors:
            extraction_plan = _build_deterministic_extraction_plan(
                user_message=user_message,
                tools=tools,
                provider="openai",
                model=self.settings.default_model,
                history=history,
            )
            if extraction_plan is not None:
                return extraction_plan

        payload = {
            "model": self.settings.default_model,
            "messages": self._build_messages(
                system_prompt=build_planning_system_prompt(),
                history=history,
                trailing_user_content=self._build_planning_content(
                    user_message=user_message,
                    tool_results=tool_results,
                    tool_errors=tool_errors,
                ),
            ),
            "tools": [self._tool_to_chat_completion(tool) for tool in tools],
            "tool_choice": "auto",
            "temperature": 0,
        }
        data = await self._post_json("/chat/completions", payload)
        choice = data["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls") or []

        if tool_calls:
            first_call = tool_calls[0]
            function_block = first_call.get("function", {})
            return PlanResult(
                action=PlanAction.CALL_TOOL,
                provider="openai",
                model=data.get("model"),
                tool_call=ToolInvocation(
                    name=function_block.get("name", ""),
                    arguments=self._parse_tool_arguments(function_block.get("arguments", "{}")),
                ),
                raw_response={
                    "id": data.get("id"),
                    "finish_reason": choice.get("finish_reason"),
                },
            )

        return PlanResult(
            action=PlanAction.RESPOND,
            provider="openai",
            model=data.get("model"),
            answer=self._extract_message_content(message.get("content")),
            raw_response={
                "id": data.get("id"),
                "finish_reason": choice.get("finish_reason"),
            },
        )

    async def answer(
        self,
        user_message: str,
        tool_result: ToolExecutionResult | None = None,
        tool_error: str | None = None,
        history: list[ConversationMessage] | None = None,
        tool_results: list[ToolExecutionResult] | None = None,
        tool_errors: list[str] | None = None,
    ) -> str:
        evidence = self._format_tool_evidence(
            tool_result=tool_result,
            tool_error=tool_error,
            tool_results=tool_results,
            tool_errors=tool_errors,
        )

        payload = {
            "model": self.settings.default_model,
            "messages": self._build_messages(
                system_prompt=build_answer_system_prompt(),
                history=history,
                trailing_user_content=(
                    f"User request:\n{user_message}\n\n"
                    f"Available evidence:\n{evidence or 'No tool output was provided.'}"
                ),
            ),
            "temperature": 0.2,
        }
        data = await self._post_json("/chat/completions", payload)
        choice = data["choices"][0]
        message = choice["message"]
        content = self._extract_message_content(message.get("content"))
        if content:
            return content

        return "The model returned an empty response."

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        base_url = self.settings.llm_base_url.rstrip("/")
        headers = {
            "Authorization": f"Bearer {self.settings.resolved_llm_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
            response = await client.post(f"{base_url}{path}", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    def _tool_to_chat_completion(self, tool: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    def _parse_tool_arguments(self, raw_arguments: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            logger.warning("Failed to parse tool arguments from model response: %s", raw_arguments)
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _build_planning_content(
        self,
        user_message: str,
        tool_results: list[ToolExecutionResult] | None = None,
        tool_errors: list[str] | None = None,
    ) -> str:
        evidence = self._format_tool_evidence(
            tool_results=tool_results,
            tool_errors=tool_errors,
        )
        if not evidence:
            return user_message
        return (
            f"User request:\n{user_message}\n\n"
            f"Tool results so far:\n{evidence}\n\n"
            "Decide whether another tool call is necessary. If the evidence is sufficient, answer directly."
        )

    def _format_tool_evidence(
        self,
        tool_result: ToolExecutionResult | None = None,
        tool_error: str | None = None,
        tool_results: list[ToolExecutionResult] | None = None,
        tool_errors: list[str] | None = None,
    ) -> str:
        lines: list[str] = []
        all_tool_results = list(tool_results or [])
        if tool_result is not None:
            all_tool_results.append(tool_result)
        for index, result in enumerate(all_tool_results, start=1):
            lines.append(f"{index}. Tool {result.tool_name} returned:\n{result.content}")

        all_tool_errors = [item for item in (tool_errors or []) if item]
        if tool_error:
            all_tool_errors.append(tool_error)
        for index, error in enumerate(all_tool_errors, start=1):
            lines.append(f"Error {index}: {error}")

        return "\n\n".join(lines)

    def _build_messages(
        self,
        system_prompt: str,
        history: list[ConversationMessage] | None,
        trailing_user_content: str,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for item in history or []:
            if item.role not in {"user", "assistant"}:
                continue
            messages.append({"role": item.role, "content": item.content})
        messages.append({"role": "user", "content": trailing_user_content})
        return messages

    def _extract_message_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")).strip())
            return "\n".join(part for part in text_parts if part)
        return ""


class FallbackLLMGateway(LLMGateway):
    def __init__(self, primary: LLMGateway, fallback: LLMGateway) -> None:
        self.primary = primary
        self.fallback = fallback

    async def plan(
        self,
        user_message: str,
        tools: list[ToolDefinition],
        history: list[ConversationMessage] | None = None,
        tool_results: list[ToolExecutionResult] | None = None,
        tool_errors: list[str] | None = None,
    ) -> PlanResult:
        if not tool_results and not tool_errors:
            extraction_plan = _build_deterministic_extraction_plan(
                user_message=user_message,
                tools=tools,
                provider="deterministic",
                history=history,
            )
            if extraction_plan is not None:
                return extraction_plan

        try:
            return await self.primary.plan(
                user_message=user_message,
                tools=tools,
                history=history,
                tool_results=tool_results,
                tool_errors=tool_errors,
            )
        except Exception as exc:  # pragma: no cover - network/provider fallback
            logger.warning("Primary LLM planning failed, falling back to mock gateway: %s", exc)
            return await self.fallback.plan(
                user_message=user_message,
                tools=tools,
                history=history,
                tool_results=tool_results,
                tool_errors=tool_errors,
            )

    async def answer(
        self,
        user_message: str,
        tool_result: ToolExecutionResult | None = None,
        tool_error: str | None = None,
        history: list[ConversationMessage] | None = None,
        tool_results: list[ToolExecutionResult] | None = None,
        tool_errors: list[str] | None = None,
    ) -> str:
        try:
            return await self.primary.answer(
                user_message=user_message,
                tool_result=tool_result,
                tool_error=tool_error,
                history=history,
                tool_results=tool_results,
                tool_errors=tool_errors,
            )
        except Exception as exc:  # pragma: no cover - network/provider fallback
            logger.warning("Primary LLM answer generation failed, falling back to mock gateway: %s", exc)
            return await self.fallback.answer(
                user_message=user_message,
                tool_result=tool_result,
                tool_error=tool_error,
                history=history,
                tool_results=tool_results,
                tool_errors=tool_errors,
            )


def build_llm_gateway(settings: Settings) -> LLMGateway:
    provider = settings.llm_provider.strip().lower()
    fallback = MockLLMGateway()

    if provider == "mock" or not settings.resolved_llm_api_key:
        return fallback

    if provider == "openai":
        return FallbackLLMGateway(
            primary=OpenAICompatibleGateway(settings),
            fallback=fallback,
        )

    logger.warning("Unknown LLM provider '%s', falling back to mock gateway.", settings.llm_provider)
    return fallback


def _build_deterministic_extraction_plan(
    *,
    user_message: str,
    tools: list[ToolDefinition],
    provider: str,
    model: str | None = None,
    history: list[ConversationMessage] | None = None,
) -> PlanResult | None:
    tool_names = {tool.name for tool in tools}
    continuation = _build_extraction_continuation(user_message=user_message, history=history)
    if "extract_document_items" in tool_names and continuation is not None:
        return PlanResult(
            action=PlanAction.CALL_TOOL,
            provider=provider,
            model=model,
            tool_call=ToolInvocation(
                name="extract_document_items",
                arguments=continuation,
            ),
        )
    if "extract_document_items" in tool_names and _should_extract_document_items(user_message):
        return PlanResult(
            action=PlanAction.CALL_TOOL,
            provider=provider,
            model=model,
            tool_call=ToolInvocation(
                name="extract_document_items",
                arguments={"query": user_message},
            ),
        )
    if "extract_ccf_c_journals" in tool_names and _should_extract_ccf_c_journals(user_message):
        return PlanResult(
            action=PlanAction.CALL_TOOL,
            provider=provider,
            model=model,
            tool_call=ToolInvocation(name="extract_ccf_c_journals", arguments={}),
        )
    return None


def _build_extraction_continuation(
    *,
    user_message: str,
    history: list[ConversationMessage] | None,
) -> dict[str, Any] | None:
    if not history or not _is_continue_request(user_message):
        return None

    offset: int | None = None
    for item in reversed(history):
        if item.role != "assistant":
            continue
        offset = _extract_next_offset(item.content)
        if offset is not None:
            break
    if offset is None:
        return None

    for item in reversed(history):
        if item.role != "user":
            continue
        if _should_extract_document_items(item.content):
            return {"query": item.content, "offset": offset}
    return None


def _is_continue_request(user_message: str) -> bool:
    normalized = re.sub(r"\s+", "", user_message.lower())
    return normalized in {
        "继续",
        "继续输出",
        "下一页",
        "下页",
        "继续下一页",
        "next",
        "nextpage",
        "continue",
    }


def _extract_next_offset(content: str) -> int | None:
    patterns = [
        r"offset\s*(?:设为|=|:)\s*(\d+)",
        r"offset\s+(\d+)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, content, flags=re.IGNORECASE))
        if matches:
            return int(matches[-1].group(1))
    return None


def _should_extract_document_items(user_message: str) -> bool:
    lowered = user_message.lower()
    compact = re.sub(r"\s+", "", lowered)
    has_exhaustive_intent = any(
        term in lowered
        for term in [
            "所有",
            "全部",
            "完整",
            "全量",
            "列出",
            "找出",
            "提取",
            "输出",
            "all",
            "complete",
            "every",
            "list",
            "extract",
        ]
    )
    has_document_context = any(term in lowered for term in ["文件", "文档", "上传", "pdf", "docx", "document", "file"])
    has_filter = any(
        term in lowered
        for term in [
            "不要",
            "不是",
            "排除",
            "只要",
            "包含",
            "期刊",
            "会议",
            "journal",
            "conference",
            "exclude",
            "not ",
        ]
    )
    has_class_filter = bool(re.search(r"[abc](?:类|-class|class)", compact, re.IGNORECASE))
    return has_exhaustive_intent and (has_document_context or has_filter or has_class_filter)


def _should_extract_ccf_c_journals(user_message: str) -> bool:
    lowered = user_message.lower()
    compact = re.sub(r"\s+", "", lowered)
    has_c_class = "c类" in compact or "c-class" in compact or "cclass" in compact
    has_journal = "期刊" in user_message or "journal" in lowered
    has_exhaustive_intent = (
        "会议" in user_message
        or "conference" in lowered
        or "所有" in user_message
        or "全部" in user_message
        or "文件" in user_message
        or "文档" in user_message
    )
    has_ccf_context = "ccf" in lowered or "中国计算机学会" in user_message
    return (
        has_c_class
        and has_journal
        and has_exhaustive_intent
        and (has_ccf_context or "文件" in user_message or "文档" in user_message)
    )

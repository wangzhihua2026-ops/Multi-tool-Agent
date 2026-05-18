from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.agent.state import ChatRequest
from app.api.dependencies import get_message_repository, get_run_repository, get_runtime, settings_dependency
from app.api.sse import encode_sse
from app.services.chat_service import ChatService
from app.core.config import Settings
from app.persistence.message_repository import SqliteMessageRepository
from app.persistence.run_repository import SqliteRunRepository

router = APIRouter(tags=["chat"])


def get_chat_service(
    repository: SqliteRunRepository = Depends(get_run_repository),
    message_repository: SqliteMessageRepository = Depends(get_message_repository),
    settings: Settings = Depends(settings_dependency),
) -> ChatService:
    return ChatService(
        runtime=get_runtime(),
        repository=repository,
        message_repository=message_repository,
        history_limit=settings.session_history_limit,
    )


@router.post("/chat/stream")
async def stream_chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        async for event in service.stream_chat(request):
            yield encode_sse(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

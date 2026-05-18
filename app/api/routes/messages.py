from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_message_service
from app.persistence.models import SessionMessageRecord
from app.services.message_service import MessageService

router = APIRouter(tags=["messages"])


@router.get("/sessions/{session_id}/messages", response_model=list[SessionMessageRecord])
async def list_session_messages(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    service: MessageService = Depends(get_message_service),
) -> list[SessionMessageRecord]:
    return service.list_session_messages(session_id=session_id, limit=limit)

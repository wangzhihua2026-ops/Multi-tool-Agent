from app.persistence.message_repository import SqliteMessageRepository
from app.persistence.models import SessionMessageRecord


class MessageService:
    def __init__(self, repository: SqliteMessageRepository) -> None:
        self.repository = repository

    def list_session_messages(self, session_id: str, limit: int = 50) -> list[SessionMessageRecord]:
        return self.repository.list_messages(session_id=session_id, limit=limit)

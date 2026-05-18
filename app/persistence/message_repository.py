import json
import sqlite3
from pathlib import Path
from threading import RLock
from uuid import uuid4

from app.persistence.models import SessionMessageRecord


class SqliteMessageRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self._lock = RLock()
        self._ensure_parent_directory()
        self._ensure_schema()

    def add_message(
        self,
        session_id: str,
        run_id: str | None,
        role: str,
        content: str,
        created_at: str,
        metadata: dict | None = None,
    ) -> str:
        message_id = str(uuid4())
        serialized_metadata = json.dumps(metadata or {}, ensure_ascii=False, default=str)

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO session_messages (
                    message_id, session_id, run_id, role, content, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (message_id, session_id, run_id, role, content, created_at, serialized_metadata),
            )
            connection.commit()

        return message_id

    def list_messages(self, session_id: str, limit: int = 50) -> list[SessionMessageRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT message_id, session_id, run_id, role, content, created_at, metadata_json
                FROM session_messages
                WHERE session_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        rows = list(reversed(rows))
        return [self._row_to_message(row) for row in rows]

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM session_messages")
            connection.commit()

    def _row_to_message(self, row: sqlite3.Row) -> SessionMessageRecord:
        return SessionMessageRecord(
            message_id=row["message_id"],
            session_id=row["session_id"],
            run_id=row["run_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_parent_directory(self) -> None:
        if self.db_path != Path(":memory:"):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    run_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_messages_session_created
                ON session_messages(session_id, created_at)
                """
            )
            connection.commit()

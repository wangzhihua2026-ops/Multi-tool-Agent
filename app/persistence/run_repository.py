import json
import sqlite3
from pathlib import Path
from threading import RLock

from app.agent.events import AgentEvent
from app.core.exceptions import RunNotFoundError
from app.persistence.models import RunDetail, RunEventRecord, RunSummary


class SqliteRunRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self._lock = RLock()
        self._ensure_parent_directory()
        self._ensure_schema()

    def create_run(self, run_id: str, session_id: str, user_message: str, created_at: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO runs (
                    run_id, session_id, user_message, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, session_id, user_message, "running", created_at, created_at),
            )
            connection.commit()

    def append_event(self, event: AgentEvent, sequence: int) -> None:
        created_at = event.created_at.isoformat()
        serialized_data = json.dumps(event.data, ensure_ascii=False, default=str)

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_events (
                    run_id, sequence, event_type, created_at, data_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (event.run_id, sequence, event.type, created_at, serialized_data),
            )
            self._apply_run_updates(connection, event)
            connection.commit()

    def list_runs(self, limit: int = 50) -> list[RunSummary]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, session_id, user_message, status, created_at, updated_at, provider, model,
                       approval_status, pending_tool_name
                FROM runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [self._row_to_summary(row) for row in rows]

    def list_waiting_approval_runs(self) -> list[RunSummary]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, session_id, user_message, status, created_at, updated_at, provider, model,
                       approval_status, pending_tool_name
                FROM runs
                WHERE status = 'waiting_approval'
                ORDER BY created_at DESC
                """
            ).fetchall()

        return [self._row_to_summary(row) for row in rows]

    def get_run(self, run_id: str) -> RunDetail:
        with self._lock, self._connect() as connection:
            run_row = connection.execute(
                """
                SELECT run_id, session_id, user_message, status, created_at, updated_at, provider, model,
                       approval_status, pending_tool_name, pending_tool_arguments_json, final_response
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if run_row is None:
                raise RunNotFoundError(run_id)

            event_rows = connection.execute(
                """
                SELECT sequence, event_type, created_at, data_json
                FROM run_events
                WHERE run_id = ?
                ORDER BY sequence ASC
                """,
                (run_id,),
            ).fetchall()

        summary = self._row_to_summary(run_row)
        return RunDetail(
            **summary.model_dump(),
            final_response=run_row["final_response"],
            pending_tool_arguments=json.loads(run_row["pending_tool_arguments_json"] or "{}"),
            events=[self._row_to_event(row) for row in event_rows],
        )

    def get_next_sequence(self, run_id: str) -> int:
        with self._lock, self._connect() as connection:
            run_row = connection.execute(
                "SELECT run_id FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run_row is None:
                raise RunNotFoundError(run_id)

            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), -1) AS max_sequence FROM run_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()

        return int(row["max_sequence"]) + 1

    def claim_pending_approval(self, run_id: str, approval_status: str, updated_at: str) -> bool:
        with self._lock, self._connect() as connection:
            run_row = connection.execute(
                "SELECT run_id FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run_row is None:
                raise RunNotFoundError(run_id)

            cursor = connection.execute(
                """
                UPDATE runs
                SET approval_status = ?, status = ?, updated_at = ?
                WHERE run_id = ?
                  AND status = 'waiting_approval'
                  AND approval_status = 'pending'
                  AND pending_tool_name IS NOT NULL
                """,
                (approval_status, "running", updated_at, run_id),
            )
            connection.commit()
            return cursor.rowcount == 1

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM run_events")
            connection.execute("DELETE FROM runs")
            connection.commit()

    def _apply_run_updates(self, connection: sqlite3.Connection, event: AgentEvent) -> None:
        updates = ["updated_at = ?"]
        params: list[object] = [event.created_at.isoformat()]

        if event.type == "planner.completed":
            updates.extend(["provider = ?", "model = ?"])
            params.extend([event.data.get("provider"), event.data.get("model")])
        elif event.type == "assistant.message":
            updates.append("final_response = ?")
            params.append(event.data.get("content"))
        elif event.type == "approval.required":
            updates.extend(
                [
                    "status = ?",
                    "approval_status = ?",
                    "pending_tool_name = ?",
                    "pending_tool_arguments_json = ?",
                ]
            )
            params.extend(
                [
                    "waiting_approval",
                    "pending",
                    event.data.get("tool_name"),
                    json.dumps(event.data.get("arguments", {}), ensure_ascii=False, default=str),
                ]
            )
        elif event.type == "run.waiting_approval":
            updates.append("status = ?")
            params.append(str(event.data.get("status", "waiting_approval")))
        elif event.type == "approval.approved":
            updates.extend(["approval_status = ?", "status = ?"])
            params.extend(["approved", "running"])
        elif event.type == "approval.rejected":
            updates.append("approval_status = ?")
            params.append("rejected")
        elif event.type == "run.resumed":
            updates.append("status = ?")
            params.append("running")
        elif event.type == "run.completed":
            updates.extend(
                [
                    "status = ?",
                    "pending_tool_name = NULL",
                    "pending_tool_arguments_json = NULL",
                ]
            )
            params.append(str(event.data.get("status", "completed")))
        elif event.type == "run.failed":
            updates.extend(
                [
                    "status = ?",
                    "pending_tool_name = NULL",
                    "pending_tool_arguments_json = NULL",
                ]
            )
            params.append("failed")

        query = f"UPDATE runs SET {', '.join(updates)} WHERE run_id = ?"
        params.append(event.run_id)
        connection.execute(query, params)

    def _row_to_summary(self, row: sqlite3.Row) -> RunSummary:
        return RunSummary(
            run_id=row["run_id"],
            session_id=row["session_id"],
            user_message=row["user_message"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            provider=row["provider"],
            model=row["model"],
            approval_status=row["approval_status"],
            pending_tool_name=row["pending_tool_name"],
        )

    def _row_to_event(self, row: sqlite3.Row) -> RunEventRecord:
        return RunEventRecord(
            sequence=row["sequence"],
            event_type=row["event_type"],
            created_at=row["created_at"],
            data=json.loads(row["data_json"] or "{}"),
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
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    approval_status TEXT,
                    pending_tool_name TEXT,
                    pending_tool_arguments_json TEXT,
                    final_response TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_run_events_run_sequence
                ON run_events(run_id, sequence)
                """
            )
            self._ensure_run_columns(connection)
            connection.commit()

    def _ensure_run_columns(self, connection: sqlite3.Connection) -> None:
        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }
        required_columns = {
            "approval_status": "TEXT",
            "pending_tool_name": "TEXT",
            "pending_tool_arguments_json": "TEXT",
        }

        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                connection.execute(f"ALTER TABLE runs ADD COLUMN {column_name} {column_type}")

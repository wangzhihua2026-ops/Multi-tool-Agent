import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from pydantic import BaseModel

from app.rag.models import DocumentReindexSummary
from app.services.document_service import DocumentService


class ReindexJobRecord(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    clear_vector_store: bool
    summary: DocumentReindexSummary | None = None
    error: str | None = None


class ReindexJobService:
    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = Path(db_path)
        self._lock = RLock()
        self._memory_connection: sqlite3.Connection | None = None
        if self.db_path == Path(":memory:"):
            self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_connection.row_factory = sqlite3.Row
        self._ensure_parent_directory()
        self._ensure_schema()

    def create_job(self, clear_vector_store: bool) -> ReindexJobRecord:
        now = datetime.now(timezone.utc)
        job = ReindexJobRecord(
            job_id=str(uuid4()),
            status="queued",
            created_at=now,
            updated_at=now,
            clear_vector_store=clear_vector_store,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO reindex_jobs (
                    job_id, status, created_at, updated_at, clear_vector_store, summary_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.status,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                    int(job.clear_vector_store),
                    None,
                    None,
                ),
            )
            connection.commit()
        return job

    def get_job(self, job_id: str) -> ReindexJobRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, status, created_at, updated_at, clear_vector_store, summary_json, error
                FROM reindex_jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        return self._row_to_job(row) if row is not None else None

    def run_job(self, job_id: str, document_service: DocumentService) -> None:
        self._update_job(job_id, status="running")
        try:
            job = self.get_job(job_id)
            if job is None:
                return
            summary = document_service.reindex_documents(clear_vector_store=job.clear_vector_store)
            self._update_job(job_id, status="completed", summary=summary)
        except Exception as exc:
            self._update_job(job_id, status="failed", error=str(exc))

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM reindex_jobs")
            connection.commit()

    def _update_job(
        self,
        job_id: str,
        status: str,
        summary: DocumentReindexSummary | None = None,
        error: str | None = None,
    ) -> None:
        updated_at = datetime.now(timezone.utc).isoformat()
        summary_json = (
            json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, default=str)
            if summary is not None
            else None
        )
        with self._lock, self._connect() as connection:
            if summary is None:
                connection.execute(
                    """
                    UPDATE reindex_jobs
                    SET status = ?, updated_at = ?, error = ?
                    WHERE job_id = ?
                    """,
                    (status, updated_at, error, job_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE reindex_jobs
                    SET status = ?, updated_at = ?, summary_json = ?, error = ?
                    WHERE job_id = ?
                    """,
                    (status, updated_at, summary_json, error, job_id),
                )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        if self._memory_connection is not None:
            return self._memory_connection
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
                CREATE TABLE IF NOT EXISTS reindex_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    clear_vector_store INTEGER NOT NULL,
                    summary_json TEXT,
                    error TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_reindex_jobs_created_at
                ON reindex_jobs(created_at)
                """
            )
            connection.commit()

    def _row_to_job(self, row: sqlite3.Row) -> ReindexJobRecord:
        summary = None
        if row["summary_json"]:
            summary = DocumentReindexSummary.model_validate(json.loads(row["summary_json"]))
        return ReindexJobRecord(
            job_id=row["job_id"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            clear_vector_store=bool(row["clear_vector_store"]),
            summary=summary,
            error=row["error"],
        )

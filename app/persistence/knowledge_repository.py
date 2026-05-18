import json
import logging
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any

from app.core.config import Settings
from app.core.exceptions import DocumentNotFoundError
from app.rag.models import ChunkRecord, DocumentRecord, DocumentSummary
from app.rag.store import InMemoryKnowledgeStore, KnowledgeStore

logger = logging.getLogger(__name__)


def _load_psycopg() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Postgres knowledge store requires the optional dependency "
            "'psycopg'. Install it with `pip install -e .[postgres]`."
        ) from exc

    return psycopg, dict_row


class SqliteKnowledgeStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self._lock = RLock()
        self._ensure_parent_directory()
        self._ensure_schema()

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM knowledge_chunks")
            connection.execute("DELETE FROM knowledge_documents")
            connection.commit()

    def add_document(self, document: DocumentRecord, chunks: list[ChunkRecord]) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO knowledge_documents (
                    document_id, title, content, metadata_json, index_status, index_error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.document_id,
                    document.title,
                    document.content,
                    json.dumps(document.metadata, ensure_ascii=False, default=str),
                    document.index_status,
                    document.index_error,
                    document.created_at.isoformat(),
                ),
            )
            connection.execute(
                "DELETE FROM knowledge_chunks WHERE document_id = ?",
                (document.document_id,),
            )
            connection.executemany(
                """
                INSERT INTO knowledge_chunks (
                    chunk_id, document_id, document_title, chunk_index, content, tokens_json, embedding_provider
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.document_title,
                        chunk.index,
                        chunk.content,
                        json.dumps(chunk.tokens, ensure_ascii=False, default=str),
                        chunk.embedding_provider,
                    )
                    for chunk in chunks
                ],
            )
            connection.commit()

    def list_documents(self) -> list[DocumentSummary]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT d.document_id, d.title, d.metadata_json, d.index_status, d.index_error, d.created_at,
                       COUNT(c.chunk_id) AS chunk_count
                FROM knowledge_documents d
                LEFT JOIN knowledge_chunks c ON c.document_id = d.document_id
                GROUP BY d.document_id, d.title, d.metadata_json, d.index_status, d.index_error, d.created_at
                ORDER BY d.created_at DESC, d.rowid DESC
                """
            ).fetchall()

        return [self._row_to_summary(row) for row in rows]

    def get_document(self, document_id: str) -> DocumentRecord:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT document_id, title, content, metadata_json, index_status, index_error, created_at
                FROM knowledge_documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

        if row is None:
            raise DocumentNotFoundError(document_id)

        return self._row_to_document(row)

    def get_document_chunks(self, document_id: str) -> list[ChunkRecord]:
        with self._lock, self._connect() as connection:
            document_exists = connection.execute(
                "SELECT document_id FROM knowledge_documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if document_exists is None:
                raise DocumentNotFoundError(document_id)

            rows = connection.execute(
                """
                SELECT chunk_id, document_id, document_title, chunk_index, content, tokens_json, embedding_provider
                FROM knowledge_chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC, rowid ASC
                """,
                (document_id,),
            ).fetchall()

        return [self._row_to_chunk(row) for row in rows]

    def get_chunks(self) -> list[ChunkRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, document_id, document_title, chunk_index, content, tokens_json, embedding_provider
                FROM knowledge_chunks
                ORDER BY document_id ASC, chunk_index ASC, rowid ASC
                """
            ).fetchall()

        return [self._row_to_chunk(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT chunk_id, document_id, document_title, chunk_index, content, tokens_json, embedding_provider
                FROM knowledge_chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_chunk(row)

    def set_chunk_embedding_provider(self, chunk_id: str, embedding_provider: str | None) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE knowledge_chunks
                SET embedding_provider = ?
                WHERE chunk_id = ?
                """,
                (embedding_provider, chunk_id),
            )
            connection.commit()

    def set_document_index_status(
        self,
        document_id: str,
        index_status: str,
        index_error: str | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE knowledge_documents
                SET index_status = ?, index_error = ?
                WHERE document_id = ?
                """,
                (index_status, index_error, document_id),
            )
            connection.commit()

    def _row_to_summary(self, row: sqlite3.Row) -> DocumentSummary:
        return DocumentSummary(
            document_id=row["document_id"],
            title=row["title"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            chunk_count=int(row["chunk_count"] or 0),
            index_status=row["index_status"] or "ready",
            index_error=row["index_error"],
            created_at=row["created_at"],
        )

    def _row_to_document(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            title=row["title"],
            content=row["content"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            index_status=row["index_status"] or "ready",
            index_error=row["index_error"],
            created_at=row["created_at"],
        )

    def _row_to_chunk(self, row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            document_title=row["document_title"],
            index=int(row["chunk_index"]),
            content=row["content"],
            tokens=json.loads(row["tokens_json"] or "[]"),
            embedding_provider=row["embedding_provider"],
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _ensure_parent_directory(self) -> None:
        if self.db_path != Path(":memory:"):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    index_status TEXT NOT NULL DEFAULT 'ready',
                    index_error TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    document_title TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    tokens_json TEXT NOT NULL,
                    embedding_provider TEXT,
                    FOREIGN KEY(document_id) REFERENCES knowledge_documents(document_id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document_index
                ON knowledge_chunks(document_id, chunk_index)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_documents_created_at
                ON knowledge_documents(created_at)
                """
            )
            self._ensure_document_columns(connection)
            connection.commit()

    def _ensure_document_columns(self, connection: sqlite3.Connection) -> None:
        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(knowledge_documents)").fetchall()
        }
        required_columns = {
            "index_status": "TEXT NOT NULL DEFAULT 'ready'",
            "index_error": "TEXT",
        }

        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                connection.execute(f"ALTER TABLE knowledge_documents ADD COLUMN {column_name} {column_type}")


class PostgresKnowledgeStore:
    def __init__(self, database_url: str) -> None:
        if not database_url.strip():
            raise ValueError("KNOWLEDGE_STORE_DATABASE_URL must be set when using postgres.")

        self.database_url = database_url
        self._lock = RLock()
        self._ensure_schema()

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM knowledge_chunks")
                cursor.execute("DELETE FROM knowledge_documents")
            connection.commit()

    def add_document(self, document: DocumentRecord, chunks: list[ChunkRecord]) -> None:
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO knowledge_documents (
                        document_id, title, content, metadata_json, index_status, index_error, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        metadata_json = EXCLUDED.metadata_json,
                        index_status = EXCLUDED.index_status,
                        index_error = EXCLUDED.index_error,
                        created_at = EXCLUDED.created_at
                    """,
                    (
                        document.document_id,
                        document.title,
                        document.content,
                        json.dumps(document.metadata, ensure_ascii=False, default=str),
                        document.index_status,
                        document.index_error,
                        document.created_at,
                    ),
                )
                cursor.execute(
                    "DELETE FROM knowledge_chunks WHERE document_id = %s",
                    (document.document_id,),
                )
                if chunks:
                    cursor.executemany(
                        """
                        INSERT INTO knowledge_chunks (
                            chunk_id, document_id, document_title, chunk_index, content, tokens_json, embedding_provider
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                chunk.chunk_id,
                                chunk.document_id,
                                chunk.document_title,
                                chunk.index,
                                chunk.content,
                                json.dumps(chunk.tokens, ensure_ascii=False, default=str),
                                chunk.embedding_provider,
                            )
                            for chunk in chunks
                        ],
                    )
            connection.commit()

    def list_documents(self) -> list[DocumentSummary]:
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT d.document_id, d.title, d.metadata_json, d.index_status, d.index_error, d.created_at,
                           COUNT(c.chunk_id) AS chunk_count
                    FROM knowledge_documents d
                    LEFT JOIN knowledge_chunks c ON c.document_id = d.document_id
                    GROUP BY d.document_id, d.title, d.metadata_json, d.index_status, d.index_error, d.created_at
                    ORDER BY d.created_at DESC
                    """
                )
                rows = cursor.fetchall()

        return [self._row_to_summary(row) for row in rows]

    def get_document(self, document_id: str) -> DocumentRecord:
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT document_id, title, content, metadata_json, index_status, index_error, created_at
                    FROM knowledge_documents
                    WHERE document_id = %s
                    """,
                    (document_id,),
                )
                row = cursor.fetchone()

        if row is None:
            raise DocumentNotFoundError(document_id)

        return self._row_to_document(row)

    def get_document_chunks(self, document_id: str) -> list[ChunkRecord]:
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT document_id FROM knowledge_documents WHERE document_id = %s",
                    (document_id,),
                )
                document_exists = cursor.fetchone()
                if document_exists is None:
                    raise DocumentNotFoundError(document_id)

                cursor.execute(
                    """
                    SELECT chunk_id, document_id, document_title, chunk_index, content, tokens_json, embedding_provider
                    FROM knowledge_chunks
                    WHERE document_id = %s
                    ORDER BY chunk_index ASC
                    """,
                    (document_id,),
                )
                rows = cursor.fetchall()

        return [self._row_to_chunk(row) for row in rows]

    def get_chunks(self) -> list[ChunkRecord]:
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT chunk_id, document_id, document_title, chunk_index, content, tokens_json, embedding_provider
                    FROM knowledge_chunks
                    ORDER BY document_id ASC, chunk_index ASC
                    """
                )
                rows = cursor.fetchall()

        return [self._row_to_chunk(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT chunk_id, document_id, document_title, chunk_index, content, tokens_json, embedding_provider
                    FROM knowledge_chunks
                    WHERE chunk_id = %s
                    """,
                    (chunk_id,),
                )
                row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_chunk(row)

    def set_chunk_embedding_provider(self, chunk_id: str, embedding_provider: str | None) -> None:
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE knowledge_chunks
                    SET embedding_provider = %s
                    WHERE chunk_id = %s
                    """,
                    (embedding_provider, chunk_id),
                )
            connection.commit()

    def set_document_index_status(
        self,
        document_id: str,
        index_status: str,
        index_error: str | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE knowledge_documents
                    SET index_status = %s, index_error = %s
                    WHERE document_id = %s
                    """,
                    (index_status, index_error, document_id),
                )
            connection.commit()

    def _row_to_summary(self, row: dict[str, Any]) -> DocumentSummary:
        return DocumentSummary(
            document_id=row["document_id"],
            title=row["title"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            chunk_count=int(row["chunk_count"] or 0),
            index_status=row["index_status"] or "ready",
            index_error=row["index_error"],
            created_at=row["created_at"],
        )

    def _row_to_document(self, row: dict[str, Any]) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            title=row["title"],
            content=row["content"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            index_status=row["index_status"] or "ready",
            index_error=row["index_error"],
            created_at=row["created_at"],
        )

    def _row_to_chunk(self, row: dict[str, Any]) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            document_title=row["document_title"],
            index=int(row["chunk_index"]),
            content=row["content"],
            tokens=json.loads(row["tokens_json"] or "[]"),
            embedding_provider=row["embedding_provider"],
        )

    def _connect(self):
        psycopg, dict_row = _load_psycopg()
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS knowledge_documents (
                        document_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        index_status TEXT NOT NULL DEFAULT 'ready',
                        index_error TEXT,
                        created_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS knowledge_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL REFERENCES knowledge_documents(document_id) ON DELETE CASCADE,
                        document_title TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        tokens_json TEXT NOT NULL,
                        embedding_provider TEXT
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document_index
                    ON knowledge_chunks(document_id, chunk_index)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_knowledge_documents_created_at
                    ON knowledge_documents(created_at)
                    """
                )
                self._ensure_document_columns(cursor)
            connection.commit()

    def _ensure_document_columns(self, cursor) -> None:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'knowledge_documents'
            """
        )
        existing_columns = {row["column_name"] for row in cursor.fetchall()}
        required_columns = {
            "index_status": "TEXT NOT NULL DEFAULT 'ready'",
            "index_error": "TEXT",
        }

        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(f"ALTER TABLE knowledge_documents ADD COLUMN {column_name} {column_type}")


def build_knowledge_store(settings: Settings) -> KnowledgeStore:
    provider = settings.knowledge_store_provider.strip().lower()

    if provider == "memory":
        return InMemoryKnowledgeStore()

    if provider == "sqlite":
        return SqliteKnowledgeStore(settings.knowledge_store_path)

    if provider == "postgres":
        database_url = settings.knowledge_store_database_url
        if not database_url:
            raise ValueError("KNOWLEDGE_STORE_DATABASE_URL must be set when KNOWLEDGE_STORE_PROVIDER=postgres.")
        return PostgresKnowledgeStore(database_url)

    logger.warning(
        "Unknown knowledge store provider '%s', falling back to sqlite store.",
        settings.knowledge_store_provider,
    )
    return SqliteKnowledgeStore(settings.knowledge_store_path)

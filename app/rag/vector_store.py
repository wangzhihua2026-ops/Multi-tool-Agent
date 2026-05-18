import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.core.config import Settings

logger = logging.getLogger(__name__)

MIN_VECTOR_SCORE = 0.05


@dataclass(slots=True)
class VectorMatch:
    chunk_id: str
    score: float


class VectorStore(ABC):
    backend_name: str

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, chunk_id: str, vector: list[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def replace_all(self, entries: dict[str, list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int = 5) -> list[VectorMatch]:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    backend_name = "memory"

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._lock = RLock()

    def clear(self) -> None:
        with self._lock:
            self._vectors.clear()

    def upsert(self, chunk_id: str, vector: list[float]) -> None:
        with self._lock:
            self._vectors[chunk_id] = list(vector)

    def delete(self, chunk_ids: list[str]) -> None:
        with self._lock:
            for chunk_id in chunk_ids:
                self._vectors.pop(chunk_id, None)

    def replace_all(self, entries: dict[str, list[float]]) -> None:
        with self._lock:
            self._vectors = {
                chunk_id: list(vector)
                for chunk_id, vector in entries.items()
            }

    def count(self) -> int:
        with self._lock:
            return len(self._vectors)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[VectorMatch]:
        if not query_vector:
            return []

        with self._lock:
            matches = [
                VectorMatch(chunk_id=chunk_id, score=_dot_product(query_vector, vector))
                for chunk_id, vector in self._vectors.items()
            ]

        matches.sort(key=lambda match: match.score, reverse=True)
        return [match for match in matches[:top_k] if match.score >= MIN_VECTOR_SCORE]


class QdrantBackedVectorStore(VectorStore):
    def __init__(self, collection_name: str, dimensions: int) -> None:
        self.collection_name = collection_name
        self.dimensions = dimensions
        self._collection_ready = False
        self._lock = RLock()

    @property
    @abstractmethod
    def client(self) -> QdrantClient:
        raise NotImplementedError

    @abstractmethod
    def _point_id(self, chunk_id: str) -> str:
        raise NotImplementedError

    def clear(self) -> None:
        with self._lock:
            self._delete_collection()

    def upsert(self, chunk_id: str, vector: list[float]) -> None:
        with self._lock:
            self._ensure_collection()
            self._upsert_points([self._build_point(chunk_id, vector)])

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return

        with self._lock:
            if not self._collection_exists():
                self._collection_ready = False
                return

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.PointIdsList(
                    points=[self._point_id(chunk_id) for chunk_id in chunk_ids]
                ),
                wait=True,
            )

    def replace_all(self, entries: dict[str, list[float]]) -> None:
        with self._lock:
            snapshot = self._snapshot_points()
            try:
                self._delete_collection()
                if entries:
                    self._ensure_collection()
                    self._upsert_points(
                        [self._build_point(chunk_id, vector) for chunk_id, vector in entries.items()]
                    )
            except Exception:
                self._restore_snapshot(snapshot)
                raise

    def count(self) -> int:
        with self._lock:
            if not self._collection_exists():
                self._collection_ready = False
                return 0
            self._collection_ready = True
            return int(self.client.count(collection_name=self.collection_name, exact=True).count)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[VectorMatch]:
        if not query_vector:
            return []

        with self._lock:
            if not self._collection_exists():
                self._collection_ready = False
                return []
            self._collection_ready = True
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=MIN_VECTOR_SCORE,
                with_payload=True,
                with_vectors=False,
            )

        return [
            VectorMatch(
                chunk_id=str(hit.payload.get("chunk_id") or hit.id),
                score=float(hit.score),
            )
            for hit in response.points
            if hit.id is not None and float(hit.score) >= MIN_VECTOR_SCORE
        ]

    def _build_point(self, chunk_id: str, vector: list[float]) -> qdrant_models.PointStruct:
        return qdrant_models.PointStruct(
            id=self._point_id(chunk_id),
            vector=vector,
            payload={"chunk_id": chunk_id},
        )

    def _upsert_points(self, points: list[qdrant_models.PointStruct]) -> None:
        if not points:
            return
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points,
        )

    def _snapshot_points(self) -> list[qdrant_models.PointStruct]:
        if not self._collection_exists():
            self._collection_ready = False
            return []

        self._collection_ready = True
        snapshot: list[qdrant_models.PointStruct] = []
        offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=True,
            )
            for record in records:
                snapshot.append(
                    qdrant_models.PointStruct(
                        id=record.id,
                        vector=self._extract_vector(record.vector),
                        payload=dict(record.payload or {}),
                    )
                )
            if next_offset is None:
                break
            offset = next_offset

        return snapshot

    def _restore_snapshot(self, snapshot: list[qdrant_models.PointStruct]) -> None:
        self._delete_collection()
        if snapshot:
            self._ensure_collection()
            self._upsert_points(snapshot)

    def _ensure_collection(self) -> None:
        if self._collection_ready:
            return

        if not self._collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.dimensions,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
        self._collection_ready = True

    def _delete_collection(self) -> None:
        if self._collection_exists():
            self.client.delete_collection(self.collection_name)
        self._collection_ready = False

    def _collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    @staticmethod
    def _extract_vector(vector) -> list[float]:
        if vector is None:
            return []
        if isinstance(vector, dict):
            if not vector:
                return []
            return list(next(iter(vector.values())))
        return list(vector)


class QdrantVectorStore(QdrantBackedVectorStore):
    backend_name = "qdrant"

    def __init__(
        self,
        base_url: str,
        collection_name: str,
        dimensions: int,
        api_key: str | None = None,
        timeout_seconds: int = 15,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._client = QdrantClient(
            url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_seconds,
        )
        super().__init__(collection_name=collection_name, dimensions=dimensions)

    @property
    def client(self) -> QdrantClient:
        return self._client

    def _point_id(self, chunk_id: str) -> str:
        return chunk_id


class QdrantLocalVectorStore(QdrantBackedVectorStore):
    backend_name = "qdrant_local"
    namespace = uuid.uuid5(uuid.NAMESPACE_DNS, "multi-tool-agent-qdrant-local")

    def __init__(
        self,
        storage_path: str,
        collection_name: str,
        dimensions: int,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(self.storage_path))
        super().__init__(collection_name=collection_name, dimensions=dimensions)

    @property
    def client(self) -> QdrantClient:
        return self._client

    def _point_id(self, chunk_id: str) -> str:
        return str(uuid.uuid5(self.namespace, chunk_id))

    def close(self) -> None:
        self.client.close()


class DualWriteFallbackVectorStore(VectorStore):
    def __init__(self, primary: VectorStore, fallback: VectorStore, allow_query_fallback: bool = False) -> None:
        self.primary = primary
        self.fallback = fallback
        self.backend_name = f"{primary.backend_name}+fallback"
        self._primary_available = True
        self.allow_query_fallback = allow_query_fallback

    def clear(self) -> None:
        fallback_error = None
        primary_error = None

        try:
            self.fallback.clear()
        except Exception as exc:  # pragma: no cover - defensive
            fallback_error = exc
            logger.warning("Fallback vector store clear failed: %s", exc)

        try:
            self.primary.clear()
            self._primary_available = True
        except Exception as exc:  # pragma: no cover - external service fallback
            primary_error = exc
            self._primary_available = False
            logger.warning("Primary vector store clear failed, continuing with fallback: %s", exc)

        if fallback_error and primary_error:
            raise fallback_error

    def upsert(self, chunk_id: str, vector: list[float]) -> None:
        self.fallback.upsert(chunk_id, vector)
        try:
            self.primary.upsert(chunk_id, vector)
            self._primary_available = True
        except Exception as exc:  # pragma: no cover - external service fallback
            self._primary_available = False
            logger.warning("Primary vector store upsert failed, keeping fallback copy: %s", exc)

    def delete(self, chunk_ids: list[str]) -> None:
        self.fallback.delete(chunk_ids)
        try:
            self.primary.delete(chunk_ids)
            self._primary_available = True
        except Exception as exc:  # pragma: no cover - external service fallback
            self._primary_available = False
            logger.warning("Primary vector store delete failed, keeping fallback state: %s", exc)

    def replace_all(self, entries: dict[str, list[float]]) -> None:
        self.fallback.replace_all(entries)
        try:
            self.primary.replace_all(entries)
            self._primary_available = True
        except Exception as exc:  # pragma: no cover - external service fallback
            self._primary_available = False
            logger.warning("Primary vector store replace_all failed, continuing with fallback: %s", exc)

    def count(self) -> int:
        if not self._primary_available:
            return self.fallback.count()

        try:
            return self.primary.count()
        except Exception as exc:  # pragma: no cover - external service fallback
            self._primary_available = False
            logger.warning("Primary vector store count failed, using fallback count: %s", exc)
            return self.fallback.count()

    def search(self, query_vector: list[float], top_k: int = 5) -> list[VectorMatch]:
        if not self._primary_available:
            if self.allow_query_fallback:
                return self.fallback.search(query_vector, top_k=top_k)
            raise RuntimeError("Primary vector store is unavailable; query fallback is disabled.")

        try:
            return self.primary.search(query_vector, top_k=top_k)
        except Exception as exc:  # pragma: no cover - external service fallback
            self._primary_available = False
            logger.warning("Primary vector store search failed: %s", exc)
            if self.allow_query_fallback:
                return self.fallback.search(query_vector, top_k=top_k)
            raise RuntimeError("Primary vector store search failed; query fallback is disabled.") from exc


def build_vector_store(settings: Settings) -> VectorStore:
    provider = settings.vector_store_provider.strip().lower()

    if provider == "memory":
        return InMemoryVectorStore()

    if provider == "qdrant_local":
        return QdrantLocalVectorStore(
            storage_path=settings.vector_store_path,
            collection_name=settings.vector_store_collection,
            dimensions=settings.embedding_dimensions,
        )

    if provider == "qdrant":
        fallback = InMemoryVectorStore()
        primary = QdrantVectorStore(
            base_url=settings.vector_store_url,
            collection_name=settings.vector_store_collection,
            dimensions=settings.embedding_dimensions,
            api_key=settings.vector_store_api_key,
            timeout_seconds=settings.vector_store_timeout_seconds,
        )
        return DualWriteFallbackVectorStore(
            primary=primary,
            fallback=fallback,
            allow_query_fallback=settings.vector_store_query_fallback_enabled,
        )

    logger.warning("Unknown vector store provider '%s', falling back to in-memory store.", settings.vector_store_provider)
    return InMemoryVectorStore()


def _dot_product(left: list[float], right: list[float]) -> float:
    width = min(len(left), len(right))
    return round(sum(left[index] * right[index] for index in range(width)), 4)

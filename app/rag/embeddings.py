import hashlib
import logging
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from threading import RLock

import httpx

from app.core.config import Settings
from app.rag.text import tokenize_text

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    provider_name: str

    @property
    def embedding_signature(self) -> str:
        return self.provider_name

    @property
    def last_embedding_signature(self) -> str:
        return self.embedding_signature

    def is_compatible_signature(self, stored_signature: str | None, query_signature: str | None = None) -> bool:
        if not stored_signature:
            return False
        compatible = {self.provider_name, self.embedding_signature}
        if query_signature is not None:
            compatible = {query_signature}
            if query_signature == self.embedding_signature:
                compatible.add(self.provider_name)
        return stored_signature in compatible

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)


class HashEmbeddingProvider(EmbeddingProvider):
    provider_name = "hash"

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    @property
    def embedding_signature(self) -> str:
        return f"{self.provider_name}:{self.dimensions}"

    def embed_text(self, text: str) -> list[float]:
        tokens = tokenize_text(text)
        if not tokens:
            return [0.0] * self.dimensions

        vector = [0.0] * self.dimensions
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            for offset in (0, 4):
                bucket = int.from_bytes(digest[offset : offset + 4], "big") % self.dimensions
                sign = 1.0 if digest[offset] % 2 == 0 else -1.0
                vector[bucket] += sign

        return _normalize_vector(vector)


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    provider_name = "openai"

    def __init__(self, settings: Settings) -> None:
        if not settings.resolved_embedding_api_key:
            raise ValueError("A resolved embedding API key is required for the OpenAI-compatible embedding provider.")
        self.settings = settings

    @property
    def embedding_signature(self) -> str:
        return f"{self.provider_name}:{self.settings.embedding_model}:{self.settings.embedding_dimensions}"

    def embed_text(self, text: str) -> list[float]:
        result = self.embed_texts([text])
        return result[0] if result else []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.settings.embedding_model,
            "input": texts,
        }
        if self.settings.embedding_dimensions > 0:
            payload["dimensions"] = self.settings.embedding_dimensions

        data = self._post_embeddings(payload)
        embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda item: item["index"])]
        return [_normalize_vector([float(value) for value in embedding]) for embedding in embeddings]

    def _post_embeddings(self, payload: dict) -> dict:
        base_url = self.settings.embedding_base_url.rstrip("/")
        headers = {
            "Authorization": f"Bearer {self.settings.resolved_embedding_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
            response = client.post(f"{base_url}/embeddings", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    provider_name = "sentence_transformers"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = RLock()
        self._model = None

    @property
    def embedding_signature(self) -> str:
        return f"{self.provider_name}:{self.settings.embedding_model}:{self.settings.embedding_dimensions}"

    def embed_text(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def embed_query(self, text: str) -> list[float]:
        result = self._encode([self._apply_prefix(text, self._query_prefix())])
        return result[0] if result else []

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self._apply_prefix(text, self._passage_prefix()) for text in texts]
        return self._encode(prefixed)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self.settings.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [[float(value) for value in embedding] for embedding in embeddings]

    def _get_model(self):
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is None:
                cache_folder = self._prepare_cache()
                sentence_transformer_cls = _load_sentence_transformer()
                self._model = sentence_transformer_cls(
                    self.settings.embedding_model,
                    device=self.settings.embedding_device,
                    cache_folder=cache_folder,
                    trust_remote_code=self.settings.embedding_trust_remote_code,
                    local_files_only=self.settings.embedding_local_files_only,
                )
        return self._model

    def _prepare_cache(self) -> str | None:
        cache_path = (self.settings.embedding_cache_path or "").strip()
        if cache_path:
            os.environ.setdefault("HF_HOME", cache_path)
        if self.settings.embedding_local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        if not cache_path:
            return None
        hub_cache = Path(cache_path) / "hub"
        return str(hub_cache) if hub_cache.exists() else cache_path

    def _query_prefix(self) -> str:
        if self.settings.embedding_query_prefix:
            return self.settings.embedding_query_prefix
        if "e5" in self.settings.embedding_model.lower():
            return "query: "
        return ""

    def _passage_prefix(self) -> str:
        if self.settings.embedding_passage_prefix:
            return self.settings.embedding_passage_prefix
        if "e5" in self.settings.embedding_model.lower():
            return "passage: "
        return ""

    def _apply_prefix(self, text: str, prefix: str) -> str:
        normalized_text = text.strip()
        if not prefix:
            return normalized_text
        prefix = prefix if prefix.endswith(" ") else f"{prefix} "
        return f"{prefix}{normalized_text}"


class FallbackEmbeddingProvider(EmbeddingProvider):
    def __init__(self, primary: EmbeddingProvider, fallback: EmbeddingProvider) -> None:
        self.primary = primary
        self.fallback = fallback
        self.provider_name = f"{primary.provider_name}+fallback"
        self._last_embedding_signature = primary.embedding_signature

    @property
    def embedding_signature(self) -> str:
        return self.primary.embedding_signature

    @property
    def last_embedding_signature(self) -> str:
        return self._last_embedding_signature

    def is_compatible_signature(self, stored_signature: str | None, query_signature: str | None = None) -> bool:
        return self.primary.is_compatible_signature(
            stored_signature,
            query_signature=query_signature,
        ) or self.fallback.is_compatible_signature(
            stored_signature,
            query_signature=query_signature,
        )

    def embed_text(self, text: str) -> list[float]:
        try:
            result = self.primary.embed_text(text)
            self._last_embedding_signature = self.primary.last_embedding_signature
            return result
        except Exception as exc:  # pragma: no cover - network/provider fallback
            logger.warning("Primary embedding provider failed, falling back to hash embeddings: %s", exc)
            result = self.fallback.embed_text(text)
            self._last_embedding_signature = self.fallback.last_embedding_signature
            return result

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            result = self.primary.embed_texts(texts)
            self._last_embedding_signature = self.primary.last_embedding_signature
            return result
        except Exception as exc:  # pragma: no cover - network/provider fallback
            logger.warning("Primary embedding batch request failed, falling back to hash embeddings: %s", exc)
            result = self.fallback.embed_texts(texts)
            self._last_embedding_signature = self.fallback.last_embedding_signature
            return result

    def embed_query(self, text: str) -> list[float]:
        try:
            result = self.primary.embed_query(text)
            self._last_embedding_signature = self.primary.last_embedding_signature
            return result
        except Exception as exc:  # pragma: no cover - network/provider fallback
            logger.warning("Primary embedding query request failed, falling back to hash embeddings: %s", exc)
            result = self.fallback.embed_query(text)
            self._last_embedding_signature = self.fallback.last_embedding_signature
            return result

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        try:
            result = self.primary.embed_passages(texts)
            self._last_embedding_signature = self.primary.last_embedding_signature
            return result
        except Exception as exc:  # pragma: no cover - network/provider fallback
            logger.warning("Primary passage embedding batch request failed, falling back to hash embeddings: %s", exc)
            result = self.fallback.embed_passages(texts)
            self._last_embedding_signature = self.fallback.last_embedding_signature
            return result


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    fallback = HashEmbeddingProvider(dimensions=settings.embedding_dimensions)
    provider = settings.embedding_provider.strip().lower()

    if provider == "hash":
        return fallback

    if provider == "openai":
        if not settings.resolved_embedding_api_key:
            if settings.embedding_fallback_enabled:
                logger.warning("OpenAI-compatible embedding provider has no API key; falling back to hash embeddings.")
                return fallback
            return OpenAICompatibleEmbeddingProvider(settings)
        primary = OpenAICompatibleEmbeddingProvider(settings)
        if settings.embedding_fallback_enabled:
            return FallbackEmbeddingProvider(primary=primary, fallback=fallback)
        return primary

    if provider == "sentence_transformers":
        try:
            primary = SentenceTransformerEmbeddingProvider(settings)
            if settings.embedding_fallback_enabled:
                return FallbackEmbeddingProvider(primary=primary, fallback=fallback)
            return primary
        except Exception as exc:  # pragma: no cover - defensive constructor fallback
            if not settings.embedding_fallback_enabled:
                raise
            logger.warning("Local sentence-transformers provider failed to initialize, falling back to hash embeddings: %s", exc)
            return fallback

    logger.warning("Unknown embedding provider '%s', falling back to hash embeddings.", settings.embedding_provider)
    return fallback


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer

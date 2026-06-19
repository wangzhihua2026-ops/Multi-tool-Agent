import logging
from dataclasses import dataclass
from typing import Protocol

from app.rag.parent_aggregation import ParentCandidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankResult(ParentCandidate):
    rerank_score: float | None = None
    aggregated_score: float | None = None


class Reranker(Protocol):
    backend_name: str

    def rerank(self, query: str, candidates: list[ParentCandidate], top_k: int) -> list[RerankResult]:
        ...


class NoopReranker:
    backend_name = "none"

    def rerank(self, query: str, candidates: list[ParentCandidate], top_k: int) -> list[RerankResult]:
        del query
        ordered = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
        return [
            RerankResult(
                parent_id=candidate.parent_id,
                document_id=candidate.document_id,
                document_title=candidate.document_title,
                parent_index=candidate.parent_index,
                content=candidate.content,
                score=candidate.score,
                evidence_chunk_ids=candidate.evidence_chunk_ids,
                vector_score=candidate.vector_score,
                bm25_score=candidate.bm25_score,
                rerank_score=None,
                aggregated_score=candidate.score,
            )
            for candidate in ordered[:top_k]
        ]


class CrossEncoderReranker:
    backend_name = "cross_encoder"

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 8,
        fallback_enabled: bool = True,
        model: object | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.fallback_enabled = fallback_enabled
        self._fallback = NoopReranker()
        self._model = model or self._load_model()

    def rerank(self, query: str, candidates: list[ParentCandidate], top_k: int) -> list[RerankResult]:
        if self._model is None:
            return self._fallback.rerank(query, candidates, top_k)

        pairs = [(query, candidate.content) for candidate in candidates]
        try:
            scores = self._model.predict(pairs, batch_size=self.batch_size)
        except TypeError:
            scores = self._model.predict(pairs)
        except Exception:
            if self.fallback_enabled:
                logger.warning("CrossEncoder reranker failed; using aggregate order.", exc_info=True)
                return self._fallback.rerank(query, candidates, top_k)
            raise

        reranked = [
            RerankResult(
                parent_id=candidate.parent_id,
                document_id=candidate.document_id,
                document_title=candidate.document_title,
                parent_index=candidate.parent_index,
                content=candidate.content,
                score=float(score),
                evidence_chunk_ids=candidate.evidence_chunk_ids,
                vector_score=candidate.vector_score,
                bm25_score=candidate.bm25_score,
                rerank_score=float(score),
                aggregated_score=candidate.score,
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked[:top_k]

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            if self.fallback_enabled:
                logger.warning("sentence-transformers is not installed; using NoopReranker.")
                return None
            raise RuntimeError(
                "CrossEncoder reranker requires the optional 'sentence-transformers' dependency."
            )

        try:
            return CrossEncoder(self.model_name, device=self.device)
        except Exception:
            if self.fallback_enabled:
                logger.warning("CrossEncoder model failed to load; using NoopReranker.", exc_info=True)
                return None
            raise


def build_reranker(
    provider: str,
    model_name: str,
    device: str = "cpu",
    batch_size: int = 8,
    fallback_enabled: bool = True,
) -> Reranker:
    normalized = provider.strip().lower()
    if normalized in {"", "none", "noop"}:
        return NoopReranker()
    if normalized in {"cross_encoder", "cross-encoder", "reranker"}:
        return CrossEncoderReranker(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            fallback_enabled=fallback_enabled,
        )
    raise ValueError(f"Unknown reranker provider: {provider}")

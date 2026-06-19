from dataclasses import dataclass

from app.rag.bm25 import Bm25Index
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import ParentSearchHit
from app.rag.parent_aggregation import (
    ChildSignal,
    ParentAggregationSettings,
    aggregate_parent_candidates,
)
from app.rag.reranker import NoopReranker, Reranker
from app.rag.store import KnowledgeStore
from app.rag.vector_store import VectorStore


@dataclass(frozen=True)
class AdvancedRetrievalSettings:
    bm25_enabled: bool = True
    vector_top_k_multiplier: int = 8
    bm25_top_k_multiplier: int = 8
    min_child_candidates: int = 24
    parent_candidate_limit: int = 20
    vector_rrf_weight: float = 0.55
    bm25_rrf_weight: float = 0.45
    evidence_bonus: float = 0.01
    rrf_constant: int = 60


class ParentChildRetriever:
    def __init__(
        self,
        store: KnowledgeStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker,
        settings: AdvancedRetrievalSettings | None = None,
    ) -> None:
        self.store = store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.settings = settings or AdvancedRetrievalSettings()
        self._noop_reranker = NoopReranker()
        self._bm25_index: Bm25Index | None = None
        self._bm25_fingerprint: tuple[str, ...] = ()

    def search(
        self,
        query: str,
        top_k: int = 3,
        use_reranker: bool = False,
    ) -> list[ParentSearchHit]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        vector_matches = self._search_vector_children(normalized_query, top_k)
        bm25_matches = self._search_bm25_children(normalized_query, top_k)
        candidates = aggregate_parent_candidates(
            store=self.store,
            vector_matches=vector_matches,
            bm25_matches=bm25_matches,
            settings=ParentAggregationSettings(
                vector_rrf_weight=self.settings.vector_rrf_weight,
                bm25_rrf_weight=self.settings.bm25_rrf_weight,
                evidence_bonus=self.settings.evidence_bonus,
                rrf_constant=self.settings.rrf_constant,
            ),
        )
        candidates = candidates[: self.settings.parent_candidate_limit]
        reranker = self.reranker if use_reranker else self._noop_reranker
        reranked = reranker.rerank(normalized_query, candidates, top_k=top_k)
        retrieval_mode = "parent_child_rerank" if use_reranker else "parent_child_hybrid"

        return [
            ParentSearchHit(
                document_id=result.document_id,
                document_title=result.document_title,
                chunk_id=result.parent_id,
                chunk_index=result.parent_index,
                content=result.content,
                score=round(result.score, 6),
                retrieval_mode=retrieval_mode,
                lexical_score=result.bm25_score,
                vector_score=result.vector_score,
                parent_id=result.parent_id,
                parent_index=result.parent_index,
                evidence_chunk_ids=result.evidence_chunk_ids,
                bm25_score=result.bm25_score,
                rerank_score=result.rerank_score,
                aggregated_child_score=result.aggregated_score or result.score,
            )
            for result in reranked
        ]

    def _search_vector_children(self, query: str, top_k: int) -> list[ChildSignal]:
        query_vector = self.embedding_provider.embed_query(query)
        query_signature = self.embedding_provider.last_embedding_signature
        vector_top_k = max(top_k * self.settings.vector_top_k_multiplier, self.settings.min_child_candidates)
        signals: list[ChildSignal] = []
        for match in self.vector_store.search(query_vector, top_k=vector_top_k):
            chunk = self.store.get_chunk(match.chunk_id)
            if chunk is None:
                continue
            if not self.embedding_provider.is_compatible_signature(
                chunk.embedding_provider,
                query_signature=query_signature,
            ):
                continue
            signals.append(
                ChildSignal(
                    chunk_id=match.chunk_id,
                    score=match.score,
                    rank=len(signals) + 1,
                )
            )
        return signals

    def _search_bm25_children(self, query: str, top_k: int) -> list[ChildSignal]:
        if not self.settings.bm25_enabled:
            return []

        bm25_top_k = max(top_k * self.settings.bm25_top_k_multiplier, self.settings.min_child_candidates)
        return [
            ChildSignal(chunk_id=match.chunk_id, score=match.score, rank=match.rank)
            for match in self._get_bm25_index().search(query, top_k=bm25_top_k)
        ]

    def _get_bm25_index(self) -> Bm25Index:
        chunks = self.store.get_chunks()
        fingerprint = tuple(chunk.chunk_id for chunk in chunks)
        if self._bm25_index is None or self._bm25_fingerprint != fingerprint:
            self._bm25_index = Bm25Index(chunks)
            self._bm25_fingerprint = fingerprint
        return self._bm25_index

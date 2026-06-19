from app.rag.advanced_retriever import AdvancedRetrievalSettings, ParentChildRetriever
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import SearchHit
from app.rag.reranker import NoopReranker, Reranker
from app.rag.store import KnowledgeStore
from app.rag.text import score_lexical_match, tokenize_text
from app.rag.vector_store import VectorStore


class KnowledgeRetriever:
    def __init__(
        self,
        store: KnowledgeStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        vector_weight: float = 0.7,
        lexical_weight: float = 0.3,
        fusion_rank_constant: int = 60,
        parent_child_retriever: ParentChildRetriever | None = None,
        advanced_settings: AdvancedRetrievalSettings | None = None,
        reranker: Reranker | None = None,
        default_strategy: str = "hybrid",
    ) -> None:
        self.store = store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.vector_weight = vector_weight
        self.lexical_weight = lexical_weight
        self.fusion_rank_constant = fusion_rank_constant
        self.default_strategy = default_strategy
        self.parent_child_retriever = parent_child_retriever or ParentChildRetriever(
            store=store,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            reranker=reranker or NoopReranker(),
            settings=advanced_settings,
        )

    def search(self, query: str, top_k: int = 3, strategy: str | None = None) -> list[SearchHit]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        strategy = strategy or self.default_strategy
        if strategy in {"parent_child", "parent_child_rerank"}:
            return self.parent_child_retriever.search(
                query=normalized_query,
                top_k=top_k,
                use_reranker=strategy == "parent_child_rerank",
            )
        if strategy not in {"lexical", "vector", "hybrid"}:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

        lexical_scores: dict[str, float] = {}
        if strategy in {"lexical", "hybrid"}:
            query_tokens = set(tokenize_text(normalized_query))
            for chunk in self.store.get_chunks():
                lexical_score = score_lexical_match(
                    query=normalized_query,
                    query_tokens=query_tokens,
                    chunk_text=chunk.content,
                    chunk_tokens=set(chunk.tokens),
                )
                if lexical_score > 0:
                    lexical_scores[chunk.chunk_id] = lexical_score

        vector_scores: dict[str, float] = {}
        if strategy in {"vector", "hybrid"}:
            query_vector = self.embedding_provider.embed_query(normalized_query)
            query_signature = self.embedding_provider.last_embedding_signature
            vector_matches = self.vector_store.search(query_vector, top_k=max(top_k * 4, top_k))
            for match in vector_matches:
                chunk = self.store.get_chunk(match.chunk_id)
                if chunk is None:
                    continue
                if not self.embedding_provider.is_compatible_signature(
                    chunk.embedding_provider,
                    query_signature=query_signature,
                ):
                    continue
                vector_scores[match.chunk_id] = match.score

        candidate_ids = set(lexical_scores) | set(vector_scores)
        lexical_ranks = self._rank_scores(lexical_scores)
        vector_ranks = self._rank_scores(vector_scores)
        hits: list[SearchHit] = []
        for chunk_id in candidate_ids:
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:
                continue

            lexical_score = lexical_scores.get(chunk_id)
            vector_score = vector_scores.get(chunk_id)
            if strategy == "hybrid":
                score = self._fuse_rank_scores(
                    lexical_rank=lexical_ranks.get(chunk_id),
                    vector_rank=vector_ranks.get(chunk_id),
                )
            else:
                score = self._blend_scores(lexical_score=lexical_score, vector_score=vector_score)
            if score <= 0:
                continue
            hits.append(
                SearchHit(
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.index,
                    content=chunk.content,
                    score=score,
                    retrieval_mode=self._determine_mode(lexical_score=lexical_score, vector_score=vector_score),
                    lexical_score=lexical_score,
                    vector_score=vector_score,
                )
            )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    @staticmethod
    def _rank_scores(scores: dict[str, float]) -> dict[str, int]:
        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        return {chunk_id: index for index, chunk_id in enumerate(ranked_ids, start=1)}

    def _fuse_rank_scores(self, lexical_rank: int | None, vector_rank: int | None) -> float:
        # Rank fusion avoids directly comparing lexical scores with cosine similarities.
        score = 0.0
        if vector_rank is not None:
            score += self.vector_weight / (self.fusion_rank_constant + vector_rank)
        if lexical_rank is not None:
            score += self.lexical_weight / (self.fusion_rank_constant + lexical_rank)
        return round(score, 6)

    def _blend_scores(self, lexical_score: float | None, vector_score: float | None) -> float:
        normalized_lexical = min((lexical_score or 0.0) / 2.0, 1.0)
        normalized_vector = max(vector_score or 0.0, 0.0)

        if normalized_lexical == 0.0 and normalized_vector == 0.0:
            return 0.0

        if normalized_lexical == 0.0:
            return round(normalized_vector, 4)
        if normalized_vector == 0.0:
            return round(normalized_lexical, 4)
        return round((normalized_vector * self.vector_weight) + (normalized_lexical * self.lexical_weight), 4)

    def _determine_mode(self, lexical_score: float | None, vector_score: float | None) -> str:
        if lexical_score and vector_score:
            return "hybrid"
        if vector_score:
            return "vector"
        return "lexical"

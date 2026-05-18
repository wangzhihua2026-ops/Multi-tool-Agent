from app.rag.embeddings import EmbeddingProvider
from app.rag.models import SearchHit
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
    ) -> None:
        self.store = store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.vector_weight = vector_weight
        self.lexical_weight = lexical_weight

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        query_tokens = set(tokenize_text(normalized_query))
        lexical_scores: dict[str, float] = {}
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
        hits: list[SearchHit] = []
        for chunk_id in candidate_ids:
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:
                continue

            lexical_score = lexical_scores.get(chunk_id)
            vector_score = vector_scores.get(chunk_id)
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

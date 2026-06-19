from dataclasses import dataclass
import math

from app.rag.models import ChunkRecord
from app.rag.text import tokenize_text


@dataclass(frozen=True)
class Bm25Match:
    chunk_id: str
    score: float
    rank: int


class Bm25Index:
    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self._chunks = list(chunks)
        self._tokenized_corpus = [
            chunk.tokens or tokenize_text(chunk.content)
            for chunk in self._chunks
        ]
        self._rank_bm25 = self._build_optional_rank_bm25()
        self._doc_frequency = self._build_doc_frequency()
        self._average_doc_length = (
            sum(len(tokens) for tokens in self._tokenized_corpus) / len(self._tokenized_corpus)
            if self._tokenized_corpus
            else 0.0
        )

    def search(self, query: str, top_k: int) -> list[Bm25Match]:
        query_tokens = tokenize_text(query)
        if not query_tokens or top_k <= 0 or not self._chunks:
            return []

        if self._rank_bm25 is not None:
            raw_scores = self._rank_bm25.get_scores(query_tokens)
            scored = [
                (chunk.chunk_id, float(score))
                for chunk, score in zip(self._chunks, raw_scores, strict=True)
                if float(score) > 0
            ]
        else:
            scored = [
                (chunk.chunk_id, self._score_local_bm25(query_tokens, index))
                for index, chunk in enumerate(self._chunks)
            ]
            scored = [(chunk_id, score) for chunk_id, score in scored if score > 0]

        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            Bm25Match(chunk_id=chunk_id, score=round(score, 6), rank=rank)
            for rank, (chunk_id, score) in enumerate(scored[:top_k], start=1)
        ]

    def _build_optional_rank_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return None
        return BM25Okapi(self._tokenized_corpus)

    def _build_doc_frequency(self) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for tokens in self._tokenized_corpus:
            for token in set(tokens):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    def _score_local_bm25(self, query_tokens: list[str], document_index: int) -> float:
        tokens = self._tokenized_corpus[document_index]
        if not tokens:
            return 0.0

        token_counts: dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        score = 0.0
        document_count = len(self._tokenized_corpus)
        document_length = len(tokens)
        k1 = 1.5
        b = 0.75
        for token in query_tokens:
            frequency = token_counts.get(token, 0)
            if frequency == 0:
                continue
            document_frequency = self._doc_frequency.get(token, 0)
            idf = math.log(1 + ((document_count - document_frequency + 0.5) / (document_frequency + 0.5)))
            denominator = frequency + k1 * (
                1 - b + b * (document_length / max(self._average_doc_length, 1.0))
            )
            score += idf * ((frequency * (k1 + 1)) / denominator)
        return score

from app.rag.bm25 import Bm25Index
from app.rag.models import ChunkRecord, DocumentRecord, ParentBlockRecord
from app.rag.parent_aggregation import (
    ChildSignal,
    ParentAggregationSettings,
    ParentCandidate,
    aggregate_parent_candidates,
)
from app.rag.reranker import NoopReranker
from app.rag.store import InMemoryKnowledgeStore


def test_bm25_ranks_exact_keyword_match_first() -> None:
    chunks = [
        ChunkRecord(
            document_id="d1",
            document_title="Doc 1",
            index=0,
            content="alpha beta",
            tokens=["alpha", "beta"],
        ),
        ChunkRecord(
            document_id="d2",
            document_title="Doc 2",
            index=0,
            content="gamma delta",
            tokens=["gamma", "delta"],
        ),
    ]

    matches = Bm25Index(chunks).search("alpha", top_k=2)

    assert matches[0].chunk_id == chunks[0].chunk_id
    assert matches[0].score > 0


def test_parent_aggregation_deduplicates_parent_and_keeps_evidence() -> None:
    store = InMemoryKnowledgeStore()
    document = DocumentRecord(title="guide", content="Parent body", metadata={})
    parent = ParentBlockRecord(
        document_id=document.document_id,
        document_title=document.title,
        index=0,
        content="Parent body",
    )
    chunks = [
        ChunkRecord(
            document_id=document.document_id,
            document_title=document.title,
            index=0,
            content="first",
            parent_id=parent.parent_id,
        ),
        ChunkRecord(
            document_id=document.document_id,
            document_title=document.title,
            index=1,
            content="second",
            parent_id=parent.parent_id,
        ),
    ]
    store.add_document(document, chunks, parent_blocks=[parent])

    candidates = aggregate_parent_candidates(
        store=store,
        vector_matches=[ChildSignal(chunk_id=chunks[0].chunk_id, score=0.9, rank=1)],
        bm25_matches=[ChildSignal(chunk_id=chunks[1].chunk_id, score=3.0, rank=1)],
        settings=ParentAggregationSettings(),
    )

    assert len(candidates) == 1
    assert candidates[0].parent_id == parent.parent_id
    assert set(candidates[0].evidence_chunk_ids) == {chunks[0].chunk_id, chunks[1].chunk_id}
    assert candidates[0].vector_score is not None
    assert candidates[0].bm25_score is not None


def test_parent_aggregation_uses_synthetic_parent_for_flat_chunks() -> None:
    store = InMemoryKnowledgeStore()
    document = DocumentRecord(title="flat", content="Flat chunk body", metadata={})
    chunk = ChunkRecord(
        document_id=document.document_id,
        document_title=document.title,
        index=0,
        content="Flat chunk body",
        tokens=["flat", "chunk", "body"],
    )
    store.add_document(document, [chunk])

    candidates = aggregate_parent_candidates(
        store=store,
        vector_matches=[],
        bm25_matches=[ChildSignal(chunk_id=chunk.chunk_id, score=2.0, rank=1)],
        settings=ParentAggregationSettings(),
    )

    assert candidates[0].parent_id == chunk.chunk_id
    assert candidates[0].content == chunk.content
    assert candidates[0].evidence_chunk_ids == [chunk.chunk_id]


def test_noop_reranker_preserves_aggregate_order() -> None:
    candidates = [
        ParentCandidate(
            parent_id="p1",
            document_id="d",
            document_title="d",
            parent_index=0,
            content="one",
            score=0.2,
        ),
        ParentCandidate(
            parent_id="p2",
            document_id="d",
            document_title="d",
            parent_index=1,
            content="two",
            score=0.5,
        ),
    ]

    results = NoopReranker().rerank("query", candidates, top_k=1)

    assert results[0].parent_id == "p2"
    assert results[0].rerank_score is None
    assert results[0].score == 0.5

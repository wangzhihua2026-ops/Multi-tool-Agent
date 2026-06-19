from dataclasses import dataclass, field

from app.rag.models import ChunkRecord
from app.rag.store import KnowledgeStore


@dataclass(frozen=True)
class ChildSignal:
    chunk_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class ParentAggregationSettings:
    vector_rrf_weight: float = 0.55
    bm25_rrf_weight: float = 0.45
    evidence_bonus: float = 0.01
    rrf_constant: int = 60
    max_evidence_children: int = 3


@dataclass(frozen=True)
class ParentCandidate:
    parent_id: str
    document_id: str
    document_title: str
    parent_index: int
    content: str
    score: float
    evidence_chunk_ids: list[str] = field(default_factory=list)
    vector_score: float | None = None
    bm25_score: float | None = None


@dataclass
class _ParentAccumulator:
    parent_id: str
    document_id: str
    document_title: str
    parent_index: int
    content: str
    evidence_chunk_ids: list[str] = field(default_factory=list)
    best_vector_rrf: float = 0.0
    best_bm25_rrf: float = 0.0
    vector_score: float | None = None
    bm25_score: float | None = None

    def add_evidence(self, chunk_id: str) -> None:
        if chunk_id not in self.evidence_chunk_ids:
            self.evidence_chunk_ids.append(chunk_id)


def aggregate_parent_candidates(
    store: KnowledgeStore,
    vector_matches: list[ChildSignal],
    bm25_matches: list[ChildSignal],
    settings: ParentAggregationSettings,
) -> list[ParentCandidate]:
    accumulators: dict[str, _ParentAccumulator] = {}

    for signal in vector_matches:
        chunk = store.get_chunk(signal.chunk_id)
        if chunk is None:
            continue
        accumulator = _get_or_create_accumulator(store, accumulators, chunk)
        accumulator.add_evidence(chunk.chunk_id)
        accumulator.best_vector_rrf = max(
            accumulator.best_vector_rrf,
            _rrf(signal.rank, settings.rrf_constant),
        )
        accumulator.vector_score = max(accumulator.vector_score or signal.score, signal.score)

    for signal in bm25_matches:
        chunk = store.get_chunk(signal.chunk_id)
        if chunk is None:
            continue
        accumulator = _get_or_create_accumulator(store, accumulators, chunk)
        accumulator.add_evidence(chunk.chunk_id)
        accumulator.best_bm25_rrf = max(
            accumulator.best_bm25_rrf,
            _rrf(signal.rank, settings.rrf_constant),
        )
        accumulator.bm25_score = max(accumulator.bm25_score or signal.score, signal.score)

    candidates = [
        ParentCandidate(
            parent_id=accumulator.parent_id,
            document_id=accumulator.document_id,
            document_title=accumulator.document_title,
            parent_index=accumulator.parent_index,
            content=accumulator.content,
            score=round(
                settings.vector_rrf_weight * accumulator.best_vector_rrf
                + settings.bm25_rrf_weight * accumulator.best_bm25_rrf
                + settings.evidence_bonus
                * min(len(accumulator.evidence_chunk_ids), settings.max_evidence_children),
                6,
            ),
            evidence_chunk_ids=accumulator.evidence_chunk_ids[: settings.max_evidence_children],
            vector_score=accumulator.vector_score,
            bm25_score=accumulator.bm25_score,
        )
        for accumulator in accumulators.values()
    ]
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def _get_or_create_accumulator(
    store: KnowledgeStore,
    accumulators: dict[str, _ParentAccumulator],
    chunk: ChunkRecord,
) -> _ParentAccumulator:
    parent = store.get_parent_block(chunk.parent_id) if chunk.parent_id else None
    if parent is None:
        parent_id = chunk.chunk_id
        if parent_id not in accumulators:
            accumulators[parent_id] = _ParentAccumulator(
                parent_id=parent_id,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                parent_index=chunk.index,
                content=chunk.content,
            )
        return accumulators[parent_id]

    if parent.parent_id not in accumulators:
        accumulators[parent.parent_id] = _ParentAccumulator(
            parent_id=parent.parent_id,
            document_id=parent.document_id,
            document_title=parent.document_title,
            parent_index=parent.index,
            content=parent.content,
        )
    return accumulators[parent.parent_id]


def _rrf(rank: int, constant: int) -> float:
    return 1.0 / (constant + rank)

# RAG Parent-Child Retrieval and Reranker Design

Date: 2026-06-19

## Summary

This design upgrades the current single-layer hybrid retriever into a parent-child retrieval pipeline. Documents are split into larger parent blocks for answer context and smaller child chunks for recall. Search first retrieves child chunks through vector and BM25 keyword paths, deduplicates and aggregates them back to parent blocks, then optionally reranks parent candidates with a CrossEncoder model such as `BAAI/bge-reranker-base`. The final result keeps the existing API shape compatible where possible while exposing richer retrieval metadata for evaluation and debugging.

The design is intentionally incremental. It preserves the existing document upload API, document parser improvements, embedding providers, Qdrant Local vector store, and current lexical/vector/hybrid retriever as compatibility paths. The new pipeline is added as a configurable strategy, then becomes the recommended retrieval mode after tests and benchmark evidence are in place.

## Goals

- Add parent-child indexing:
  - Parent block target size: about 1600 characters.
  - Child chunk target size: about 450 characters.
  - Child chunks store `parent_id` and parent-relative metadata.
- Improve recall with two independent retrieval paths:
  - Vector retrieval against child chunks.
  - BM25 keyword retrieval against child chunks.
- Aggregate child hits to parent blocks before model context construction.
- Add CrossEncoder reranking over parent candidates with an optional local model.
- Preserve current upload, parsing, chunking, embedding, and API behavior for existing tests.
- Keep reranker and FAISS dependencies optional so lightweight local development remains usable.
- Add benchmark support for comparing old and new strategies with Hit@1, Recall@k, MRR, latency, and reranker contribution.

## Non-Goals

- Do not replace the A-phase document parser. PDF/OCR/table extraction remains upstream of this design.
- Do not build a Streamlit management app in this phase.
- Do not require FAISS for production. Qdrant Local remains the default vector backend; FAISS is an optional local backend.
- Do not claim production traffic metrics such as daily query count or answer acceptance rate.
- Do not add answer-quality or groundedness evaluation in the first implementation of this design.

## Current Baseline

The current RAG path stores one document record and a flat list of `ChunkRecord` items. `DocumentService.create_document` splits document text through `chunk_text(content, chunk_size=500, chunk_overlap=80)`, embeds each chunk, and writes the chunk vector to the configured vector store. `KnowledgeRetriever.search` supports `lexical`, `vector`, and `hybrid` strategies. Lexical search uses custom token overlap scoring, vector search uses the configured vector store, and hybrid ranking uses weighted reciprocal rank fusion.

The current benchmark script evaluates `lexical`, `vector`, and `hybrid` over a small labelled local corpus. It reports Hit@1, Recall@3, MRR@3, and latency.

## Design Overview

The new retrieval path has four stages:

1. **Hierarchical Indexing**
   - Parse document text as today.
   - Split into parent blocks around semantic boundaries.
   - Split each parent block into child chunks.
   - Store parent blocks, child chunks, and parent-child relations.
   - Embed child chunks only.
   - Build or refresh a BM25 index over child chunks.

2. **Dual Recall**
   - Vector search returns child chunk ids with vector scores.
   - BM25 search returns child chunk ids with keyword scores.
   - Candidate child ids are deduplicated and merged with rank metadata.

3. **Parent Aggregation**
   - Each child hit maps to its parent block.
   - Parent score is computed from best child ranks and signal coverage.
   - Parent candidates retain evidence child ids for debugging.

4. **Parent Reranking**
   - Optional CrossEncoder scores `(query, parent_content)` pairs.
   - Reranked parents become final search hits.
   - Returned content is the parent block, not the child chunk.

## Proposed Data Model

### `ParentBlockRecord`

Add a new model in `app/rag/models.py`:

```python
class ParentBlockRecord(BaseModel):
    parent_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    document_title: str
    index: int
    content: str
    tokens: list[str] = Field(default_factory=list)
    start_offset: int | None = None
    end_offset: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
```

### `ChildChunkRecord`

The existing `ChunkRecord` can be extended instead of replaced:

```python
class ChunkRecord(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    document_title: str
    index: int
    content: str
    tokens: list[str] = Field(default_factory=list)
    embedding_provider: str | None = None
    parent_id: str | None = None
    parent_index: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
```

Extending `ChunkRecord` preserves compatibility with existing vector store and retriever code. Existing single-layer chunks have `parent_id=None`; hierarchical chunks always have `parent_id`.

### `ParentSearchHit`

Add a richer result model while preserving `SearchHit` compatibility:

```python
class ParentSearchHit(SearchHit):
    parent_id: str
    parent_index: int
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    bm25_score: float | None = None
    rerank_score: float | None = None
    aggregated_child_score: float | None = None
```

Existing API endpoints can continue returning `SearchHit` fields. New debug endpoints or benchmark code can inspect `ParentSearchHit` fields.

## Storage Design

### Store Interface

Extend `KnowledgeStore` with parent operations:

```python
def add_document(
    self,
    document: DocumentRecord,
    chunks: list[ChunkRecord],
    parent_blocks: list[ParentBlockRecord] | None = None,
) -> None:
    ...

def get_parent_block(self, parent_id: str) -> ParentBlockRecord | None:
    ...

def get_parent_blocks(self, document_id: str | None = None) -> list[ParentBlockRecord]:
    ...

def get_child_chunks_for_parent(self, parent_id: str) -> list[ChunkRecord]:
    ...
```

The optional `parent_blocks` argument keeps old tests and call sites source-compatible.

### SQLite Schema

Add a new table:

```sql
CREATE TABLE IF NOT EXISTS knowledge_parent_blocks (
    parent_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    document_title TEXT NOT NULL,
    parent_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    tokens_json TEXT NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER,
    metadata_json TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES knowledge_documents(document_id)
);
```

Extend `knowledge_chunks`:

```sql
ALTER TABLE knowledge_chunks ADD COLUMN parent_id TEXT;
ALTER TABLE knowledge_chunks ADD COLUMN parent_index INTEGER;
ALTER TABLE knowledge_chunks ADD COLUMN start_offset INTEGER;
ALTER TABLE knowledge_chunks ADD COLUMN end_offset INTEGER;
```

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_knowledge_parent_blocks_document_index
ON knowledge_parent_blocks(document_id, parent_index);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_parent_id
ON knowledge_chunks(parent_id);
```

Migration should be idempotent. For existing databases, old chunks remain valid and can still be searched through the legacy path until documents are reindexed.

## Chunking Design

Create `app/rag/hierarchical_chunking.py` with two public functions:

```python
def split_parent_blocks(
    content: str,
    parent_size: int = 1600,
    parent_overlap: int = 160,
) -> list[TextBlock]:
    ...


def split_child_chunks(
    parent_block: TextBlock,
    child_size: int = 450,
    child_overlap: int = 80,
) -> list[TextBlock]:
    ...
```

`TextBlock` is a small dataclass:

```python
@dataclass(frozen=True)
class TextBlock:
    content: str
    index: int
    start_offset: int | None = None
    end_offset: int | None = None
```

Splitting rules:

- Prefer paragraph boundaries.
- Preserve headings with following paragraphs when possible.
- Split oversized paragraphs by character windows with overlap.
- Do not emit empty blocks.
- Parent blocks are used for final LLM context.
- Child chunks are used for vector and BM25 recall.

## Indexing Flow

`DocumentService.create_document` gains a configurable hierarchical mode:

1. Build `DocumentRecord` as today.
2. If `rag_indexing_mode == "flat"`:
   - Use existing `chunk_text`.
3. If `rag_indexing_mode == "hierarchical"`:
   - Create parent blocks.
   - Create child chunks per parent.
   - Store both parent blocks and child chunks.
   - Embed only child chunks.
   - Mark child chunks with embedding signature.
4. If embedding fails:
   - Roll back child vectors.
   - Preserve document status failure behavior.

Recommended settings:

```python
rag_indexing_mode: str = "hierarchical"
parent_chunk_size: int = 1600
parent_chunk_overlap: int = 160
child_chunk_size: int = 450
child_chunk_overlap: int = 80
```

The first implementation can default to `flat` if minimizing rollout risk is preferred. The design recommends `hierarchical` after tests and benchmark evidence pass.

## BM25 Design

Create `app/rag/bm25.py`.

Use `rank-bm25` as an optional dependency:

```toml
rag-advanced = [
  "rank-bm25>=0.2.2",
  "sentence-transformers>=5.4.1",
]
```

The BM25 adapter should hide the third-party library:

```python
class Bm25Index:
    def __init__(self, chunks: list[ChunkRecord]) -> None:
        ...

    def search(self, query: str, top_k: int) -> list[Bm25Match]:
        ...
```

`Bm25Match`:

```python
@dataclass(frozen=True)
class Bm25Match:
    chunk_id: str
    score: float
    rank: int
```

Index lifecycle:

- In-memory BM25 index is rebuilt lazily from `KnowledgeStore.get_chunks()`.
- A version counter or chunk-count fingerprint invalidates the index after ingestion or reindex.
- Persistence is not required in the first implementation because BM25 rebuild is cheap for the current project scale.

If `rank-bm25` is unavailable:

- The advanced strategy should raise a clear configuration error.
- The legacy `lexical`, `vector`, and `hybrid` strategies continue to work.

## Vector Retrieval Design

The default vector backend remains Qdrant Local or the configured `VectorStore`.

FAISS support is optional and should be implemented as a separate vector store backend:

```python
class FaissVectorStore(VectorStore):
    backend_name = "faiss"
```

Recommended config:

```python
vector_store_provider: str = "qdrant_local"
faiss_index_path: str = "./data/faiss.index"
faiss_id_map_path: str = "./data/faiss_ids.json"
```

FAISS behavior:

- Store vectors in an inner-product or cosine-compatible index.
- Maintain a durable chunk id map.
- Support `clear`, `upsert`, `delete`, `replace_all`, `count`, and `search`.
- Treat FAISS as a local performance option, not a required part of B-phase success.

Because the project already has a working vector store abstraction, parent-child retrieval should target `VectorStore` first. FAISS can be implemented after the parent-child pipeline is stable.

## Parent Aggregation Design

Create `app/rag/parent_aggregation.py`.

Inputs:

- Vector matches over child chunks.
- BM25 matches over child chunks.
- Store access for chunk and parent lookup.

Output:

- Ordered parent candidates with evidence.

Aggregation formula:

```text
parent_score =
  vector_rrf_weight * max_child_vector_rrf +
  bm25_rrf_weight * max_child_bm25_rrf +
  evidence_bonus * min(distinct_evidence_children, 3)
```

Recommended defaults:

```python
advanced_vector_weight: float = 0.55
advanced_bm25_weight: float = 0.45
advanced_evidence_bonus: float = 0.01
advanced_rrf_constant: int = 60
```

Rules:

- One parent appears once in final candidates.
- Keep top evidence child ids per parent.
- Preserve both original child ranks and source signals for debugging.
- If a child has no parent id, treat the child itself as a synthetic parent for backward compatibility.

## Reranker Design

Create `app/rag/reranker.py`.

Interface:

```python
class Reranker(Protocol):
    backend_name: str

    def rerank(self, query: str, candidates: list[ParentCandidate], top_k: int) -> list[RerankResult]:
        ...
```

Implementations:

- `NoopReranker`: returns parent candidates ordered by aggregation score.
- `CrossEncoderReranker`: uses `sentence_transformers.CrossEncoder`.

Recommended settings:

```python
reranker_provider: str = "none"
reranker_model: str = "BAAI/bge-reranker-base"
reranker_device: str = "cpu"
reranker_batch_size: int = 8
reranker_max_candidates: int = 20
reranker_fallback_enabled: bool = True
```

Failure behavior:

- If reranker is disabled, use aggregated parent order.
- If reranker import/model loading fails and fallback is enabled, log warning and use `NoopReranker`.
- If fallback is disabled, fail search with a clear error.
- Do not silently change document indexing status because reranker failure is query-time behavior.

Score handling:

- Reranker score becomes primary final score when reranker is active.
- Aggregated score remains available as `aggregated_child_score`.
- Retrieval mode should be `parent_child_rerank` or `parent_child_hybrid`.

## Advanced Retriever Design

Create `app/rag/advanced_retriever.py` rather than expanding `KnowledgeRetriever` into a large multi-purpose class.

```python
class ParentChildRetriever:
    def __init__(
        self,
        store: KnowledgeStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        bm25_provider: Bm25Provider,
        reranker: Reranker,
        settings: AdvancedRetrievalSettings,
    ) -> None:
        ...

    def search(self, query: str, top_k: int = 3) -> list[ParentSearchHit]:
        ...
```

Search flow:

1. Normalize query.
2. Vector search child chunks with `top_k = max(top_k * vector_multiplier, min_child_candidates)`.
3. BM25 search child chunks with `top_k = max(top_k * bm25_multiplier, min_child_candidates)`.
4. Merge child candidates.
5. Aggregate to parent candidates.
6. Trim to `reranker_max_candidates`.
7. Rerank parent candidates.
8. Return top parent hits.

Recommended defaults:

```python
advanced_child_vector_top_k_multiplier: int = 8
advanced_child_bm25_top_k_multiplier: int = 8
advanced_min_child_candidates: int = 24
advanced_parent_candidate_limit: int = 20
```

## API Compatibility

Keep `/api/documents/search` compatible:

- Existing query params remain:
  - `query`
  - `top_k`
- Add optional `strategy`:
  - `lexical`
  - `vector`
  - `hybrid`
  - `parent_child`
  - `parent_child_rerank`

Default strategy is controlled by setting:

```python
default_retrieval_strategy: str = "hybrid"
```

The `search_knowledge_base` tool should also accept an optional strategy argument. If not provided, it uses the default.

Search result content:

- Legacy strategies return chunk content as today.
- Parent-child strategies return parent block content.
- Metadata fields can include evidence child ids and reranker score.

## Configuration

Add settings:

```python
rag_indexing_mode: str = "flat"
default_retrieval_strategy: str = "hybrid"
parent_chunk_size: int = 1600
parent_chunk_overlap: int = 160
child_chunk_size: int = 450
child_chunk_overlap: int = 80
bm25_enabled: bool = True
bm25_top_k_multiplier: int = 8
advanced_vector_top_k_multiplier: int = 8
advanced_parent_candidate_limit: int = 20
reranker_provider: str = "none"
reranker_model: str = "BAAI/bge-reranker-base"
reranker_device: str = "cpu"
reranker_batch_size: int = 8
reranker_fallback_enabled: bool = True
```

The first implementation should keep defaults conservative:

- `rag_indexing_mode="flat"`
- `default_retrieval_strategy="hybrid"`
- `reranker_provider="none"`

After benchmark evidence is generated, the recommended local profile can set:

- `rag_indexing_mode="hierarchical"`
- `default_retrieval_strategy="parent_child_rerank"`
- `reranker_provider="cross_encoder"`

## Dependency Plan

Base dependencies should not include reranker or FAISS packages.

Add optional groups:

```toml
rag-advanced = [
  "rank-bm25>=0.2.2",
]

reranker = [
  "sentence-transformers>=5.4.1",
]

faiss = [
  "faiss-cpu>=1.8.0",
]
```

If dependency installation is difficult on Windows, implementation can complete parent-child + BM25 + NoopReranker first, then add CrossEncoder and FAISS as optional follow-up tasks.

## Reindexing and Migration

Existing documents indexed in flat mode cannot automatically provide parent context unless reindexed.

Migration behavior:

- Existing flat documents remain searchable through old strategies.
- Running `/api/documents/reindex` with hierarchical mode rebuilds:
  - parent blocks
  - child chunks
  - child embeddings
  - BM25 in-memory index on next query
- `DocumentSummary.chunk_count` continues counting child chunks.
- Add `parent_count` later only if UI needs it.

Reindex should clear old child vectors when replacing all vectors. Parent blocks have no vectors.

## Evaluation Design

Extend `scripts/evaluate_retrieval.py`.

Strategies:

- `lexical`
- `vector`
- `hybrid`
- `parent_child`
- `parent_child_rerank`

Additional metrics:

- Hit@1
- Recall@3
- Recall@5
- MRR@3
- P50 latency
- P95 latency
- Average parent candidate count
- Average evidence child count
- Reranker enabled/disabled

Benchmark output should clearly state:

- Corpus size
- Query count
- Embedding model
- Vector backend
- Indexing mode
- Reranker provider/model
- Whether results are local evaluation, not production traffic

Do not report a fixed percentage improvement until measured by this script.

## Testing Strategy

### Unit Tests

Add tests for:

- Parent splitting keeps non-empty blocks and respects target size.
- Child splitting maps every child to the correct parent id.
- Store implementations persist and reload parent blocks.
- BM25 index ranks exact keyword matches above unrelated chunks.
- Parent aggregation deduplicates multiple child hits from the same parent.
- Parent aggregation preserves evidence chunk ids.
- Noop reranker preserves aggregate order.
- CrossEncoder reranker can be tested with a fake model object.
- Advanced retriever falls back when reranker is unavailable and fallback is enabled.

### Integration Tests

Add tests for:

- Hierarchical document ingestion populates parent blocks, child chunks, and child vectors.
- `/api/documents/search?strategy=parent_child` returns parent content.
- `search_knowledge_base` tool can use parent-child strategy.
- Reindex in hierarchical mode rebuilds the same number of child vectors as stored child chunks.

### Regression Tests

Existing tests for:

- text upload
- PDF/DOCX/image OCR upload
- flat search
- vector store fallback
- chat using uploaded document context

must continue to pass.

## Rollout Plan

1. Add settings and data models with no behavior change.
2. Add hierarchical chunking tests and implementation.
3. Extend in-memory and SQLite knowledge stores for parent blocks.
4. Add hierarchical indexing behind `rag_indexing_mode`.
5. Add BM25 adapter and tests.
6. Add parent aggregation.
7. Add `ParentChildRetriever` with `NoopReranker`.
8. Wire API/tool strategy selection.
9. Add CrossEncoder reranker behind optional config.
10. Extend benchmark script and publish local ablation results.
11. Add optional FAISS backend after parent-child retrieval is stable.

## Risks and Mitigations

- **Large local model cost:** CrossEncoder can slow CPU queries. Mitigate with `reranker_max_candidates`, batching, and `NoopReranker` fallback.
- **Dependency friction on Windows:** Keep `rank-bm25`, reranker, and FAISS optional. Implement pure-Python fallbacks where practical.
- **Schema migration risk:** Make SQLite migrations idempotent and keep legacy flat documents searchable.
- **Retriever complexity:** Keep `KnowledgeRetriever` as legacy and introduce `ParentChildRetriever` as a separate class.
- **Benchmark overclaiming:** Report only measured local benchmark deltas. Avoid production-style claims without traffic data.

## Acceptance Criteria

- Existing flat retrieval tests still pass.
- Hierarchical ingestion can create parent blocks and child chunks.
- Child chunks carry parent ids.
- Vector retrieval searches child chunks.
- BM25 retrieval searches child chunks.
- Parent aggregation returns one hit per parent with evidence child ids.
- Parent-child search returns parent content.
- Reranker is optional and can be disabled.
- CrossEncoder reranker can rerank parent candidates when optional dependencies are installed.
- Benchmark script compares old hybrid retrieval with parent-child and parent-child-rerank strategies.
- No production metric claims are added without measured evidence.

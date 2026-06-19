# RAG Parent Child Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement parent-child RAG indexing and retrieval with optional parent reranking while preserving legacy flat retrieval behavior.

**Architecture:** Keep `KnowledgeRetriever` as the compatibility entrypoint and add a separate parent-child retrieval pipeline. Store parent blocks alongside child chunks, search child chunks through vector and BM25-style lexical recall, aggregate to parents, and optionally rerank parent candidates before returning `SearchHit`-compatible results.

**Tech Stack:** Python 3.11, Pydantic v2, FastAPI, SQLite, in-memory test stores, existing embedding and vector-store abstractions.

---

### Task 1: Data Models and Hierarchical Chunking

**Files:**
- Modify: `app/rag/models.py`
- Create: `app/rag/hierarchical_chunking.py`
- Test: `tests/test_hierarchical_chunking.py`

- [ ] **Step 1: Write failing tests for parent and child splitting**

```python
from app.rag.hierarchical_chunking import split_child_chunks, split_parent_blocks


def test_parent_splitter_prefers_paragraph_boundaries() -> None:
    content = "\n\n".join(
        [
            "# Guide\nIntro paragraph about deployment.",
            "Second paragraph covers health checks.",
            "Third paragraph covers rollback steps.",
        ]
    )

    blocks = split_parent_blocks(content, parent_size=70, parent_overlap=10)

    assert len(blocks) >= 2
    assert all(block.content.strip() for block in blocks)
    assert blocks[0].start_offset == 0
    assert blocks[0].end_offset is not None


def test_child_splitter_uses_parent_relative_offsets() -> None:
    parent = split_parent_blocks("alpha beta gamma delta epsilon zeta eta theta", parent_size=100)[0]

    children = split_child_chunks(parent, child_size=18, child_overlap=5)

    assert len(children) > 1
    assert children[0].start_offset == parent.start_offset
    assert all(child.start_offset is not None and child.end_offset is not None for child in children)
```

- [ ] **Step 2: Run tests and verify missing module failure**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_hierarchical_chunking.py -q`

Expected: FAIL because `app.rag.hierarchical_chunking` does not exist.

- [ ] **Step 3: Implement models and splitters**

Add `ParentBlockRecord` and parent metadata fields on `ChunkRecord`; add `ParentSearchHit` extending `SearchHit`. Implement `TextBlock`, `split_parent_blocks`, and `split_child_chunks` with paragraph-aware grouping and window fallback for oversized text.

- [ ] **Step 4: Run tests and verify pass**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_hierarchical_chunking.py -q`

Expected: PASS.

### Task 2: Parent Block Store Persistence

**Files:**
- Modify: `app/rag/store.py`
- Modify: `app/persistence/knowledge_repository.py`
- Test: `tests/test_knowledge_store.py`

- [ ] **Step 1: Write failing SQLite persistence test**

```python
from app.rag.models import ParentBlockRecord


def test_sqlite_knowledge_store_persists_parent_blocks(tmp_path) -> None:
    database_path = tmp_path / "knowledge_base.db"
    store = SqliteKnowledgeStore(str(database_path))
    document = DocumentRecord(title="manual", content="Parent content", metadata={})
    parent = ParentBlockRecord(
        document_id=document.document_id,
        document_title=document.title,
        index=0,
        content="Parent content",
        tokens=["parent", "content"],
        start_offset=0,
        end_offset=14,
    )
    child = ChunkRecord(
        document_id=document.document_id,
        document_title=document.title,
        index=0,
        content="Parent",
        tokens=["parent"],
        parent_id=parent.parent_id,
        parent_index=0,
        start_offset=0,
        end_offset=6,
    )

    store.add_document(document, [child], parent_blocks=[parent])
    reopened = SqliteKnowledgeStore(str(database_path))

    assert reopened.get_parent_block(parent.parent_id) == parent
    assert reopened.get_parent_blocks(document.document_id) == [parent]
    assert reopened.get_child_chunks_for_parent(parent.parent_id)[0].chunk_id == child.chunk_id
```

- [ ] **Step 2: Run targeted store test and verify signature failure**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_knowledge_store.py::test_sqlite_knowledge_store_persists_parent_blocks -q`

Expected: FAIL because `add_document` does not accept `parent_blocks` and parent accessors do not exist.

- [ ] **Step 3: Extend stores**

Update the protocol, in-memory store, SQLite schema migration, row mappers, and optional Postgres SQL to support parent block CRUD and chunk parent metadata.

- [ ] **Step 4: Run targeted store tests**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_knowledge_store.py -q`

Expected: PASS.

### Task 3: Hierarchical Ingestion

**Files:**
- Modify: `app/core/config.py`
- Modify: `app/services/document_service.py`
- Test: `tests/test_document_service.py`

- [ ] **Step 1: Write failing hierarchical ingestion test**

```python
def test_create_document_can_index_hierarchical_parent_child_chunks() -> None:
    store = InMemoryKnowledgeStore()
    vector_store = InMemoryVectorStore()
    embedding_provider = HashEmbeddingProvider(dimensions=8)
    retriever = KnowledgeRetriever(store, vector_store, embedding_provider)
    service = DocumentService(
        store=store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        retriever=retriever,
        indexing_mode="hierarchical",
        parent_chunk_size=90,
        parent_chunk_overlap=10,
        child_chunk_size=35,
        child_chunk_overlap=5,
    )

    summary = service.create_document(
        DocumentCreateRequest(
            title="ops",
            content="Deployment setup paragraph.\n\nHealth checks paragraph.\n\nRollback paragraph.",
            metadata={},
        )
    )

    parents = store.get_parent_blocks(summary.document_id)
    chunks = store.get_document_chunks(summary.document_id)
    assert parents
    assert chunks
    assert all(chunk.parent_id for chunk in chunks)
    assert vector_store.count() == len(chunks)
```

- [ ] **Step 2: Run targeted test and verify constructor failure**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_document_service.py::test_create_document_can_index_hierarchical_parent_child_chunks -q`

Expected: FAIL because `DocumentService` has no hierarchical settings.

- [ ] **Step 3: Implement hierarchical indexing path**

Add conservative settings defaults, pass them from dependency construction, create parent blocks plus child chunks in hierarchical mode, and keep the flat path unchanged.

- [ ] **Step 4: Run document service tests**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_document_service.py -q`

Expected: PASS.

### Task 4: BM25, Aggregation, and Reranker Units

**Files:**
- Create: `app/rag/bm25.py`
- Create: `app/rag/parent_aggregation.py`
- Create: `app/rag/reranker.py`
- Test: `tests/test_parent_child_retrieval_units.py`

- [ ] **Step 1: Write failing unit tests**

```python
from app.rag.bm25 import Bm25Index
from app.rag.parent_aggregation import aggregate_parent_candidates
from app.rag.reranker import NoopReranker


def test_bm25_ranks_exact_keyword_match_first() -> None:
    chunks = [
        ChunkRecord(document_id="d1", document_title="Doc 1", index=0, content="alpha beta", tokens=["alpha", "beta"]),
        ChunkRecord(document_id="d2", document_title="Doc 2", index=0, content="gamma delta", tokens=["gamma", "delta"]),
    ]

    matches = Bm25Index(chunks).search("alpha", top_k=2)

    assert matches[0].chunk_id == chunks[0].chunk_id


def test_parent_aggregation_deduplicates_parent_and_keeps_evidence() -> None:
    store = InMemoryKnowledgeStore()
    document = DocumentRecord(title="guide", content="Parent body", metadata={})
    parent = ParentBlockRecord(document_id=document.document_id, document_title=document.title, index=0, content="Parent body")
    chunks = [
        ChunkRecord(document_id=document.document_id, document_title=document.title, index=0, content="first", parent_id=parent.parent_id),
        ChunkRecord(document_id=document.document_id, document_title=document.title, index=1, content="second", parent_id=parent.parent_id),
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
    assert {child_id for child_id in candidates[0].evidence_chunk_ids} == {chunks[0].chunk_id, chunks[1].chunk_id}


def test_noop_reranker_preserves_aggregate_order() -> None:
    candidates = [
        ParentCandidate(parent_id="p1", document_id="d", document_title="d", parent_index=0, content="one", score=0.2),
        ParentCandidate(parent_id="p2", document_id="d", document_title="d", parent_index=1, content="two", score=0.5),
    ]

    results = NoopReranker().rerank("query", candidates, top_k=1)

    assert results[0].parent_id == "p2"
```

- [ ] **Step 2: Run unit tests and verify import failures**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_parent_child_retrieval_units.py -q`

Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement units**

Use `rank-bm25` when installed and a deterministic local BM25 formula otherwise. Implement reciprocal-rank aggregation and `NoopReranker`; implement `CrossEncoderReranker` with import-time fallback behavior.

- [ ] **Step 4: Run unit tests**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_parent_child_retrieval_units.py -q`

Expected: PASS.

### Task 5: ParentChildRetriever and Strategy Wiring

**Files:**
- Create: `app/rag/advanced_retriever.py`
- Modify: `app/rag/retriever.py`
- Modify: `app/api/dependencies.py`
- Modify: `app/api/routes/documents.py`
- Modify: `app/tools/builtins/knowledge_base.py`
- Test: `tests/test_parent_child_retriever.py`
- Test: `tests/test_documents.py`

- [ ] **Step 1: Write failing integration tests**

```python
def test_parent_child_search_returns_parent_content() -> None:
    store = InMemoryKnowledgeStore()
    vector_store = InMemoryVectorStore()
    embedding_provider = HashEmbeddingProvider(dimensions=8)
    retriever = KnowledgeRetriever(store, vector_store, embedding_provider)
    service = DocumentService(
        store=store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        retriever=retriever,
        indexing_mode="hierarchical",
        parent_chunk_size=120,
        child_chunk_size=35,
        child_chunk_overlap=0,
    )
    service.create_document(
        DocumentCreateRequest(
            title="incident-guide",
            content="Startup diagnostics include checking logs and environment variables before restarting services.",
            metadata={},
        )
    )

    hits = service.search("environment variables", top_k=1, strategy="parent_child")

    assert hits[0].retrieval_mode == "parent_child_hybrid"
    assert "Startup diagnostics" in hits[0].content
    assert hits[0].evidence_chunk_ids
```

- [ ] **Step 2: Run integration test and verify unsupported strategy failure**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_parent_child_retriever.py -q`

Expected: FAIL because parent-child strategy is not wired.

- [ ] **Step 3: Implement advanced retriever and routing**

Add advanced settings, strategy dispatch in `KnowledgeRetriever`, `DocumentService.search(strategy=...)`, `/api/documents/search?strategy=...`, and `search_knowledge_base` optional `strategy`.

- [ ] **Step 4: Run RAG and API tests**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_parent_child_retriever.py tests/test_vector_rag.py tests/test_documents.py -q`

Expected: PASS.

### Task 6: Benchmark Strategy Expansion

**Files:**
- Modify: `scripts/evaluate_retrieval.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add benchmark strategies and metadata**

Allow `parent_child` and `parent_child_rerank` in `--strategies`; expose `--indexing-mode`; include Recall@5, candidate/evidence averages, indexing mode, reranker provider, and vector backend in JSON/Markdown output.

- [ ] **Step 2: Add optional extras**

Add `rag-advanced`, `reranker`, and `faiss` optional dependency groups without moving those packages into base dependencies.

- [ ] **Step 3: Run a syntax-level benchmark check**

Run: `.venv312\Scripts\python.exe -m compileall app scripts`

Expected: PASS.

### Task 7: Final Verification

**Files:**
- All files changed in Tasks 1-6

- [ ] **Step 1: Run targeted tests**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_hierarchical_chunking.py tests/test_parent_child_retrieval_units.py tests/test_parent_child_retriever.py tests/test_knowledge_store.py tests/test_document_service.py tests/test_vector_rag.py -q`

Expected: PASS.

- [ ] **Step 2: Run broader existing suite**

Run: `.venv312\Scripts\python.exe -m pytest tests/test_documents.py tests/test_tools_api.py tests/test_document_extractor_tool.py tests/test_reindex_job_service.py -q`

Expected: PASS.

- [ ] **Step 3: Inspect diff**

Run: `git -c core.quotepath=false diff --stat`

Expected: Diff is limited to RAG models, stores, retrieval, document service, API/tool wiring, benchmark, optional dependencies, tests, and this plan.

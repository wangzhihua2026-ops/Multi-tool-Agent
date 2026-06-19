from app.rag.models import ParentSearchHit, SearchHit
from scripts.evaluate_retrieval import evaluate_strategy, render_markdown


class FakeRetriever:
    def search(self, query: str, top_k: int, strategy: str):
        del top_k, strategy
        if query == "alpha":
            return [
                ParentSearchHit(
                    document_id="doc-x",
                    document_title="doc-x",
                    chunk_id="p-x",
                    chunk_index=0,
                    content="wrong",
                    score=0.9,
                    parent_id="p-x",
                    parent_index=0,
                    evidence_chunk_ids=["c1", "c2"],
                ),
                ParentSearchHit(
                    document_id="doc-a",
                    document_title="doc-a",
                    chunk_id="p-a",
                    chunk_index=0,
                    content="right",
                    score=0.8,
                    parent_id="p-a",
                    parent_index=0,
                    evidence_chunk_ids=["c3"],
                ),
            ]
        return [
            SearchHit(
                document_id="doc-z",
                document_title="doc-z",
                chunk_id="c-z",
                chunk_index=0,
                content="unrelated",
                score=0.4,
            )
        ]


def test_evaluate_strategy_reports_parent_child_metrics() -> None:
    metric = evaluate_strategy(
        retriever=FakeRetriever(),
        queries=[
            {
                "query_id": "q1",
                "query": "alpha",
                "category": "demo",
                "relevant_document_ids": ["doc-a"],
            },
            {
                "query_id": "q2",
                "query": "missing",
                "category": "demo",
                "relevant_document_ids": ["doc-b"],
            },
        ],
        strategy="parent_child",
        top_k=5,
        latency_repeats=1,
    )

    assert metric["strategy"] == "parent_child"
    assert metric["recall_at_5"] == 0.5
    assert metric["average_parent_candidate_count"] == 1.5
    assert metric["average_evidence_child_count"] == 1.0


def test_render_markdown_includes_recall_at_5_and_parent_metrics() -> None:
    markdown = render_markdown(
        {
            "run_date": "2026-06-19",
            "corpus_documents": 1,
            "labelled_queries": 1,
            "embedding_signature": "hash:8",
            "vector_store": "qdrant_local",
            "indexing_mode": "hierarchical",
            "reranker_provider": "none",
            "reranker_model": "BAAI/bge-reranker-base",
            "latency_repeats": 1,
            "metrics": [
                {
                    "strategy": "parent_child",
                    "hit_at_1": 1.0,
                    "recall_at_3": 1.0,
                    "recall_at_5": 1.0,
                    "mrr_at_3": 1.0,
                    "average_parent_candidate_count": 1.0,
                    "average_evidence_child_count": 2.0,
                    "latency_ms": {"p50": 1.0, "p95": 2.0},
                }
            ],
        }
    )

    assert "Recall@5" in markdown
    assert "Avg parents" in markdown
    assert "Indexing mode: `hierarchical`" in markdown

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.rag.advanced_retriever import AdvancedRetrievalSettings
from app.rag.embeddings import build_embedding_provider
from app.rag.models import DocumentCreateRequest
from app.rag.retriever import KnowledgeRetriever
from app.rag.reranker import build_reranker
from app.rag.store import InMemoryKnowledgeStore
from app.rag.vector_store import QdrantLocalVectorStore
from app.services.document_service import DocumentService


DEFAULT_CORPUS = ROOT / "evaluation" / "retrieval_corpus.json"
DEFAULT_QUERIES = ROOT / "evaluation" / "retrieval_queries.json"


def main() -> None:
    args = parse_args()
    corpus = load_json(args.corpus)
    queries = load_json(args.queries)
    settings = build_settings(args)

    with tempfile.TemporaryDirectory(prefix="multi-tool-agent-eval-") as temporary_directory:
        provider = build_embedding_provider(settings)
        store = InMemoryKnowledgeStore()
        vector_store = QdrantLocalVectorStore(
            storage_path=temporary_directory,
            collection_name="retrieval_benchmark",
            dimensions=settings.embedding_dimensions,
        )
        retriever = KnowledgeRetriever(
            store,
            vector_store,
            provider,
            advanced_settings=AdvancedRetrievalSettings(
                bm25_enabled=True,
                vector_top_k_multiplier=8,
                bm25_top_k_multiplier=8,
                min_child_candidates=max(args.top_k * 8, 24),
                parent_candidate_limit=20,
            ),
            reranker=build_reranker(
                provider=args.reranker_provider,
                model_name=args.reranker_model,
                device="cpu",
                batch_size=8,
                fallback_enabled=True,
            ),
        )
        service = DocumentService(
            store=store,
            vector_store=vector_store,
            embedding_provider=provider,
            retriever=retriever,
            indexing_mode=args.indexing_mode,
        )

        ingestion_start = time.perf_counter()
        for document in corpus:
            service.create_document(
                DocumentCreateRequest(
                    title=document["document_id"],
                    content=document["content"],
                    metadata={"category": document["category"], "display_title": document["title"]},
                )
            )
        ingestion_seconds = time.perf_counter() - ingestion_start

        # Load model paths and vector-store caches before latency sampling.
        retriever.search(queries[0]["query"], top_k=args.top_k, strategy=args.strategies[0])
        metrics = [
            evaluate_strategy(
                retriever=retriever,
                queries=queries,
                strategy=strategy,
                top_k=args.top_k,
                latency_repeats=args.latency_repeats,
            )
            for strategy in args.strategies
        ]
        vector_store.close()

    result = {
        "benchmark": "local_knowledge_retrieval_ablation",
        "run_date": date.today().isoformat(),
        "corpus_documents": len(corpus),
        "labelled_queries": len(queries),
        "top_k": args.top_k,
        "latency_repeats": args.latency_repeats,
        "embedding_signature": provider.embedding_signature,
        "vector_store": "qdrant_local",
        "indexing_mode": args.indexing_mode,
        "reranker_provider": args.reranker_provider,
        "reranker_model": args.reranker_model,
        "ingestion_seconds": round(ingestion_seconds, 3),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"JSON result: {args.output}")
    print(f"Markdown report: {args.report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate labelled local knowledge retrieval queries.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation" / "results" / "retrieval_benchmark_latest.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "evaluation" / "results" / "retrieval_benchmark_latest.md",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["lexical", "vector", "hybrid", "parent_child", "parent_child_rerank"],
        default=["lexical", "vector", "hybrid", "parent_child", "parent_child_rerank"],
    )
    parser.add_argument("--indexing-mode", choices=["flat", "hierarchical"], default="hierarchical")
    parser.add_argument("--reranker-provider", choices=["none", "cross_encoder"], default="none")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--embedding-dimensions", type=int, default=384)
    parser.add_argument("--embedding-cache-path", default=str(ROOT / "data" / "hf-cache"))
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> Settings:
    return Settings(
        embedding_provider="sentence_transformers",
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        embedding_device="cpu",
        embedding_cache_path=args.embedding_cache_path,
        embedding_local_files_only=True,
        embedding_fallback_enabled=False,
    )


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_strategy(
    retriever: KnowledgeRetriever,
    queries: list[dict[str, Any]],
    strategy: str,
    top_k: int,
    latency_repeats: int,
) -> dict[str, Any]:
    ranks: list[int | None] = []
    latencies_ms: list[float] = []
    parent_candidate_counts: list[int] = []
    evidence_child_total = 0
    returned_hit_total = 0
    per_query: list[dict[str, Any]] = []
    for query in queries:
        hits = retriever.search(query["query"], top_k=top_k, strategy=strategy)
        retrieved_document_ids = [hit.document_title for hit in hits]
        parent_candidate_counts.append(len(hits))
        returned_hit_total += len(hits)
        query_evidence_count = sum(
            len(getattr(hit, "evidence_chunk_ids", []) or [])
            for hit in hits
        )
        evidence_child_total += query_evidence_count
        relevant_document_ids = set(query["relevant_document_ids"])
        rank = next(
            (
                index
                for index, document_id in enumerate(retrieved_document_ids, start=1)
                if document_id in relevant_document_ids
            ),
            None,
        )
        ranks.append(rank)
        per_query.append(
            {
                "query_id": query["query_id"],
                "category": query["category"],
                "rank": rank,
                "retrieved_document_ids": retrieved_document_ids,
                "parent_candidate_count": len(hits),
                "evidence_child_count": query_evidence_count,
            }
        )
        for _ in range(latency_repeats):
            start = time.perf_counter()
            retriever.search(query["query"], top_k=top_k, strategy=strategy)
            latencies_ms.append((time.perf_counter() - start) * 1000)

    query_count = len(queries)
    return {
        "strategy": strategy,
        "hit_at_1": round(sum(rank == 1 for rank in ranks) / query_count, 4),
        "recall_at_3": round(sum(rank is not None and rank <= 3 for rank in ranks) / query_count, 4),
        "recall_at_5": round(sum(rank is not None and rank <= 5 for rank in ranks) / query_count, 4),
        "mrr_at_3": round(sum(1 / rank if rank is not None and rank <= 3 else 0 for rank in ranks) / query_count, 4),
        "average_parent_candidate_count": round(sum(parent_candidate_counts) / query_count, 2),
        "average_evidence_child_count": round(
            evidence_child_total / returned_hit_total if returned_hit_total else 0.0,
            2,
        ),
        "latency_ms": {
            "samples": len(latencies_ms),
            "p50": round(statistics.median(latencies_ms), 2),
            "p95": round(percentile(latencies_ms, 0.95), 2),
        },
        "per_query": per_query,
    }


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * proportion) - 1))
    return ordered[index]


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Retrieval Benchmark",
        "",
        f"- Date: {result['run_date']}",
        f"- Corpus: {result['corpus_documents']} manually authored knowledge snippets grounded in project features",
        f"- Queries: {result['labelled_queries']} manually labelled queries, one relevant snippet per query",
        f"- Embedding: `{result['embedding_signature']}`",
        f"- Vector store: `{result['vector_store']}`",
        f"- Indexing mode: `{result.get('indexing_mode', 'flat')}`",
        f"- Reranker: `{result.get('reranker_provider', 'none')}` / `{result.get('reranker_model', 'none')}`",
        f"- Latency: local CPU query time; {result['latency_repeats']} measured runs per query after warm-up",
        "",
        "| Strategy | Hit@1 | Recall@3 | Recall@5 | MRR@3 | Avg parents | Avg evidence | P50 latency | P95 latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in result["metrics"]:
        lines.append(
            (
                "| {strategy} | {hit:.1%} | {recall3:.1%} | {recall5:.1%} | {mrr:.1%} | "
                "{parents:.2f} | {evidence:.2f} | {p50:.2f} ms | {p95:.2f} ms |"
            ).format(
                strategy=metric["strategy"],
                hit=metric["hit_at_1"],
                recall3=metric["recall_at_3"],
                recall5=metric.get("recall_at_5", metric["recall_at_3"]),
                mrr=metric["mrr_at_3"],
                parents=metric.get("average_parent_candidate_count", 0.0),
                evidence=metric.get("average_evidence_child_count", 0.0),
                p50=metric["latency_ms"]["p50"],
                p95=metric["latency_ms"]["p95"],
            )
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This is a small local retrieval ablation for the project's knowledge-base scenario. "
            "It validates ranking behavior on labelled queries; it is not a production traffic benchmark "
            "or an end-to-end answer-quality evaluation.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            ".\\.venv312\\Scripts\\python.exe -X utf8 .\\scripts\\evaluate_retrieval.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

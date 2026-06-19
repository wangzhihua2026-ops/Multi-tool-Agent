# Retrieval Benchmark

- Date: 2026-06-15
- Corpus: 12 manually authored knowledge snippets grounded in project features
- Queries: 30 manually labelled queries, one relevant snippet per query
- Embedding: `sentence_transformers:intfloat/multilingual-e5-small:384`
- Vector store: `qdrant_local`
- Latency: local CPU query time; 3 measured runs per query after warm-up

| Strategy | Hit@1 | Recall@3 | MRR@3 | P50 latency | P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical | 80.0% | 96.7% | 87.8% | 0.06 ms | 0.16 ms |
| vector | 90.0% | 96.7% | 93.3% | 12.07 ms | 14.22 ms |
| hybrid | 93.3% | 100.0% | 96.1% | 12.24 ms | 14.12 ms |

## Scope

This is a small local retrieval ablation for the project's knowledge-base scenario. It validates ranking behavior on labelled queries; it is not a production traffic benchmark or an end-to-end answer-quality evaluation.

## Reproduce

```powershell
.\.venv312\Scripts\python.exe -X utf8 .\scripts\evaluate_retrieval.py
```

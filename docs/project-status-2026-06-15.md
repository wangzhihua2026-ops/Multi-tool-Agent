# Project Status - 2026-06-15

## Current Positioning

Project_Loverboy01 is now best described as:

> A local-first enterprise knowledge-base Agent system with FastAPI + SSE, hybrid RAG, bounded tool calling, human approval, persisted traces, reproducible evaluation, and container/CI delivery assets.

The project is no longer just an Agent demo. It has enough engineering evidence to support a resume and interview story, as long as the remaining gaps are stated honestly.

## Verified Baseline

Command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-baseline.ps1 -DemoPort 8016
```

Results:

- Test suite: `78 passed in 3.64s`
- Demo path: document upload, search, reindex, and SSE chat passed
- Planner provider for demo: `mock`
- Retrieval benchmark: included in the baseline script

## Retrieval Results

Dataset:

- Corpus: `12` manually authored project knowledge snippets
- Queries: `30` manually labelled queries
- Embedding: `sentence_transformers:intfloat/multilingual-e5-small:384`
- Vector store: `qdrant_local`

| Strategy | Hit@1 | Recall@3 | MRR@3 | P50 latency | P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical | 80.0% | 96.7% | 87.8% | 0.06 ms | 0.16 ms |
| vector | 90.0% | 96.7% | 93.3% | 12.07 ms | 14.22 ms |
| hybrid | 93.3% | 100.0% | 96.1% | 12.24 ms | 14.12 ms |

Report:

- `evaluation/results/retrieval_benchmark_latest.md`
- `evaluation/results/retrieval_benchmark_latest.json`

## Delivery Assets

Completed:

- `Dockerfile`: builds the FastAPI app image with local-first defaults.
- `docker-compose.yml`: runs the app at `http://127.0.0.1:8000/app/` with persisted SQLite data volume.
- `.github/workflows/ci.yml`: installs dependencies, runs tests, builds the Docker image, starts the container, and checks `/api/health`.
- `scripts/verify-local-embeddings.ps1`: verifies upload, search, reindex, and chat.
- `scripts/check-baseline.ps1`: wraps tests, demo verification, and retrieval benchmark.
- `docs/baseline-audit-2026-06-15.md`: records the current verified evidence chain.
- `docs/internship-project-prep.md`: keeps interview bullets and Q&A aligned with current metrics.

Known limitation:

- Docker is not installed on the current machine, so Docker build verification must run in GitHub Actions or another Docker-enabled environment.

## Security Status

- Root `.env` exists for local development and contains a non-empty `LLM_API_KEY`.
- `.gitignore` excludes `.env`, local data, virtualenvs, caches, and build outputs.
- Git index inspection found `.env` is not tracked.
- Source/document scan found variable names and test tokens, but no obvious real secret values in tracked source files.

Before publishing:

- Confirm `.env`, `data/`, `.venv/`, `.venv312/`, `.pytest_cache/`, and local generated files are not staged.
- Avoid committing `main_verify.pdf` unless it is intentionally part of the project story.

## Resume-Ready Claims

Safe to claim:

- Built a FastAPI + SSE knowledge-base Agent with document upload, RAG retrieval, tool calling, human approval, and run trace persistence.
- Implemented lexical + vector + weighted RRF hybrid retrieval with Qdrant Local and sentence-transformers.
- Built a reproducible retrieval benchmark and reported Hit@1, Recall@3, MRR@3, and P95 latency.
- Added a one-command baseline script covering tests, demo verification, and retrieval benchmark.
- Added Docker/Compose and GitHub Actions workflow for reproducible delivery.

Do not overclaim yet:

- Production cloud deployment.
- Large-scale or real-user benchmark.
- End-to-end answer groundedness evaluation.
- Reranker-backed retrieval.
- stdio MCP support.

## Next Slice

1. Capture Web UI screenshot or short demo recording.
2. Push to GitHub and verify GitHub Actions passes.
3. Expand benchmark to `80-100` queries and `40-60` snippets with difficult negatives.
4. Add answer-quality evaluation: citation coverage, groundedness, and unsupported-claim detection.
5. Decide whether to add a lightweight reranker interface after the evaluation expansion.

# Baseline Audit - 2026-06-15

## Summary

Project_Loverboy01 now has a runnable AI application foundation plus the first delivery evidence chain: FastAPI + SSE, bounded Agent runtime, hybrid RAG, Qdrant Local, approval flow, persisted traces, tests, retrieval benchmark assets, one-command demo verification, Docker packaging, and CI configuration.

The current gap is no longer basic packaging. The remaining gap is stronger external proof: larger RAG evaluation data, end-to-end answer-quality evaluation, screenshots or recorded demo evidence, and a real Docker/CI run after publishing to GitHub.

## Verified Results

- Baseline command: `powershell -ExecutionPolicy Bypass -File .\scripts\check-baseline.ps1 -DemoPort 8016`
- Full test suite: `78 passed in 3.64s`
- Demo verification: upload, search, reindex, and SSE chat all passed
- Retrieval benchmark command: included in `scripts/check-baseline.ps1`
- Retrieval benchmark date: `2026-06-15`
- Corpus: `12` snippets
- Queries: `30` labelled queries
- Embedding: `sentence_transformers:intfloat/multilingual-e5-small:384`
- Vector store: `qdrant_local`

| Strategy | Hit@1 | Recall@3 | MRR@3 | P50 latency | P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical | 80.0% | 96.7% | 87.8% | 0.06 ms | 0.16 ms |
| vector | 90.0% | 96.7% | 93.3% | 12.07 ms | 14.22 ms |
| hybrid | 93.3% | 100.0% | 96.1% | 12.24 ms | 14.12 ms |

## Demo Verification

The one-command verification script now passes:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify-local-embeddings.ps1 -Port 8012
```

- Server health check succeeded on localhost.
- Document upload succeeded.
- Search returned `3` hits.
- Reindex job completed.
- Chat streaming returned an assistant answer grounded in tool output.
- Planner provider used for the demo run: `mock`.

`scripts/verify-local-embeddings.ps1` now validates Python candidates before starting the server, skips broken migrated virtualenv executables, forces deterministic `LLM_PROVIDER=mock`, and polls `GET /api/documents/reindex/{job_id}` for the current reindex job summary.

`scripts/check-baseline.ps1` wraps the full local evidence chain: pytest, the demo verification script, and the retrieval benchmark.

## Docker and CI

- `Dockerfile` builds a lightweight app image with default mock/hash settings.
- `docker-compose.yml` runs the local demo at `http://127.0.0.1:8000/app/` with SQLite data persisted in a named volume.
- `.github/workflows/ci.yml` runs Python tests, builds the Docker image, starts the container, and checks `/api/health`.
- Docker is not installed in the current local workspace, so container build verification is delegated to GitHub Actions.

## Security Check

- Root `.env` exists and contains a non-empty `LLM_API_KEY`.
- `.gitignore` includes `.env`, `data/`, `.venv/`, `.venv312/`, caches, and build outputs.
- Git index inspection shows `.env` is not tracked.
- Source/document scan found variable names and test tokens, but no obvious real secret values in tracked source files.

## Current Delivery Status

- Reproducible local baseline is available through `scripts/check-baseline.ps1`.
- Demo verification is available through `scripts/verify-local-embeddings.ps1`.
- Retrieval benchmark outputs are refreshed under `evaluation/results/`.
- Docker and Compose files are present for local container startup.
- GitHub Actions workflow is present for tests, Docker build, container smoke test, and `/api/health` check.
- Root `.env` remains local-only and should not be committed.

## Remaining Gaps

- Expand retrieval evaluation from `30` labelled queries to `80-100` queries with difficult negatives.
- Expand corpus from `12` snippets to `40-60` snippets with near-duplicate distractors.
- Add end-to-end answer quality metrics, such as citation coverage and groundedness.
- Add screenshot or short recording evidence for the built-in web UI.
- Verify Docker build and GitHub Actions in an environment with Docker installed.
- Stage/commit the new delivery artifacts before publishing the repository.

## Recommended Next Slice

1. Capture a Web UI screenshot or short GIF of the demo path.
2. Add end-to-end QA evaluation for citation coverage and groundedness.
3. Expand the benchmark dataset and keep the latest report aligned with README and resume bullets.
4. Push to GitHub and confirm the CI workflow passes.

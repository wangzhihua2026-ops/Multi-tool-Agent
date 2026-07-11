# Agent Platform Verification

Verified locally on 2026-07-12 (Asia/Shanghai), branch `codex/agent-execution-platform`.

## Baseline

```powershell
$env:TEST_DATABASE_URL='postgresql+asyncpg://agent:agent@127.0.0.1:55432/agent'
$env:TEST_REDIS_URL='redis://127.0.0.1:56379/0'
E:\Project_Loverboy01\.venv312\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result: `182 passed in 13.54s`. PostgreSQL and Redis integration tests were enabled.

## Four-service Compose

`docker compose up -d --build` and `docker compose ps` showed `api`, `worker`, `postgres`, and `redis` healthy. The one-shot `migrate` service exited 0 after Alembic upgraded PostgreSQL. Worker logs showed successful `dispatch_outbox` and `recover_runs` jobs.

An initial worker start exposed missing ARQ `redis_settings` and attempted `localhost:6379`. A regression test and explicit `RedisSettings.from_dsn(REDIS_URL)` fixed it; the rebuilt worker remained healthy.

## 50-run concurrency evidence

An `httpx.AsyncClient` script submitted 50 concurrent `POST /api/runs` requests and polled `GET /api/runs/{id}` to a boundary state:

```text
submitted=50 terminal=50 pending=0 completed=50
submit_p50_ms=510.69 submit_p95_ms=532.97 batch_total_ms=5018.41
```

Environment: Windows Docker Desktop, mock LLM, hash embeddings, one ARQ worker with `max_jobs=8`. These are local demonstration measurements, not production capacity claims.

## Recovery and idempotency evidence

PostgreSQL integration tests verify duplicate claim rejection, approval compare-and-set, expired lease requeue exactly once with retained outbox, and cancellation while waiting for approval. Unit/integration tests also verify persisted plan checkpoint resume and replayable ordered SSE event IDs.

The deterministic Agent evaluation contains 30 cases and is explicitly marked `mock`; it demonstrates workflow repeatability, not real-model quality.

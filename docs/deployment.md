# Deployment Guide

## Local verified profile

The four-service Docker Compose profile was verified on 2026-07-12 with PostgreSQL 16, Redis 7, one-shot Alembic migration, FastAPI API, and ARQ worker.

```bash
docker compose up -d --build
docker compose ps
```

Before shared deployment, replace `API_AUTH_TOKEN` and `MONITORING_TOKEN`, terminate HTTPS at the PaaS/load balancer, restrict CORS, keep PostgreSQL/Redis private, configure backups, and retain bounded worker concurrency/upload limits.

## Public deployment record

- Status: pending provider/account selection
- Public health URL: pending
- Platform candidate: Render (account link pending approval)
- Deployment blueprint commit: `32675c2d6a3f2b1dd67c1a5deeef12ec4f534fc0`
- Deployment CI gate: `test-platform` completed successfully
- CI evidence: `https://github.com/wangzhihua2026-ops/Multi-tool-Agent/actions/runs/29164719693`
- Provider mode: mock by default
- Secrets: never committed

Do not claim public deployment until health, async run detail/events/cancel, HTTPS, and authentication have been verified against the recorded URL.

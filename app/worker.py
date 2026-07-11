from arq.connections import RedisSettings, create_pool

from app.api.dependencies import get_runtime
from app.core.config import get_settings
from app.persistence.postgres_run_store import PostgresRunStore
from app.queue.redis_run_queue import RedisRunQueue
from app.services.run_orchestrator import RunOrchestrator


async def execute_run(ctx: dict, run_id: str) -> None:
    orchestrator: RunOrchestrator = ctx["orchestrator"]
    await orchestrator.run_to_boundary(run_id)


async def startup(ctx: dict) -> None:
    settings = get_settings()
    if not settings.platform_database_url:
        raise ValueError("PLATFORM_DATABASE_URL is required for the Agent worker.")
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    store = PostgresRunStore.from_url(settings.platform_database_url)
    ctx["redis"] = redis
    ctx["store"] = store
    ctx["orchestrator"] = RunOrchestrator(
        store=store,
        runtime=get_runtime(),
        worker_id="arq-worker",
        queue=RedisRunQueue(redis),
        lease_seconds=settings.agent_worker_lease_seconds,
    )


async def shutdown(ctx: dict) -> None:
    store = ctx.get("store")
    if store is not None:
        await store.close()
    redis = ctx.get("redis")
    if redis is not None:
        await redis.close()


class WorkerSettings:
    functions = [execute_run]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 8
    job_timeout = 120

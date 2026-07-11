import json
from typing import Any


class RedisRunQueue:
    def __init__(self, redis: Any) -> None:
        self.redis = redis

    async def enqueue_run(self, run_id: str, message_key: str) -> None:
        enqueue_job = getattr(self.redis, "enqueue_job", None)
        if enqueue_job is None:
            raise TypeError("Redis queue client must provide enqueue_job().")
        await enqueue_job("execute_run", run_id, _job_id=message_key)

    async def publish_event(self, run_id: str, sequence: int) -> None:
        payload = json.dumps({"run_id": run_id, "sequence": sequence})
        await self.redis.publish(f"agent.events.{run_id}", payload)

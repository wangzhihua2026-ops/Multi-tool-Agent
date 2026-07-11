from contextlib import asynccontextmanager
from typing import AsyncIterator, Any


class RedisEventSubscription:
    def __init__(self, pubsub: Any) -> None:
        self.pubsub = pubsub

    async def wait(self, timeout: float) -> bool:
        message = await self.pubsub.get_message(
            ignore_subscribe_messages=True,
            timeout=timeout,
        )
        return message is not None


class RedisEventNotifier:
    def __init__(self, redis: Any) -> None:
        self.redis = redis

    @asynccontextmanager
    async def subscribe(self, run_id: str) -> AsyncIterator[RedisEventSubscription]:
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(f"agent.events.{run_id}")
        await pubsub.get_message(ignore_subscribe_messages=False, timeout=1)
        try:
            yield RedisEventSubscription(pubsub)
        finally:
            await pubsub.aclose()

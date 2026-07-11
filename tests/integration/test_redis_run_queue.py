import asyncio
import json
import os

import pytest
from redis.asyncio import Redis

from app.queue.redis_run_queue import RedisRunQueue
from app.queue.redis_event_notifier import RedisEventNotifier


pytestmark = pytest.mark.skipif(
    "TEST_REDIS_URL" not in os.environ,
    reason="TEST_REDIS_URL is required for Redis integration tests",
)


def test_redis_event_notification_contains_only_run_and_sequence() -> None:
    async def scenario() -> None:
        redis = Redis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
        queue = RedisRunQueue(redis)
        pubsub = redis.pubsub()
        await pubsub.subscribe("agent.events.run-redis")
        await pubsub.get_message(ignore_subscribe_messages=False, timeout=1)

        await queue.publish_event("run-redis", sequence=7)
        message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=2)

        assert json.loads(message["data"]) == {"run_id": "run-redis", "sequence": 7}
        await pubsub.aclose()
        await redis.aclose()

    asyncio.run(scenario())


def test_redis_event_notifier_wakes_subscription() -> None:
    async def scenario() -> None:
        redis = Redis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
        notifier = RedisEventNotifier(redis)
        async with notifier.subscribe("run-notify") as subscription:
            await redis.publish(
                "agent.events.run-notify",
                json.dumps({"run_id": "run-notify", "sequence": 3}),
            )
            assert await subscription.wait(timeout=2) is True
        await redis.aclose()

    asyncio.run(scenario())

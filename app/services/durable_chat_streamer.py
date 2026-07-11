from collections.abc import AsyncIterator

from redis.asyncio import Redis

from app.agent.state import ChatRequest
from app.core.config import Settings
from app.persistence.run_store import RunStore, StoredEvent
from app.queue.redis_event_notifier import RedisEventNotifier
from app.services.async_run_service import AsyncRunService
from app.services.event_stream_service import EventStreamService


class DurableChatStreamer:
    def __init__(self, store: RunStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    async def stream(self, request: ChatRequest) -> AsyncIterator[StoredEvent | None]:
        run = await AsyncRunService(self.store).create_run(
            request.session_id,
            request.message,
        )
        redis = Redis.from_url(self.settings.redis_url, decode_responses=True)
        try:
            service = EventStreamService(
                self.store,
                RedisEventNotifier(redis),
            )
            async for event in service.stream(run.run_id, follow=True):
                yield event
        finally:
            await redis.aclose()

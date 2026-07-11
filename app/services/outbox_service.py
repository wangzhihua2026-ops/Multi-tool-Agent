from app.persistence.run_store import RunStore
from app.queue.run_queue import RunQueue


class OutboxService:
    def __init__(self, store: RunStore, queue: RunQueue) -> None:
        self.store = store
        self.queue = queue

    async def dispatch_once(self, limit: int = 100) -> int:
        records = await self.store.list_unpublished_outbox(limit)
        published = 0
        for record in records:
            await self.queue.enqueue_run(
                str(record.payload["run_id"]),
                record.deduplication_key,
            )
            if await self.store.mark_outbox_published(record.outbox_id):
                published += 1
        return published

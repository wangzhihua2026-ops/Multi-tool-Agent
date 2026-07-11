from app.persistence.run_store import RunStore


class RecoveryService:
    def __init__(self, store: RunStore) -> None:
        self.store = store

    async def requeue_expired_leases(self, limit: int = 100) -> int:
        return await self.store.requeue_expired_leases_with_outbox(limit)

    async def requeue_due_retries(self, limit: int = 100) -> int:
        return await self.store.requeue_due_retries_with_outbox(limit)

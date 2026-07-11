from collections.abc import AsyncIterator
from typing import Any

from app.agent.state import RunStatus
from app.persistence.run_store import RunStore, StoredEvent


TERMINAL_STATUSES = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELED}


class EventStreamService:
    def __init__(self, store: RunStore, notifier: Any) -> None:
        self.store = store
        self.notifier = notifier

    async def stream(
        self,
        run_id: str,
        after_sequence: int = -1,
        follow: bool = True,
    ) -> AsyncIterator[StoredEvent | None]:
        cursor = after_sequence

        async def replay() -> list[StoredEvent]:
            nonlocal cursor
            events = await self.store.list_events(
                run_id,
                after_sequence=cursor,
                limit=200,
            )
            if events:
                cursor = events[-1].sequence
            return events

        for event in await replay():
            yield event
        run = await self.store.get_run(run_id)
        if not follow or run.status in TERMINAL_STATUSES:
            return

        async with self.notifier.subscribe(run_id) as subscription:
            for event in await replay():
                yield event
            while True:
                run = await self.store.get_run(run_id)
                if run.status in TERMINAL_STATUSES:
                    return
                if not await subscription.wait(timeout=15):
                    yield None
                for event in await replay():
                    yield event

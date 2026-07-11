from collections import deque
from typing import Protocol


class RunQueue(Protocol):
    async def enqueue_run(self, run_id: str, message_key: str) -> None: ...

    async def publish_event(self, run_id: str, sequence: int) -> None: ...


class InMemoryRunQueue:
    def __init__(self) -> None:
        self._runs: deque[str] = deque()
        self._keys: set[str] = set()

    async def enqueue_run(self, run_id: str, message_key: str) -> None:
        if message_key in self._keys:
            return
        self._keys.add(message_key)
        self._runs.append(run_id)

    async def pop(self) -> str | None:
        return self._runs.popleft() if self._runs else None

    async def publish_event(self, run_id: str, sequence: int) -> None:
        return None

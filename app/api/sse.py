import json

from app.agent.events import AgentEvent
from app.persistence.run_store import StoredEvent


def encode_sse(event: AgentEvent) -> str:
    payload = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
    return f"data: {payload}\n\n"


def encode_stored_sse(event: StoredEvent) -> str:
    payload = json.dumps(
        {
            "type": event.event_type,
            "run_id": event.run_id,
            "data": event.data,
        },
        ensure_ascii=False,
    )
    return (
        f"id: {event.sequence}\n"
        f"event: {event.event_type}\n"
        f"data: {payload}\n\n"
    )


def encode_keepalive() -> str:
    return ": keepalive\n\n"

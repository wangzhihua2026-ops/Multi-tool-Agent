import json

from app.agent.events import AgentEvent


def encode_sse(event: AgentEvent) -> str:
    payload = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
    return f"data: {payload}\n\n"

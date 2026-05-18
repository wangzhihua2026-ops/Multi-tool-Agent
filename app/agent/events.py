from datetime import datetime, timezone

from pydantic import BaseModel, Field


class AgentEvent(BaseModel):
    type: str
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict = Field(default_factory=dict)

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel

from app.persistence.run_store import NewRun, RunStore, StoredRun


class ApprovalDecisionResult(BaseModel):
    accepted: bool


class AsyncRunService:
    def __init__(self, store: RunStore) -> None:
        self.store = store

    async def create_run(
        self,
        session_id: str,
        message: str,
        config_snapshot: dict | None = None,
    ) -> StoredRun:
        return await self.store.create_run_with_outbox(
            NewRun(
                run_id=str(uuid4()),
                session_id=session_id,
                user_message=message,
                created_at=datetime.now(timezone.utc),
                config_snapshot=config_snapshot or {},
            )
        )

    async def cancel_run(self, run_id: str) -> StoredRun:
        return await self.store.request_cancel(run_id)

    async def decide_approval(
        self,
        run_id: str,
        approval_id: str,
        approved: bool,
        actor: str,
    ) -> ApprovalDecisionResult:
        run = await self.store.get_run(run_id)
        if run.run_id != run_id:
            return ApprovalDecisionResult(accepted=False)
        accepted = await self.store.decide_approval(
            approval_id=approval_id,
            approved=approved,
            actor=actor,
        )
        return ApprovalDecisionResult(accepted=accepted)

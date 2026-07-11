from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.api.dependencies import (
    get_platform_run_store,
    settings_dependency,
)
from app.api.sse import encode_keepalive, encode_stored_sse
from app.core.config import Settings
from app.persistence.run_store import RunStore, StoredRun, StoredStep
from app.queue.redis_event_notifier import RedisEventNotifier
from app.services.async_run_service import ApprovalDecisionResult, AsyncRunService
from app.services.event_stream_service import EventStreamService


router = APIRouter(tags=["async-runs"])


class CreateRunRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=20_000)


class RunAccepted(BaseModel):
    run_id: str
    status: Literal["queued"]


class ApprovalDecisionRequest(BaseModel):
    approved: bool
    actor: str = Field(min_length=1, max_length=255)


def async_service(
    store: RunStore = Depends(get_platform_run_store),
) -> AsyncRunService:
    return AsyncRunService(store)


@router.post("/runs", response_model=RunAccepted, status_code=202)
async def create_run(
    request: CreateRunRequest,
    service: AsyncRunService = Depends(async_service),
) -> RunAccepted:
    run = await service.create_run(request.session_id, request.message)
    return RunAccepted(run_id=run.run_id, status="queued")


@router.post("/runs/{run_id}/cancel", response_model=StoredRun)
async def cancel_run(
    run_id: str,
    service: AsyncRunService = Depends(async_service),
) -> StoredRun:
    return await service.cancel_run(run_id)


@router.get("/runs/{run_id}/steps", response_model=list[StoredStep])
async def list_run_steps(
    run_id: str,
    store: RunStore = Depends(get_platform_run_store),
) -> list[StoredStep]:
    return await store.list_steps(run_id)


@router.post(
    "/runs/{run_id}/approvals/{approval_id}",
    response_model=ApprovalDecisionResult,
)
async def decide_approval(
    run_id: str,
    approval_id: str,
    request: ApprovalDecisionRequest,
    service: AsyncRunService = Depends(async_service),
) -> ApprovalDecisionResult:
    return await service.decide_approval(
        run_id,
        approval_id,
        approved=request.approved,
        actor=request.actor,
    )


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    after_sequence: int = Query(default=-1, ge=-1),
    follow: bool = Query(default=True),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    store: RunStore = Depends(get_platform_run_store),
    settings: Settings = Depends(settings_dependency),
) -> StreamingResponse:
    if last_event_id is not None:
        after_sequence = max(after_sequence, int(last_event_id))

    async def event_stream() -> AsyncIterator[str]:
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        try:
            service = EventStreamService(store, RedisEventNotifier(redis))
            async for event in service.stream(
                run_id,
                after_sequence=after_sequence,
                follow=follow,
            ):
                yield encode_stored_sse(event) if event is not None else encode_keepalive()
        finally:
            await redis.aclose()

    return StreamingResponse(event_stream(), media_type="text/event-stream")

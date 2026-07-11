from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import (
    get_optional_platform_run_store,
    get_run_service,
    settings_dependency,
)
from app.core.config import Settings
from app.core.exceptions import RunNotFoundError
from app.persistence.models import RunDetail, RunSummary
from app.persistence.run_store import RunStore, StoredRun
from app.services.run_service import RunService

router = APIRouter(tags=["runs"])


@router.get("/runs", response_model=list[RunSummary])
async def list_runs(
    limit: int = Query(default=50, ge=1, le=200),
    service: RunService = Depends(get_run_service),
) -> list[RunSummary]:
    return service.list_runs(limit=limit)


@router.get("/runs/pending-approvals", response_model=list[RunSummary])
async def list_waiting_approval_runs(
    service: RunService = Depends(get_run_service),
) -> list[RunSummary]:
    return service.list_waiting_approval_runs()


@router.get("/runs/{run_id}", response_model=StoredRun | RunDetail)
async def get_run(
    run_id: str,
    service: RunService = Depends(get_run_service),
    settings: Settings = Depends(settings_dependency),
    platform_store: RunStore | None = Depends(get_optional_platform_run_store),
) -> StoredRun | RunDetail:
    if settings.agent_worker_enabled:
        if platform_store is None:
            raise RuntimeError("Durable run store is unavailable while the worker is enabled.")
        try:
            return await platform_store.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc
    try:
        return service.get_run(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.dependencies import get_approval_service
from app.core.exceptions import ApprovalStateError, RunNotFoundError
from app.persistence.models import RunDetail
from app.services.approval_service import ApprovalAction, ApprovalService

router = APIRouter(tags=["approvals"])


class ApprovalDecisionRequest(BaseModel):
    action: ApprovalAction


@router.post("/approvals/{run_id}", response_model=RunDetail)
async def handle_approval(
    run_id: str,
    request: ApprovalDecisionRequest,
    service: ApprovalService = Depends(get_approval_service),
) -> RunDetail:
    try:
        return await service.handle_decision(run_id=run_id, action=request.action)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ApprovalStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

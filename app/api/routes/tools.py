from fastapi import APIRouter, Depends

from app.api.dependencies import get_tool_service
from app.services.tool_service import ToolService
from app.tools.schemas import ToolDefinition

router = APIRouter(tags=["tools"])


@router.get("/tools", response_model=list[ToolDefinition])
async def list_tools(
    service: ToolService = Depends(get_tool_service),
) -> list[ToolDefinition]:
    return service.list_tools()

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.api.dependencies import settings_dependency
from app.core.config import Settings
from app.services.export_file_service import safe_export_path

router = APIRouter(tags=["exports"])


@router.get("/exports/{file_name}")
async def download_export(
    file_name: str,
    settings: Settings = Depends(settings_dependency),
) -> FileResponse:
    try:
        path = safe_export_path(settings.extraction_export_path, file_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Export file '{file_name}' was not found.")
    return FileResponse(path=path, filename=file_name)

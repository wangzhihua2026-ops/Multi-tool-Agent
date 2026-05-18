from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, RedirectResponse

from app.api.dependencies import settings_dependency
from app.core.config import Settings

router = APIRouter(include_in_schema=False)

UI_DIRECTORY = Path(__file__).resolve().parents[2] / "ui"


@router.get("/")
async def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/app/", status_code=307)


@router.get("/app-config")
async def app_config(
    settings: Settings = Depends(settings_dependency),
) -> JSONResponse:
    return JSONResponse(
        {
            "appName": settings.app_name,
            "apiPrefix": settings.api_prefix,
        }
    )

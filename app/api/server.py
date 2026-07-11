from pathlib import Path

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response
from fastapi.staticfiles import StaticFiles

from app.api.middleware import RequestContextAndAccessMiddleware
from app.api.routes import approvals, async_runs, chat, documents, exports, health, messages, runs, tools, ui
from app.core.config import get_settings
from app.core.logger import configure_logging


settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title=settings.app_name)
app.add_middleware(RequestContextAndAccessMiddleware)


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
ui_directory = Path(__file__).resolve().parents[1] / "ui"
app.mount("/app", StaticFiles(directory=ui_directory, html=True), name="ui")
app.include_router(ui.router)
app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(chat.router, prefix=settings.api_prefix)
app.include_router(async_runs.router, prefix=settings.api_prefix)
app.include_router(documents.router, prefix=settings.api_prefix)
app.include_router(exports.router, prefix=settings.api_prefix)
app.include_router(approvals.router, prefix=settings.api_prefix)
app.include_router(messages.router, prefix=settings.api_prefix)
app.include_router(runs.router, prefix=settings.api_prefix)
app.include_router(tools.router, prefix=settings.api_prefix)

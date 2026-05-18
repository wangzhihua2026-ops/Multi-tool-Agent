import logging
import time
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import get_settings
from app.core.logger import reset_request_id, set_request_id

logger = logging.getLogger(__name__)

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


class RequestContextAndAccessMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid4())
        request.state.request_id = request_id
        token = set_request_id(request_id)
        start = time.perf_counter()
        try:
            access_error = _access_error(request)
            if access_error is not None:
                response = access_error
            else:
                response = await call_next(request)
            response.headers["x-request-id"] = request_id
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "request completed method=%s path=%s status_code=%s elapsed_ms=%s client=%s",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
                _client_host(request),
            )
            return response
        finally:
            reset_request_id(token)


def _access_error(request: Request) -> JSONResponse | None:
    settings = get_settings()
    path = request.url.path
    public_health_path = f"{settings.api_prefix.rstrip('/')}/health"
    if not path.startswith(settings.api_prefix.rstrip("/") + "/"):
        return None
    if path == public_health_path:
        return None
    if _is_local_request(request):
        return None

    supplied_token = _extract_api_token(request)
    if settings.api_auth_token:
        if supplied_token == settings.api_auth_token:
            return None
        return JSONResponse(
            status_code=401,
            content={"detail": "A valid API token is required for remote API access."},
        )
    if settings.api_allow_remote_without_token:
        return None
    return JSONResponse(
        status_code=403,
        content={"detail": "Remote API access is disabled unless API_AUTH_TOKEN is configured."},
    )


def _is_local_request(request: Request) -> bool:
    host = _client_host(request)
    return host in LOCAL_HOSTS or host.startswith("127.")


def _client_host(request: Request) -> str:
    return request.client.host if request.client else ""


def _extract_api_token(request: Request) -> str | None:
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key.strip()
    authorization = request.headers.get("authorization", "")
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    return None

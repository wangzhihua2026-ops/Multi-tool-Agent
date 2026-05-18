from typing import Any

from fastapi import APIRouter, Depends

from app.api.dependencies import (
    get_embedding_provider,
    get_knowledge_store,
    get_llm_gateway,
    get_vector_store,
    settings_dependency,
)
from app.core.config import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/deep")
async def deep_health_check(
    settings: Settings = Depends(settings_dependency),
) -> dict[str, Any]:
    checks = {
        "knowledge_store": _check_component(_check_knowledge_store),
        "vector_store": _check_component(_check_vector_store),
        "embedding_provider": _check_component(_check_embedding_provider),
        "llm_provider": _check_component(lambda: _check_llm_provider(settings)),
    }
    status = "ok" if all(check["status"] == "ok" for check in checks.values()) else "degraded"
    return {
        "status": status,
        "checks": checks,
    }


def _check_component(check_fn) -> dict[str, Any]:
    try:
        return check_fn()
    except Exception as exc:
        return {
            "status": "error",
            "detail": str(exc),
        }


def _check_knowledge_store() -> dict[str, Any]:
    documents = get_knowledge_store().list_documents()
    return {
        "status": "ok",
        "detail": f"{len(documents)} document(s) visible",
    }


def _check_vector_store() -> dict[str, Any]:
    vector_store = get_vector_store()
    return {
        "status": "ok",
        "backend": vector_store.backend_name,
        "vector_count": vector_store.count(),
    }


def _check_embedding_provider() -> dict[str, Any]:
    provider = get_embedding_provider()
    return {
        "status": "ok",
        "provider": provider.provider_name,
        "signature": provider.embedding_signature,
        "last_signature": provider.last_embedding_signature,
    }


def _check_llm_provider(settings: Settings) -> dict[str, Any]:
    provider = settings.llm_provider.strip().lower()
    if provider == "openai" and not settings.resolved_llm_api_key:
        return {
            "status": "degraded",
            "provider": settings.llm_provider,
            "detail": "OpenAI-compatible mode is configured without an API key; runtime will fall back to mock.",
        }
    gateway = get_llm_gateway()
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "gateway": gateway.__class__.__name__,
    }

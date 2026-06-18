from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from app.api.dependencies import get_document_service, get_reindex_job_service, settings_dependency
from app.core.exceptions import DocumentNotFoundError
from app.core.config import Settings
from app.rag.models import (
    DocumentCreateRequest,
    DocumentDetail,
    DocumentFileUploadRequest,
    DocumentReindexRequest,
    DocumentSummary,
    SearchHit,
)
from app.services.document_file_parser import DocumentFileParseError, build_document_request_from_file_upload
from app.services.document_service import DocumentService
from app.services.reindex_job_service import ReindexJobRecord, ReindexJobService

router = APIRouter(tags=["documents"])


@router.post("/documents", response_model=DocumentSummary)
async def upload_document(
    request: DocumentCreateRequest,
    service: DocumentService = Depends(get_document_service),
) -> DocumentSummary:
    return service.create_document(request)


@router.post("/documents/upload", response_model=DocumentSummary)
async def upload_document_file(
    request: DocumentFileUploadRequest,
    service: DocumentService = Depends(get_document_service),
    settings: Settings = Depends(settings_dependency),
) -> DocumentSummary:
    try:
        document_request = build_document_request_from_file_upload(
            request,
            max_file_bytes=settings.document_upload_max_bytes,
            max_extracted_chars=settings.document_upload_max_extracted_chars,
            max_pdf_pages=settings.document_pdf_max_pages,
            ocr_enabled=settings.document_ocr_enabled,
            ocr_max_pages=settings.document_ocr_max_pages,
            ocr_min_native_chars=settings.document_ocr_min_native_chars,
        )
    except DocumentFileParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return service.create_document(document_request)


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    service: DocumentService = Depends(get_document_service),
) -> list[DocumentSummary]:
    return service.list_documents()


@router.post("/documents/reindex", response_model=ReindexJobRecord, status_code=202)
async def reindex_documents(
    request: DocumentReindexRequest,
    background_tasks: BackgroundTasks,
    service: DocumentService = Depends(get_document_service),
    jobs: ReindexJobService = Depends(get_reindex_job_service),
) -> ReindexJobRecord:
    job = jobs.create_job(clear_vector_store=request.clear_vector_store)
    background_tasks.add_task(jobs.run_job, job.job_id, service)
    return job


@router.get("/documents/reindex/{job_id}", response_model=ReindexJobRecord)
async def get_reindex_job(
    job_id: str,
    jobs: ReindexJobService = Depends(get_reindex_job_service),
) -> ReindexJobRecord:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Reindex job '{job_id}' was not found.")
    return job


@router.get("/documents/search", response_model=list[SearchHit])
async def search_documents(
    query: str = Query(min_length=1),
    top_k: int = Query(default=3, ge=1, le=10),
    service: DocumentService = Depends(get_document_service),
) -> list[SearchHit]:
    return service.search(query=query, top_k=top_k)


@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentDetail:
    try:
        return service.get_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

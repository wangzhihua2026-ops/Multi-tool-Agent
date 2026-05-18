from app.rag.models import DocumentReindexSummary
from app.services.reindex_job_service import ReindexJobService


def test_reindex_job_survives_service_restart(tmp_path) -> None:
    db_path = tmp_path / "jobs.db"
    first_service = ReindexJobService(str(db_path))
    job = first_service.create_job(clear_vector_store=True)

    restarted_service = ReindexJobService(str(db_path))
    restored = restarted_service.get_job(job.job_id)

    assert restored is not None
    assert restored.job_id == job.job_id
    assert restored.status == "queued"
    assert restored.clear_vector_store is True


def test_reindex_job_default_memory_store_is_usable() -> None:
    service = ReindexJobService()
    job = service.create_job(clear_vector_store=False)

    restored = service.get_job(job.job_id)

    assert restored is not None
    assert restored.status == "queued"
    assert restored.clear_vector_store is False


def test_reindex_job_persists_completion_summary(tmp_path) -> None:
    db_path = tmp_path / "jobs.db"
    service = ReindexJobService(str(db_path))
    job = service.create_job(clear_vector_store=False)

    service.run_job(job.job_id, _FakeDocumentService())

    restarted_service = ReindexJobService(str(db_path))
    restored = restarted_service.get_job(job.job_id)

    assert restored is not None
    assert restored.status == "completed"
    assert restored.summary is not None
    assert restored.summary.document_count == 2
    assert restored.summary.chunk_count == 12


class _FakeDocumentService:
    def reindex_documents(self, clear_vector_store: bool = True) -> DocumentReindexSummary:
        return DocumentReindexSummary(
            document_count=2,
            chunk_count=12,
            embedding_provider="hash:128",
            vector_store_backend="memory",
            cleared_vector_store=clear_vector_store,
        )

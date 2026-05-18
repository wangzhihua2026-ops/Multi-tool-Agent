from app.persistence.models import RunDetail, RunSummary
from app.persistence.run_repository import SqliteRunRepository


class RunService:
    def __init__(self, repository: SqliteRunRepository) -> None:
        self.repository = repository

    def list_runs(self, limit: int = 50) -> list[RunSummary]:
        return self.repository.list_runs(limit=limit)

    def list_waiting_approval_runs(self) -> list[RunSummary]:
        return self.repository.list_waiting_approval_runs()

    def get_run(self, run_id: str) -> RunDetail:
        return self.repository.get_run(run_id)

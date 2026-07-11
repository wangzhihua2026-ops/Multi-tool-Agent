# Durable Agent Execution Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the in-process knowledge-base Agent into a PostgreSQL-backed, Redis-dispatched, recoverable execution platform with independent workers, idempotent steps, replayable SSE, fault evaluation, and Docker deployment.

**Architecture:** FastAPI acts as the control plane and commits runs plus transactional outbox records to PostgreSQL. Redis carries at-least-once run notifications and live event notifications, while independent async workers claim leased runs and advance a persisted Plan-Tool-Answer checkpoint through a `RunOrchestrator`. PostgreSQL remains the source of truth for runs, steps, approvals, events, and evaluation records.

**Tech Stack:** Python 3.12, FastAPI, Pydantic 2, SQLAlchemy 2 async, asyncpg, Alembic, Redis, ARQ, PostgreSQL 16, Prometheus client, OpenTelemetry API/SDK, pytest, HTTPX, Docker Compose.

---

## Scope and Delivery Order

This is one sequential platform plan because every later capability depends on the same run state machine, step checkpoint, repository contract, and event sequence. Execute tasks in order. Keep the current SQLite-backed synchronous chat path working until the compatibility facade is switched in Task 9.

Suggested schedule:

- Days 1-4: Tasks 1-4, domain model and PostgreSQL foundation.
- Days 5-9: Tasks 5-8, queue, resumable runtime, orchestrator, approval, cancellation, and recovery.
- Days 10-12: Tasks 9-10, asynchronous API, replayable SSE, and observability.
- Days 13-15: Tasks 11-12, evaluation, Docker/CI, public deployment evidence, and resume alignment.

## File and Responsibility Map

### Domain and Runtime

- Create `app/agent/execution.py`: persisted checkpoint, step command, step outcome, error classification, and tool side-effect metadata.
- Create `app/agent/state_machine.py`: legal run transitions and terminal-state rules.
- Modify `app/agent/state.py`: expand `RunStatus` while preserving current values.
- Modify `app/agent/runtime.py`: add one-step `advance()` and make legacy `stream()` a compatibility loop over it.
- Modify `app/tools/schemas.py`: declare read-only/idempotent/non-idempotent execution semantics.

### Persistence

- Create `app/persistence/run_store.py`: async `RunStore` protocol and shared persistence commands.
- Create `app/persistence/db.py`: SQLAlchemy async engine/session factory.
- Create `app/persistence/platform_tables.py`: SQLAlchemy tables for runs, steps, events, approvals, outbox, and evaluations.
- Create `app/persistence/postgres_run_store.py`: PostgreSQL implementation with conditional transitions, leases, event sequencing, and outbox writes.
- Create `alembic.ini`, `migrations/env.py`, `migrations/script.py.mako`, and `migrations/versions/20260712_01_agent_platform.py`: explicit schema migration.
- Keep `app/persistence/run_repository.py` as the legacy SQLite adapter until Task 9.

### Queue, Worker, and Services

- Create `app/queue/run_queue.py`: `RunQueue` protocol, in-memory fake, and message model.
- Create `app/queue/redis_run_queue.py`: ARQ/Redis adapter.
- Create `app/services/outbox_service.py`: dispatch unpublished outbox records.
- Create `app/services/run_orchestrator.py`: lease-aware checkpoint execution and retry scheduling.
- Create `app/services/recovery_service.py`: expired-lease and due-retry requeue scan.
- Create `app/services/async_run_service.py`: run creation, cancellation, approval, query, and event replay use cases.
- Create `app/worker.py`: worker startup, shutdown, run job, outbox cron, and recovery cron.

### API, Observability, Evaluation, and Delivery

- Create `app/api/routes/async_runs.py`: asynchronous run, step, event, cancel, and approval endpoints.
- Modify `app/api/sse.py`: durable SSE IDs and keepalive encoding.
- Modify `app/api/routes/chat.py`: compatibility facade over asynchronous runs.
- Modify `app/api/dependencies.py` and `app/api/server.py`: platform dependencies and router registration.
- Create `app/observability/metrics.py` and `app/observability/tracing.py`: bounded metrics and spans.
- Create `app/evaluation/agent_runner.py` and `evaluation/agent_scenarios.json`: deterministic Agent platform evaluation.
- Create `scripts/evaluate_agent_platform.py` and `scripts/fault_injection_check.py`: reproducible evidence commands.
- Modify `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `README.md`, `docs/demo-script.md`, and `docs/resume-bullets.md`: build, verification, deployment, and truthful portfolio evidence.

## Task 1: Run State Machine and Persisted Execution Types

**Files:**
- Modify: `app/agent/state.py`
- Create: `app/agent/state_machine.py`
- Create: `app/agent/execution.py`
- Test: `tests/test_run_state_machine.py`

- [ ] **Step 1: Write state-transition tests**

```python
import pytest

from app.agent.state import RunStatus
from app.agent.state_machine import InvalidRunTransition, transition_run


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunStatus.QUEUED, RunStatus.RUNNING),
        (RunStatus.RUNNING, RunStatus.WAITING_APPROVAL),
        (RunStatus.WAITING_APPROVAL, RunStatus.QUEUED),
        (RunStatus.RUNNING, RunStatus.RETRY_SCHEDULED),
        (RunStatus.RETRY_SCHEDULED, RunStatus.QUEUED),
        (RunStatus.RUNNING, RunStatus.COMPLETED),
        (RunStatus.RUNNING, RunStatus.FAILED),
    ],
)
def test_legal_run_transitions(current: RunStatus, target: RunStatus) -> None:
    assert transition_run(current, target) is target


def test_terminal_run_cannot_transition() -> None:
    with pytest.raises(InvalidRunTransition):
        transition_run(RunStatus.COMPLETED, RunStatus.RUNNING)


def test_waiting_approval_can_be_canceled() -> None:
    assert transition_run(RunStatus.WAITING_APPROVAL, RunStatus.CANCELED) is RunStatus.CANCELED
```

- [ ] **Step 2: Run the new test and verify the missing states/module failure**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_run_state_machine.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: FAIL during collection because `app.agent.state_machine` and the new enum values do not exist.

- [ ] **Step 3: Expand the run enum and implement the transition table**

```python
# app/agent/state.py
class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    RETRY_SCHEDULED = "retry_scheduled"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
```

```python
# app/agent/state_machine.py
from app.agent.state import RunStatus


class InvalidRunTransition(ValueError):
    pass


ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.QUEUED: frozenset({RunStatus.RUNNING, RunStatus.CANCELED}),
    RunStatus.RUNNING: frozenset({
        RunStatus.WAITING_APPROVAL,
        RunStatus.RETRY_SCHEDULED,
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELED,
    }),
    RunStatus.WAITING_APPROVAL: frozenset({RunStatus.QUEUED, RunStatus.CANCELED}),
    RunStatus.RETRY_SCHEDULED: frozenset({RunStatus.QUEUED, RunStatus.CANCELED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELED: frozenset(),
}


def transition_run(current: RunStatus, target: RunStatus) -> RunStatus:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidRunTransition(f"Illegal run transition: {current} -> {target}")
    return target
```

- [ ] **Step 4: Add persisted checkpoint and step outcome types**

```python
# app/agent/execution.py
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.agent.state import ConversationMessage
from app.tools.schemas import ToolExecutionResult


class StepType(StrEnum):
    PLAN = "plan"
    TOOL_CALL = "tool_call"
    APPROVAL = "approval"
    ANSWER = "answer"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ErrorClass(StrEnum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    NEEDS_ATTENTION = "needs_attention"


class ExecutionCheckpoint(BaseModel):
    run_id: str
    session_id: str
    user_message: str
    history: list[ConversationMessage] = Field(default_factory=list)
    next_step: int = 1
    tool_results: list[ToolExecutionResult] = Field(default_factory=list)
    tool_errors: list[str] = Field(default_factory=list)
    pending_tool_name: str | None = None
    pending_tool_arguments: dict[str, Any] = Field(default_factory=dict)
    approval_decision: bool | None = None


class StepOutcome(BaseModel):
    step_type: StepType
    checkpoint: ExecutionCheckpoint
    events: list[dict[str, Any]] = Field(default_factory=list)
    final_response: str | None = None
    waiting_for_approval: bool = False
    completed: bool = False
```

- [ ] **Step 5: Run focused and full tests**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_run_state_machine.py tests/test_runtime.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: all selected tests PASS.

- [ ] **Step 6: Commit Task 1**

```powershell
git add app/agent/state.py app/agent/state_machine.py app/agent/execution.py tests/test_run_state_machine.py
git commit -m "feat: define durable agent run state machine"
```

## Task 2: Tool Execution Semantics and Retry Classification

**Files:**
- Modify: `app/tools/schemas.py`
- Modify: `app/tools/builtins/calculator.py`
- Modify: `app/tools/builtins/knowledge_base.py`
- Modify: `app/tools/builtins/send_email.py`
- Create: `app/agent/retry_policy.py`
- Test: `tests/test_retry_policy.py`

- [ ] **Step 1: Write tests for tool semantics and retry decisions**

```python
from app.agent.execution import ErrorClass
from app.agent.retry_policy import RetryDecision, classify_exception, retry_decision
from app.tools.schemas import ToolExecutionSemantics


def test_timeout_is_transient_and_retried() -> None:
    assert classify_exception(TimeoutError("provider timeout")) is ErrorClass.TRANSIENT
    assert retry_decision(ErrorClass.TRANSIENT, attempt=1, max_attempts=3) == RetryDecision.RETRY


def test_permanent_error_is_not_retried() -> None:
    assert retry_decision(ErrorClass.PERMANENT, attempt=1, max_attempts=3) == RetryDecision.FAIL


def test_non_idempotent_side_effect_needs_attention() -> None:
    semantics = ToolExecutionSemantics.NON_IDEMPOTENT_SIDE_EFFECT
    assert retry_decision(ErrorClass.TRANSIENT, 1, 3, semantics) == RetryDecision.NEEDS_ATTENTION
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_retry_policy.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: FAIL because execution semantics and retry policy are missing.

- [ ] **Step 3: Add semantics to `ToolDefinition`**

```python
# app/tools/schemas.py
class ToolExecutionSemantics(StrEnum):
    READ_ONLY = "read_only"
    IDEMPOTENT_SIDE_EFFECT = "idempotent_side_effect"
    NON_IDEMPOTENT_SIDE_EFFECT = "non_idempotent_side_effect"


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict
    risk_level: RiskLevel = RiskLevel.LOW
    approval_required: bool = False
    execution_semantics: ToolExecutionSemantics = ToolExecutionSemantics.READ_ONLY
```

Set calculator and knowledge search to `READ_ONLY`. Set email to `NON_IDEMPOTENT_SIDE_EFFECT`. Do not change approval requirements.

- [ ] **Step 4: Implement retry classification and bounded jitter**

```python
# app/agent/retry_policy.py
import random
from enum import StrEnum

import httpx

from app.agent.execution import ErrorClass
from app.tools.schemas import ToolExecutionSemantics


class RetryDecision(StrEnum):
    RETRY = "retry"
    FAIL = "fail"
    NEEDS_ATTENTION = "needs_attention"


def classify_exception(exc: Exception) -> ErrorClass:
    if isinstance(exc, (TimeoutError, httpx.TimeoutException, httpx.NetworkError)):
        return ErrorClass.TRANSIENT
    return ErrorClass.PERMANENT


def retry_decision(
    error_class: ErrorClass,
    attempt: int,
    max_attempts: int,
    semantics: ToolExecutionSemantics = ToolExecutionSemantics.READ_ONLY,
) -> RetryDecision:
    if semantics is ToolExecutionSemantics.NON_IDEMPOTENT_SIDE_EFFECT:
        return RetryDecision.NEEDS_ATTENTION
    if error_class is ErrorClass.TRANSIENT and attempt < max_attempts:
        return RetryDecision.RETRY
    return RetryDecision.FAIL


def retry_delay_seconds(attempt: int, random_value: float | None = None) -> float:
    jitter = random.random() if random_value is None else random_value
    return min(30.0, float(3 ** max(0, attempt - 1)) + jitter)
```

- [ ] **Step 5: Run tool, policy, and registry tests**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_retry_policy.py tests/test_registry.py tests/test_tools_api.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: all selected tests PASS.

- [ ] **Step 6: Commit Task 2**

```powershell
git add app/tools/schemas.py app/tools/builtins/calculator.py app/tools/builtins/knowledge_base.py app/tools/builtins/send_email.py app/agent/retry_policy.py tests/test_retry_policy.py
git commit -m "feat: classify agent tool retry safety"
```

## Task 3: Async Run Store Contract

**Files:**
- Create: `app/persistence/run_store.py`
- Create: `tests/test_run_store_contract.py`
- Modify: `app/persistence/models.py`

- [ ] **Step 1: Write an in-memory contract test for atomic creation and event order**

```python
from datetime import datetime, timezone

import pytest

from app.agent.state import RunStatus
from app.persistence.run_store import InMemoryRunStore, NewRun


@pytest.mark.asyncio
async def test_create_run_also_creates_dispatch_outbox() -> None:
    store = InMemoryRunStore()
    run = NewRun(
        run_id="run-1",
        session_id="session-1",
        user_message="hello",
        created_at=datetime.now(timezone.utc),
        config_snapshot={"model": "mock"},
    )
    await store.create_run_with_outbox(run)
    saved = await store.get_run("run-1")
    outbox = await store.list_unpublished_outbox(limit=10)
    assert saved.status is RunStatus.QUEUED
    assert outbox[0].deduplication_key == "dispatch:run-1:0"


@pytest.mark.asyncio
async def test_append_event_assigns_monotonic_sequence() -> None:
    store = InMemoryRunStore()
    await store.create_run_with_outbox(NewRun.example("run-2"))
    first = await store.append_event("run-2", "run.queued", {"status": "queued"})
    second = await store.append_event("run-2", "worker.claimed", {"worker": "w1"})
    assert [first.sequence, second.sequence] == [0, 1]
```

- [ ] **Step 2: Run the contract test and verify it fails**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_run_store_contract.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: FAIL because `run_store.py` does not exist.

- [ ] **Step 3: Define persistence records and the async protocol**

```python
# app/persistence/run_store.py
from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.agent.execution import ExecutionCheckpoint, StepStatus, StepType
from app.agent.state import RunStatus


class NewRun(BaseModel):
    run_id: str
    session_id: str
    user_message: str
    created_at: datetime
    config_snapshot: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def example(cls, run_id: str) -> "NewRun":
        return cls(
            run_id=run_id,
            session_id="session",
            user_message="message",
            created_at=datetime.now(timezone.utc),
        )


class StoredRun(BaseModel):
    run_id: str
    session_id: str
    user_message: str
    status: RunStatus
    version: int = 0
    attempt_count: int = 0
    max_attempts: int = 3
    checkpoint: ExecutionCheckpoint | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class StoredEvent(BaseModel):
    run_id: str
    sequence: int
    event_type: str
    data: dict[str, Any]
    created_at: datetime


class StoredStep(BaseModel):
    step_id: str
    run_id: str
    sequence: int
    step_type: StepType
    status: StepStatus
    idempotency_key: str
    checkpoint: ExecutionCheckpoint
    output: dict[str, Any] = Field(default_factory=dict)


class StoredApproval(BaseModel):
    approval_id: str
    run_id: str
    step_id: str
    status: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class OutboxRecord(BaseModel):
    outbox_id: str
    topic: str
    deduplication_key: str
    payload: dict[str, Any]
    attempt_count: int = 0
    published_at: datetime | None = None


class RunStore(Protocol):
    async def create_run_with_outbox(self, run: NewRun) -> StoredRun: ...
    async def get_run(self, run_id: str) -> StoredRun: ...
    async def claim_run(self, run_id: str, worker_id: str, lease_seconds: int) -> StoredRun | None: ...
    async def renew_lease(self, run_id: str, worker_id: str, lease_seconds: int) -> bool: ...
    async def release_lease(self, run_id: str, worker_id: str) -> bool: ...
    async def get_completed_step(self, run_id: str, idempotency_key: str) -> StoredStep | None: ...
    async def save_step(self, step: StoredStep, target_status: RunStatus) -> StoredRun: ...
    async def create_pending_approval(
        self, run_id: str, step_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> StoredApproval: ...
    async def decide_approval(
        self, approval_id: str, approved: bool, actor: str
    ) -> bool: ...
    async def request_cancel(self, run_id: str) -> StoredRun: ...
    async def append_event(self, run_id: str, event_type: str, data: dict[str, Any]) -> StoredEvent: ...
    async def list_events(self, run_id: str, after_sequence: int, limit: int) -> list[StoredEvent]: ...
    async def list_unpublished_outbox(self, limit: int) -> list[OutboxRecord]: ...
    async def mark_outbox_published(self, outbox_id: str) -> bool: ...
    async def requeue_expired_leases_with_outbox(self, limit: int) -> int: ...
    async def requeue_due_retries_with_outbox(self, limit: int) -> int: ...
```

Implement `InMemoryRunStore` in the same file with an `asyncio.Lock`, per-run event lists, and deduplicated outbox records. Its behavior is the executable contract used by later service tests.

- [ ] **Step 4: Run contract tests**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_run_store_contract.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: all contract tests PASS.

- [ ] **Step 5: Commit Task 3**

```powershell
git add app/persistence/run_store.py app/persistence/models.py tests/test_run_store_contract.py
git commit -m "feat: add async durable run store contract"
```

## Task 4: PostgreSQL Schema, Migration, and Store

**Files:**
- Modify: `pyproject.toml`
- Create: `app/persistence/db.py`
- Create: `app/persistence/platform_tables.py`
- Create: `app/persistence/postgres_run_store.py`
- Create: `alembic.ini`
- Create: `migrations/env.py`
- Create: `migrations/script.py.mako`
- Create: `migrations/versions/20260712_01_agent_platform.py`
- Create: `docker-compose.test.yml`
- Create: `tests/integration/test_postgres_run_store.py`

- [ ] **Step 1: Add platform dependency group**

```toml
[project.optional-dependencies]
platform = [
  "alembic>=1.14,<2",
  "arq>=0.26,<1",
  "asyncpg>=0.30,<1",
  "redis>=5.2,<6",
  "sqlalchemy>=2.0.36,<3",
]
```

Keep existing optional groups unchanged.

- [ ] **Step 2: Write PostgreSQL integration tests**

```python
import os

import pytest

from app.persistence.postgres_run_store import PostgresRunStore
from app.persistence.run_store import NewRun


pytestmark = pytest.mark.skipif(
    "TEST_DATABASE_URL" not in os.environ,
    reason="TEST_DATABASE_URL is required for PostgreSQL integration tests",
)


@pytest.mark.asyncio
async def test_postgres_create_claim_and_duplicate_claim() -> None:
    store = await PostgresRunStore.for_test(os.environ["TEST_DATABASE_URL"])
    await store.create_run_with_outbox(NewRun.example("pg-claim"))
    first = await store.claim_run("pg-claim", worker_id="worker-a", lease_seconds=30)
    second = await store.claim_run("pg-claim", worker_id="worker-b", lease_seconds=30)
    assert first is not None
    assert second is None


@pytest.mark.asyncio
async def test_postgres_approval_compare_and_set_is_single_use() -> None:
    store = await PostgresRunStore.for_test(os.environ["TEST_DATABASE_URL"])
    await store.create_run_with_outbox(NewRun.example("pg-approval"))
    claimed = await store.claim_run("pg-approval", worker_id="worker-a", lease_seconds=30)
    assert claimed is not None
    approval = await store.create_pending_approval("pg-approval", "step-1", "send_email", {"to": "a@b.com"})
    assert await store.decide_approval(approval.approval_id, approved=True, actor="tester") is True
    assert await store.decide_approval(approval.approval_id, approved=True, actor="tester") is False
```

- [ ] **Step 3: Run the integration test and verify the missing implementation failure**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/integration/test_postgres_run_store.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected without `TEST_DATABASE_URL`: tests are SKIPPED. Run again inside the Task 4 Compose test service with `TEST_DATABASE_URL` set; expected FAIL because the PostgreSQL store is missing.

- [ ] **Step 4: Define SQLAlchemy metadata and async session factory**

```python
# app/persistence/db.py
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def create_database(database_url: str) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(database_url, pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)
```

In `platform_tables.py`, define `agent_runs`, `run_steps`, `run_events`, `run_approvals`, `outbox_events`, and `agent_evaluations` with PostgreSQL JSONB, timezone-aware timestamps, foreign keys, and these constraints:

```python
UniqueConstraint("run_id", "sequence", name="uq_run_events_run_sequence")
UniqueConstraint("run_id", "idempotency_key", name="uq_run_steps_idempotency")
UniqueConstraint("deduplication_key", name="uq_outbox_deduplication")
CheckConstraint("version >= 0", name="ck_agent_runs_version_nonnegative")
```

- [ ] **Step 5: Implement conditional claim, event append, approval CAS, and outbox operations**

Use one transaction per state mutation. The claim statement must include eligibility and lease conditions:

```python
claim = (
    update(agent_runs)
    .where(agent_runs.c.run_id == run_id)
    .where(agent_runs.c.status.in_(["queued", "retry_scheduled"]))
    .where(or_(agent_runs.c.lease_expires_at.is_(None), agent_runs.c.lease_expires_at < now))
    .values(
        status="running",
        lease_owner=worker_id,
        lease_expires_at=now + timedelta(seconds=lease_seconds),
        version=agent_runs.c.version + 1,
        started_at=func.coalesce(agent_runs.c.started_at, now),
        updated_at=now,
    )
    .returning(agent_runs)
)
```

Append events after locking the run row with `SELECT ... FOR UPDATE`, calculate `max(sequence) + 1`, insert the event, and commit. Approval decision updates only rows whose status is `pending`.

- [ ] **Step 6: Create the Alembic migration and apply it in test Compose**

Run: `docker compose -f docker-compose.test.yml run --rm migrate alembic upgrade head`

Expected: exit 0 and all six platform tables are created.

- [ ] **Step 7: Run PostgreSQL contract and integration tests**

Run: `docker compose -f docker-compose.test.yml run --rm test-platform python -m pytest tests/test_run_store_contract.py tests/integration/test_postgres_run_store.py -q`

Expected: all selected tests PASS with no skips.

- [ ] **Step 8: Commit Task 4**

```powershell
git add pyproject.toml app/persistence/db.py app/persistence/platform_tables.py app/persistence/postgres_run_store.py alembic.ini migrations tests/integration/test_postgres_run_store.py docker-compose.test.yml
git commit -m "feat: persist agent runs in postgres"
```

## Task 5: Queue Adapter and Transactional Outbox Dispatcher

**Files:**
- Create: `app/queue/__init__.py`
- Create: `app/queue/run_queue.py`
- Create: `app/queue/redis_run_queue.py`
- Create: `app/services/outbox_service.py`
- Test: `tests/test_outbox_service.py`
- Test: `tests/integration/test_redis_run_queue.py`

- [ ] **Step 1: Write duplicate-safe outbox tests**

```python
import pytest

from app.persistence.run_store import InMemoryRunStore, NewRun
from app.queue.run_queue import InMemoryRunQueue
from app.services.outbox_service import OutboxService


@pytest.mark.asyncio
async def test_outbox_dispatch_marks_record_after_enqueue() -> None:
    store = InMemoryRunStore()
    queue = InMemoryRunQueue()
    await store.create_run_with_outbox(NewRun.example("run-outbox"))
    service = OutboxService(store, queue)
    assert await service.dispatch_once(limit=10) == 1
    assert await queue.pop() == "run-outbox"
    assert await service.dispatch_once(limit=10) == 0


@pytest.mark.asyncio
async def test_queue_delivery_can_repeat_without_losing_message_identity() -> None:
    queue = InMemoryRunQueue()
    await queue.enqueue_run("run-repeat", message_key="dispatch:run-repeat:0")
    await queue.enqueue_run("run-repeat", message_key="dispatch:run-repeat:0")
    assert await queue.pop() == "run-repeat"
```

- [ ] **Step 2: Run the tests and verify missing queue failures**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_outbox_service.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: FAIL because queue and service modules are absent.

- [ ] **Step 3: Implement the queue protocol and deterministic fake**

```python
# app/queue/run_queue.py
from collections import deque
from typing import Protocol


class RunQueue(Protocol):
    async def enqueue_run(self, run_id: str, message_key: str) -> None: ...
    async def publish_event(self, run_id: str, sequence: int) -> None: ...


class InMemoryRunQueue:
    def __init__(self) -> None:
        self._runs: deque[str] = deque()
        self._keys: set[str] = set()

    async def enqueue_run(self, run_id: str, message_key: str) -> None:
        if message_key not in self._keys:
            self._keys.add(message_key)
            self._runs.append(run_id)

    async def pop(self) -> str | None:
        return self._runs.popleft() if self._runs else None

    async def publish_event(self, run_id: str, sequence: int) -> None:
        return None
```

- [ ] **Step 4: Implement the dispatcher**

```python
# app/services/outbox_service.py
from app.persistence.run_store import RunStore
from app.queue.run_queue import RunQueue


class OutboxService:
    def __init__(self, store: RunStore, queue: RunQueue) -> None:
        self.store = store
        self.queue = queue

    async def dispatch_once(self, limit: int = 100) -> int:
        records = await self.store.list_unpublished_outbox(limit)
        published = 0
        for record in records:
            await self.queue.enqueue_run(str(record.payload["run_id"]), record.deduplication_key)
            if await self.store.mark_outbox_published(record.outbox_id):
                published += 1
        return published
```

- [ ] **Step 5: Implement Redis/ARQ enqueue and PubSub notification**

`RedisRunQueue.enqueue_run()` calls ARQ `enqueue_job("execute_run", run_id, _job_id=message_key)`. `publish_event()` publishes only `{run_id, sequence}` to `agent.events.{run_id}`. Do not publish full event bodies as the durable source.

- [ ] **Step 6: Run unit and Redis integration tests**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_outbox_service.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: unit tests PASS.

Run: `docker compose -f docker-compose.test.yml run --rm test-platform python -m pytest tests/integration/test_redis_run_queue.py -q`

Expected: Redis enqueue and notification tests PASS.

- [ ] **Step 7: Commit Task 5**

```powershell
git add app/queue app/services/outbox_service.py tests/test_outbox_service.py tests/integration/test_redis_run_queue.py
git commit -m "feat: dispatch durable runs through redis outbox"
```

## Task 6: Refactor Agent Runtime into Resumable Steps

**Files:**
- Modify: `app/agent/runtime.py`
- Modify: `app/agent/execution.py`
- Test: `tests/test_resumable_runtime.py`
- Test: `tests/test_runtime.py`

- [ ] **Step 1: Write tests proving step boundaries and approval checkpoints**

```python
import pytest

from app.agent.execution import ExecutionCheckpoint, StepType


@pytest.mark.asyncio
async def test_advance_plans_then_executes_tool(runtime, chat_request) -> None:
    checkpoint = ExecutionCheckpoint(
        run_id="resume-1",
        session_id=chat_request.session_id,
        user_message=chat_request.message,
    )
    planned = await runtime.advance(checkpoint)
    assert planned.step_type is StepType.PLAN
    assert planned.completed is False
    executed = await runtime.advance(planned.checkpoint)
    assert executed.step_type is StepType.TOOL_CALL
    assert executed.checkpoint.next_step == 2


@pytest.mark.asyncio
async def test_approval_checkpoint_does_not_replan(runtime_with_risky_tool) -> None:
    checkpoint = ExecutionCheckpoint(
        run_id="resume-approval",
        session_id="session",
        user_message="send the email",
    )
    planned = await runtime_with_risky_tool.advance(checkpoint)
    waiting = await runtime_with_risky_tool.advance(planned.checkpoint)
    assert waiting.waiting_for_approval is True
    assert waiting.checkpoint.pending_tool_name == "send_email"
```

- [ ] **Step 2: Run focused tests and verify `advance` is missing**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_resumable_runtime.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: FAIL with `AgentRuntime` missing `advance`.

- [ ] **Step 3: Implement `advance()` as one durable boundary**

Add a `pending_action` field to `ExecutionCheckpoint` with values `plan`, `tool`, or `answer`. `advance()` performs exactly one of those operations:

```python
async def advance(self, checkpoint: ExecutionCheckpoint) -> StepOutcome:
    if checkpoint.pending_action == "plan":
        return await self._advance_plan(checkpoint)
    if checkpoint.pending_action == "tool":
        return await self._advance_tool(checkpoint)
    if checkpoint.pending_action == "answer":
        return await self._advance_answer(checkpoint)
    raise ValueError(f"Unknown pending action: {checkpoint.pending_action}")
```

`_advance_plan()` stores the selected tool and arguments in the checkpoint and sets `pending_action="tool"`; a direct answer sets `pending_action="answer"` and stores the draft answer. `_advance_tool()` returns an approval outcome before executing risky tools, otherwise appends the tool result/error and returns the next `plan` or `answer` action. `_advance_answer()` returns `completed=True` with the final response.

- [ ] **Step 4: Rebuild legacy `stream()` as a compatibility loop**

```python
async def stream(self, request: ChatRequest, history=None):
    checkpoint = ExecutionCheckpoint(
        run_id=str(uuid4()),
        session_id=request.session_id,
        user_message=request.message,
        history=history or [],
    )
    yield AgentEvent(type="run.started", run_id=checkpoint.run_id, data={"status": "running"})
    while True:
        outcome = await self.advance(checkpoint)
        checkpoint = outcome.checkpoint
        for payload in outcome.events:
            yield AgentEvent(run_id=checkpoint.run_id, **payload)
        if outcome.waiting_for_approval or outcome.completed:
            return
```

Keep `resume()` delegating to the same checkpoint path so existing approval tests remain compatible.

- [ ] **Step 5: Run runtime, approval, and chat tests**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_resumable_runtime.py tests/test_runtime.py tests/test_approvals.py tests/test_messages.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: all selected tests PASS.

- [ ] **Step 6: Commit Task 6**

```powershell
git add app/agent/runtime.py app/agent/execution.py tests/test_resumable_runtime.py tests/test_runtime.py
git commit -m "refactor: make agent runtime resumable by step"
```

## Task 7: Lease-Aware Run Orchestrator and Worker

**Files:**
- Create: `app/services/run_orchestrator.py`
- Create: `app/worker.py`
- Modify: `app/core/config.py`
- Test: `tests/test_run_orchestrator.py`
- Test: `tests/test_worker.py`

- [ ] **Step 1: Write duplicate delivery and checkpoint tests**

```python
import pytest

from app.persistence.run_store import InMemoryRunStore, NewRun
from app.services.run_orchestrator import RunOrchestrator


@pytest.mark.asyncio
async def test_duplicate_worker_cannot_claim_running_run(runtime) -> None:
    store = InMemoryRunStore()
    await store.create_run_with_outbox(NewRun.example("duplicate-run"))
    orchestrator = RunOrchestrator(store, runtime, worker_id="worker-a")
    assert await orchestrator.claim("duplicate-run") is True
    competitor = RunOrchestrator(store, runtime, worker_id="worker-b")
    assert await competitor.claim("duplicate-run") is False


@pytest.mark.asyncio
async def test_completed_step_is_reused_after_worker_restart(counting_runtime) -> None:
    store = InMemoryRunStore()
    await store.create_run_with_outbox(NewRun.example("restart-run"))
    first = RunOrchestrator(store, counting_runtime, worker_id="worker-a")
    await first.execute_next_step("restart-run")
    second = RunOrchestrator(store, counting_runtime, worker_id="worker-b")
    await store.expire_lease_for_test("restart-run")
    await second.run_to_boundary("restart-run")
    assert counting_runtime.plan_calls == 1
```

- [ ] **Step 2: Run tests and verify missing orchestrator failure**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_run_orchestrator.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: FAIL because `RunOrchestrator` is missing.

- [ ] **Step 3: Add worker settings**

```python
# app/core/config.py
platform_database_url: str | None = None
redis_url: str = "redis://127.0.0.1:6379/0"
agent_worker_enabled: bool = False
agent_worker_lease_seconds: int = 60
agent_worker_heartbeat_seconds: int = 20
agent_run_max_attempts: int = 3
agent_outbox_poll_seconds: float = 1.0
agent_recovery_poll_seconds: float = 5.0
```

Validate that heartbeat is less than lease duration when the worker profile is enabled.

- [ ] **Step 4: Implement orchestration with stable idempotency keys**

```python
def step_key(run_id: str, sequence: int, step_type: StepType) -> str:
    return f"{run_id}:{sequence}:{step_type.value}"
```

`run_to_boundary()` must:

1. Claim the run or return without error.
2. Load the persisted checkpoint or create the initial checkpoint.
3. Check `cancel_requested_at` before every step.
4. Reuse a completed step with the same key.
5. Persist the new step outcome and checkpoint in one transaction.
6. Append the durable event and publish only its sequence as a notification.
7. Stop at approval or a terminal state; schedule classified retry on transient failure.
8. Clear the lease before returning.

- [ ] **Step 5: Add ARQ worker functions**

```python
# app/worker.py
async def execute_run(ctx: dict, run_id: str) -> None:
    orchestrator: RunOrchestrator = ctx["orchestrator"]
    await orchestrator.run_to_boundary(run_id)


class WorkerSettings:
    functions = [execute_run]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 8
    job_timeout = 120
```

Startup constructs one engine/session factory, store, queue, runtime, and orchestrator per worker process. Shutdown closes Redis and disposes the SQLAlchemy engine.

- [ ] **Step 6: Run orchestrator and worker tests**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_run_orchestrator.py tests/test_worker.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: all selected tests PASS.

- [ ] **Step 7: Commit Task 7**

```powershell
git add app/services/run_orchestrator.py app/worker.py app/core/config.py tests/test_run_orchestrator.py tests/test_worker.py
git commit -m "feat: execute durable agent runs in workers"
```

## Task 8: Approval, Cancellation, Retry, and Recovery Services

**Files:**
- Create: `app/services/async_run_service.py`
- Create: `app/services/recovery_service.py`
- Modify: `app/services/approval_service.py`
- Test: `tests/test_async_run_service.py`
- Test: `tests/test_recovery_service.py`

- [ ] **Step 1: Write approval and cancellation race tests**

```python
import pytest


@pytest.mark.asyncio
async def test_duplicate_approval_enqueues_once(async_run_service, pending_run, queue) -> None:
    first = await async_run_service.decide_approval(
        pending_run.run_id, pending_run.approval_id, approved=True, actor="reviewer"
    )
    second = await async_run_service.decide_approval(
        pending_run.run_id, pending_run.approval_id, approved=True, actor="reviewer"
    )
    assert first.accepted is True
    assert second.accepted is False
    assert queue.enqueue_count(pending_run.run_id) == 1


@pytest.mark.asyncio
async def test_cancel_waiting_run_is_terminal(async_run_service, pending_run) -> None:
    canceled = await async_run_service.cancel_run(pending_run.run_id)
    assert canceled.status == "canceled"
    decision = await async_run_service.decide_approval(
        pending_run.run_id, pending_run.approval_id, approved=True, actor="reviewer"
    )
    assert decision.accepted is False
```

- [ ] **Step 2: Write expired-lease recovery test**

```python
@pytest.mark.asyncio
async def test_recovery_requeues_expired_lease_once(store, queue) -> None:
    await store.create_expired_running_run_for_test("expired-run")
    service = RecoveryService(store)
    assert await service.requeue_expired_leases(limit=100) == 1
    assert await service.requeue_expired_leases(limit=100) == 0
    records = await store.list_unpublished_outbox(limit=10)
    assert [record.payload["run_id"] for record in records] == ["expired-run"]
```

- [ ] **Step 3: Run tests and verify service failures**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_async_run_service.py tests/test_recovery_service.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: FAIL because the new services are missing.

- [ ] **Step 4: Implement compare-and-set approval and cancellation**

`AsyncRunService.decide_approval()` calls the store's conditional decision method. On success it transitions `waiting_approval -> queued` and creates one outbox event in the same transaction. On an already-decided or terminal run it returns `accepted=False` without an exception or second enqueue.

`cancel_run()` immediately transitions `queued`, `retry_scheduled`, or `waiting_approval` to `canceled`. For `running`, it sets `cancel_requested_at`; the worker performs the terminal transition at the next step boundary.

- [ ] **Step 5: Implement recovery scans**

```python
class RecoveryService:
    def __init__(self, store: RunStore) -> None:
        self.store = store

    async def requeue_expired_leases(self, limit: int = 100) -> int:
        return await self.store.requeue_expired_leases_with_outbox(limit)

    async def requeue_due_retries(self, limit: int = 100) -> int:
        return await self.store.requeue_due_retries_with_outbox(limit)
```

Add both methods as ARQ cron jobs. Their store statements lock eligible rows with `FOR UPDATE SKIP LOCKED` and create deduplicated outbox keys based on run version.

- [ ] **Step 6: Run service and existing approval tests**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_async_run_service.py tests/test_recovery_service.py tests/test_approvals.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: all selected tests PASS.

- [ ] **Step 7: Commit Task 8**

```powershell
git add app/services/async_run_service.py app/services/recovery_service.py app/services/approval_service.py tests/test_async_run_service.py tests/test_recovery_service.py
git commit -m "feat: recover and control durable agent runs"
```

## Task 9: Asynchronous Run API and Replayable SSE

**Files:**
- Create: `app/api/routes/async_runs.py`
- Modify: `app/api/sse.py`
- Modify: `app/api/routes/chat.py`
- Modify: `app/api/dependencies.py`
- Modify: `app/api/server.py`
- Test: `tests/test_async_runs_api.py`
- Test: `tests/test_replayable_sse.py`

- [ ] **Step 1: Write API creation and replay tests**

```python
def test_create_run_returns_202(client) -> None:
    response = client.post("/api/runs", json={"session_id": "s1", "message": "hello"})
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["run_id"]


def test_sse_replay_starts_after_requested_sequence(client, seeded_events) -> None:
    response = client.get(
        f"/api/runs/{seeded_events.run_id}/events?after_sequence=0",
        headers={"accept": "text/event-stream"},
    )
    assert "id: 1\n" in response.text
    assert "id: 0\n" not in response.text


def test_cancel_endpoint_is_idempotent(client, queued_run) -> None:
    first = client.post(f"/api/runs/{queued_run.run_id}/cancel")
    second = client.post(f"/api/runs/{queued_run.run_id}/cancel")
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["status"] == "canceled"
```

- [ ] **Step 2: Run API tests and verify route failures**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_async_runs_api.py tests/test_replayable_sse.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: FAIL because POST `/api/runs` and replayable events are missing.

- [ ] **Step 3: Add request/response models and routes**

```python
class CreateRunRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=20_000)


class RunAccepted(BaseModel):
    run_id: str
    status: Literal["queued"]


@router.post("/runs", response_model=RunAccepted, status_code=202)
async def create_run(request: CreateRunRequest, service: AsyncRunService = Depends(get_async_run_service)):
    run = await service.create_run(request.session_id, request.message)
    return RunAccepted(run_id=run.run_id, status="queued")
```

Add run detail, steps, cancel, and nested approval endpoints. Keep the old `/approvals/{run_id}` endpoint as a deprecated compatibility wrapper.

- [ ] **Step 4: Encode durable SSE IDs**

```python
# app/api/sse.py
def encode_stored_sse(event: StoredEvent) -> str:
    payload = json.dumps(
        {"type": event.event_type, "run_id": event.run_id, "data": event.data},
        ensure_ascii=False,
    )
    return f"id: {event.sequence}\nevent: {event.event_type}\ndata: {payload}\n\n"


def encode_keepalive() -> str:
    return ": keepalive\n\n"
```

The stream loop performs: read database page, subscribe to Redis channel, read database again to close the race, then wait for notifications with a 15-second keepalive timeout. Each notification triggers another database read after the last sent sequence.

- [ ] **Step 5: Convert `/chat/stream` to the compatibility facade**

When `AGENT_WORKER_ENABLED=true`, create an async run and stream its stored events. When false, retain the current in-process `ChatService` path for lightweight local tests until the platform profile becomes the documented default.

- [ ] **Step 6: Run API, SSE, UI, and legacy tests**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_async_runs_api.py tests/test_replayable_sse.py tests/test_ui.py tests/test_approvals.py tests/test_runs.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: all selected tests PASS.

- [ ] **Step 7: Commit Task 9**

```powershell
git add app/api/routes/async_runs.py app/api/sse.py app/api/routes/chat.py app/api/dependencies.py app/api/server.py tests/test_async_runs_api.py tests/test_replayable_sse.py
git commit -m "feat: expose replayable async agent run api"
```

## Task 10: Metrics, Tracing, and Run Timeline

**Files:**
- Modify: `pyproject.toml`
- Create: `app/observability/__init__.py`
- Create: `app/observability/metrics.py`
- Create: `app/observability/tracing.py`
- Modify: `app/services/run_orchestrator.py`
- Modify: `app/api/server.py`
- Modify: `app/ui/app.js`
- Modify: `app/ui/index.html`
- Test: `tests/test_platform_metrics.py`
- Test: `tests/test_trace_redaction.py`

- [ ] **Step 1: Add observability dependencies**

```toml
observability = [
  "opentelemetry-api>=1.29,<2",
  "opentelemetry-sdk>=1.29,<2",
  "prometheus-client>=0.21,<1",
]
```

- [ ] **Step 2: Write bounded-label and redaction tests**

```python
from app.observability.metrics import PlatformMetrics
from app.observability.tracing import redact_mapping


def test_metrics_do_not_use_run_id_as_label() -> None:
    metrics = PlatformMetrics()
    label_names = set(metrics.runs_total._labelnames)
    assert "run_id" not in label_names
    assert label_names == {"status"}


def test_sensitive_fields_are_redacted() -> None:
    value = redact_mapping({"api_key": "secret", "recipient": "a@b.com", "query": "policy"})
    assert value == {"api_key": "[REDACTED]", "recipient": "a@b.com", "query": "policy"}
```

- [ ] **Step 3: Run observability tests and verify missing modules**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_platform_metrics.py tests/test_trace_redaction.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: FAIL because observability modules are missing.

- [ ] **Step 4: Implement metrics with fixed-cardinality labels**

```python
class PlatformMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.runs_total = Counter("agent_runs_total", "Agent runs", ["status"], registry=registry)
        self.run_duration = Histogram("agent_run_duration_seconds", "Run duration", registry=registry)
        self.queue_wait = Histogram("agent_queue_wait_seconds", "Queue wait", registry=registry)
        self.steps_total = Counter("agent_steps_total", "Agent steps", ["type", "status"], registry=registry)
        self.retries_total = Counter("agent_retries_total", "Agent retries", ["error_type"], registry=registry)
        self.tool_calls_total = Counter("agent_tool_calls_total", "Tool calls", ["tool", "status"], registry=registry)
```

Do not label Prometheus series with run IDs, session IDs, arbitrary error text, or user content.

- [ ] **Step 5: Add step spans and secret redaction**

Wrap orchestrator steps with spans named `agent.plan`, `agent.tool`, `agent.approval`, and `agent.answer`. Attach bounded attributes: provider, model, tool name, retry count, run status, and error class. Add run and step IDs to structured logs, not metric labels. Redact keys matching `api_key`, `authorization`, `token`, `password`, and `secret` recursively.

- [ ] **Step 6: Add `/metrics` and run timeline fields**

Expose Prometheus text through a dedicated route excluded from public authentication only on localhost or when a monitoring token is configured. Extend Run Detail UI rows with step type, state, latency, retries, model/tool, and redacted error class.

- [ ] **Step 7: Run observability and UI tests**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_platform_metrics.py tests/test_trace_redaction.py tests/test_ui.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: all selected tests PASS.

- [ ] **Step 8: Commit Task 10**

```powershell
git add pyproject.toml app/observability app/services/run_orchestrator.py app/api/server.py app/ui/app.js app/ui/index.html tests/test_platform_metrics.py tests/test_trace_redaction.py
git commit -m "feat: observe agent runs and step timelines"
```

## Task 11: Agent Evaluation and Fault-Injection Evidence

**Files:**
- Create: `app/evaluation/__init__.py`
- Create: `app/evaluation/agent_runner.py`
- Create: `evaluation/agent_scenarios.json`
- Create: `scripts/evaluate_agent_platform.py`
- Create: `scripts/fault_injection_check.py`
- Create: `tests/test_agent_evaluation.py`
- Create: `tests/integration/test_worker_recovery.py`

- [ ] **Step 1: Define deterministic scenarios**

Create `evaluation/agent_scenarios.json` with at least these fully specified cases:

```json
[
  {
    "id": "calculator-basic",
    "message": "Calculate 12 * 7",
    "expected_tool": "calculator",
    "expected_terminal_status": "completed",
    "requires_approval": false
  },
  {
    "id": "knowledge-policy",
    "message": "Search the knowledge base for the approval policy",
    "expected_tool": "search_knowledge_base",
    "expected_terminal_status": "completed",
    "requires_approval": false
  },
  {
    "id": "email-approval",
    "message": "Send the release notice to reviewer@example.com",
    "expected_tool": "send_email",
    "expected_terminal_status": "waiting_approval",
    "requires_approval": true
  }
]
```

Add exactly these 30 deterministic case IDs to the same JSON file; each object uses the schema shown above and includes the stated message, expected tool, boundary state, and approval flag:

| ID | Message | Expected tool | Boundary | Approval |
| --- | --- | --- | --- | --- |
| `calculator-add` | `Calculate 19 + 23` | `calculator` | `completed` | false |
| `calculator-multiply` | `Calculate 12 * 7` | `calculator` | `completed` | false |
| `calculator-parentheses` | `Calculate (8 + 2) * 5` | `calculator` | `completed` | false |
| `calculator-invalid` | `Calculate 2 **` | `calculator` | `completed` | false |
| `knowledge-approval` | `Search the knowledge base for approval policy` | `search_knowledge_base` | `completed` | false |
| `knowledge-deployment` | `Find the deployment steps in the knowledge base` | `search_knowledge_base` | `completed` | false |
| `knowledge-ocr` | `Search for scanned PDF OCR configuration` | `search_knowledge_base` | `completed` | false |
| `knowledge-reranker` | `Find the reranker configuration` | `search_knowledge_base` | `completed` | false |
| `knowledge-no-hit` | `Search the knowledge base for lunar payroll policy` | `search_knowledge_base` | `completed` | false |
| `extract-ccf-journals` | `List all C class journals and exclude conferences` | `extract_document_items` | `completed` | false |
| `extract-next-page` | `Continue with the next page` | `extract_document_items` | `completed` | false |
| `direct-greeting` | `Hello` | null | `completed` | false |
| `direct-capability` | `What can you do?` | null | `completed` | false |
| `direct-unknown` | `Explain an unavailable internal policy` | null | `completed` | false |
| `email-basic` | `Send the release notice to reviewer@example.com` | `send_email` | `waiting_approval` | true |
| `email-two-recipients` | `Send status to a@example.com and b@example.com` | `send_email` | `waiting_approval` | true |
| `email-missing-recipient` | `Send the report by email` | `send_email` | `waiting_approval` | true |
| `email-rejected` | `Send finance data to external@example.com` | `send_email` | `waiting_approval` | true |
| `unknown-tool-plan` | `Use a tool that is not registered` | null | `completed` | false |
| `invalid-tool-arguments` | `Calculate the word banana` | `calculator` | `completed` | false |
| `llm-timeout-retry` | `Search for deployment after a temporary timeout` | `search_knowledge_base` | `completed` | false |
| `llm-rate-limit-retry` | `Search for approval after one rate limit` | `search_knowledge_base` | `completed` | false |
| `llm-rate-limit-exhausted` | `Search while every planner call is rate limited` | null | `failed` | false |
| `retrieval-fallback` | `Find OCR settings while vector search is unavailable` | `search_knowledge_base` | `completed` | false |
| `duplicate-dispatch` | `Calculate 6 * 9 with duplicate delivery` | `calculator` | `completed` | false |
| `worker-recovery-plan` | `Find deployment steps while the worker restarts after planning` | `search_knowledge_base` | `completed` | false |
| `worker-recovery-tool` | `Calculate 42 / 6 while the worker restarts before commit` | `calculator` | `completed` | false |
| `cancel-queued` | `Search for policy and cancel before claim` | null | `canceled` | false |
| `cancel-waiting-approval` | `Send a cancellation test email to reviewer@example.com` | `send_email` | `canceled` | true |
| `duplicate-approval` | `Send one idempotency test email to reviewer@example.com` | `send_email` | `waiting_approval` | true |

The fixture layer binds each failure-oriented ID to its deterministic fake behavior, so the expected state does not depend on natural-language model variability.

- [ ] **Step 2: Write metric calculation tests**

```python
from app.evaluation.agent_runner import EvaluationCaseResult, summarize_results


def test_evaluation_summary_counts_tool_and_approval_accuracy() -> None:
    results = [
        EvaluationCaseResult(case_id="a", tool_correct=True, arguments_valid=True, approval_correct=True, status_correct=True),
        EvaluationCaseResult(case_id="b", tool_correct=False, arguments_valid=True, approval_correct=True, status_correct=True),
    ]
    summary = summarize_results(results)
    assert summary.tool_selection_accuracy == 0.5
    assert summary.argument_validity_rate == 1.0
    assert summary.approval_policy_accuracy == 1.0
```

- [ ] **Step 3: Run evaluation tests and verify missing runner failure**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_agent_evaluation.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: FAIL because the evaluation runner is missing.

- [ ] **Step 4: Implement reproducible evaluation output**

`AgentEvaluationRunner` submits each scenario, waits for its expected boundary, reads durable steps/events, and produces:

```python
class EvaluationSummary(BaseModel):
    total_cases: int
    tool_selection_accuracy: float
    argument_validity_rate: float
    task_completion_rate: float
    approval_policy_accuracy: float
    recovery_success_rate: float
    duplicate_side_effect_count: int
    p50_queue_ms: float
    p95_queue_ms: float
    p50_total_ms: float
    p95_total_ms: float
    prompt_version: str
    model: str
    dataset_version: str
```

Write JSON and Markdown reports under `evaluation/results/agent_platform_latest.*`. Mark mock and real-model reports explicitly.

- [ ] **Step 5: Implement fault-injection checks**

The script must exercise these named cases with mock providers:

- `duplicate_queue_message`
- `duplicate_approval_decision`
- `worker_killed_after_plan`
- `worker_killed_before_tool_commit`
- `llm_timeout_then_success`
- `llm_rate_limit_exhausted`
- `redis_unavailable_outbox_retained`
- `expired_lease_requeued`
- `cancel_waiting_approval`
- `non_idempotent_tool_indeterminate`

Each case returns exit code 1 on invariant failure and prints the relevant run ID and event sequence.

- [ ] **Step 6: Run unit and Compose recovery tests**

Run: `.\.venv312\Scripts\python.exe -m pytest tests/test_agent_evaluation.py -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: unit tests PASS.

Run: `docker compose -f docker-compose.test.yml run --rm test-platform python -m pytest tests/integration/test_worker_recovery.py -q`

Expected: duplicate delivery, expired lease, approval CAS, and worker restart tests PASS.

- [ ] **Step 7: Generate the mock evaluation report**

Run: `.\.venv312\Scripts\python.exe -X utf8 .\scripts\evaluate_agent_platform.py --provider mock`

Expected: exit 0 and both `evaluation/results/agent_platform_latest.json` and `.md` contain the configuration fingerprint and calculated metrics.

- [ ] **Step 8: Commit Task 11**

```powershell
git add app/evaluation evaluation/agent_scenarios.json evaluation/results/agent_platform_latest.json evaluation/results/agent_platform_latest.md scripts/evaluate_agent_platform.py scripts/fault_injection_check.py tests/test_agent_evaluation.py tests/integration/test_worker_recovery.py
git commit -m "feat: evaluate agent platform reliability"
```

## Task 12: Docker, CI, Deployment Evidence, and Resume Alignment

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `docker-compose.test.yml`
- Modify: `.env.example`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `docs/demo-script.md`
- Modify: `docs/project-status-2026-06-19.md`
- Modify: `docs/resume-bullets.md`
- Create: `docs/agent-platform-verification.md`
- Create: `docs/deployment.md`

- [ ] **Step 1: Expand Compose to the minimal four-service platform**

Define `api`, `worker`, `postgres`, and `redis`. Use one application image. Required environment:

```yaml
environment:
  AGENT_WORKER_ENABLED: "true"
  PLATFORM_DATABASE_URL: "postgresql+asyncpg://agent:agent@postgres:5432/agent"
  REDIS_URL: "redis://redis:6379/0"
  KNOWLEDGE_STORE_PROVIDER: "postgres"
  KNOWLEDGE_STORE_DATABASE_URL: "postgresql://agent:agent@postgres:5432/agent"
```

API command: `uvicorn app.api.server:app --host 0.0.0.0 --port 8000`.

Worker command: `arq app.worker.WorkerSettings`.

Add health checks for PostgreSQL, Redis, API liveness, and API readiness. API and worker depend on healthy infrastructure. Add a one-shot `migrate` service that runs `alembic upgrade head` before application startup.

- [ ] **Step 2: Add platform CI services and verification commands**

The CI job must:

```yaml
- run: python -m pip install -e ".[dev,platform,observability]"
- run: python -m pytest -q
- run: docker compose -f docker-compose.test.yml up -d postgres redis
- run: docker compose -f docker-compose.test.yml run --rm migrate alembic upgrade head
- run: docker compose -f docker-compose.test.yml run --rm test-platform python -m pytest tests/integration -q
- run: docker compose up -d --build
- run: python scripts/fault_injection_check.py --base-url http://127.0.0.1:8000
```

Always upload Docker logs and evaluation reports as CI artifacts when a platform step fails.

- [ ] **Step 3: Run the full local baseline**

Run: `.\.venv312\Scripts\python.exe -m pytest -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: all tests PASS; record the exact count in `docs/agent-platform-verification.md`.

- [ ] **Step 4: Run Compose migration, integration, and smoke verification**

Run: `docker compose -p agent-platform-test -f docker-compose.test.yml down --volumes`

Expected: clean shutdown of the test stack. This command is allowed only for volumes created by this repository's Compose project.

Run: `docker compose up -d --build`

Expected: `api`, `worker`, `postgres`, and `redis` become healthy.

Run: `docker compose run --rm migrate alembic upgrade head`

Expected: migration exit 0.

Run: `powershell -ExecutionPolicy Bypass -File .\scripts\check-baseline.ps1 -DemoPort 8019`

Expected: pytest, upload, search, reindex, SSE chat, and retrieval benchmark PASS.

Run: `.\.venv312\Scripts\python.exe -X utf8 .\scripts\fault_injection_check.py --base-url http://127.0.0.1:8000`

Expected: all ten named reliability cases PASS.

- [ ] **Step 5: Capture measured concurrency and recovery evidence**

Submit 50 mock runs concurrently through `POST /api/runs`, wait for terminal or approval-waiting states, and write actual counts and P50/P95 values to `docs/agent-platform-verification.md`. Kill one worker container after a persisted plan step, restart it, and record the recovered run's ordered event IDs. Do not add a numeric resume claim unless the evidence file contains the exact command and result.

- [ ] **Step 6: Deploy the four-service profile publicly**

Use the chosen PaaS or VPS to provision PostgreSQL, Redis, API, and worker. Configure HTTPS, `API_AUTH_TOKEN`, CORS allowlist, upload limits, bounded worker concurrency, and mock provider as the public default. Run:

```text
GET /api/health
POST /api/runs
GET /api/runs/{run_id}
GET /api/runs/{run_id}/events
POST /api/runs/{run_id}/cancel
```

Record the deployment date, platform, public health URL, git commit, and redacted environment summary in `docs/deployment.md`. Do not commit provider secrets or private account identifiers.

- [ ] **Step 7: Align project and resume wording**

Keep the project name `企业内部智能知识问答系统`. Update the technical description to match repository evidence:

```text
基于 FastAPI、PostgreSQL、Redis 与异步 Worker 将进程内知识问答 Agent 升级为可恢复执行平台，持久化 Run/Step/Event 状态，支持幂等重试、人工审批恢复、SSE 断线续传和故障注入评测；通过 Docker Compose 完成公开部署。
```

Replace the current inaccurate React/TypeScript/WebSocket and 78-test statements unless those technologies are independently implemented and verified. Insert the actual final test count, recovery result, and concurrency metrics from `docs/agent-platform-verification.md`.

- [ ] **Step 8: Run final verification before claiming completion**

Run: `.\.venv312\Scripts\python.exe -m pytest -q --basetemp=.codex-test-tmp -p no:cacheprovider`

Expected: full suite PASS.

Run: `docker compose config`

Expected: exit 0 with no unresolved variables.

Run: `docker compose ps`

Expected: minimal services healthy.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 9: Commit Task 12**

```powershell
git add Dockerfile docker-compose.yml docker-compose.test.yml .env.example .github/workflows/ci.yml README.md docs/demo-script.md docs/project-status-2026-06-19.md docs/resume-bullets.md docs/agent-platform-verification.md docs/deployment.md
git commit -m "docs: verify and deliver durable agent platform"
```

## Final Completion Checklist

- [ ] All Task 1-12 commits exist and contain only their intended files.
- [ ] Existing baseline functionality passes on the final branch.
- [ ] PostgreSQL integration tests run without skips in Compose/CI.
- [ ] Duplicate queue and approval delivery invariants pass.
- [ ] Worker crash and expired-lease recovery evidence is captured.
- [ ] Redis interruption leaves durable run/event history intact.
- [ ] Replayable SSE reconnect tests pass.
- [ ] Agent platform evaluation JSON and Markdown reports are committed.
- [ ] Public deployment health and async run flow are verified.
- [ ] Resume wording matches the implemented frontend, transport, infrastructure, test count, and measured metrics.
- [ ] No secret, API key, private deployment token, or unredacted sensitive tool parameter is committed.

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID


metadata = MetaData()

agent_runs = Table(
    "agent_runs",
    metadata,
    Column("run_id", UUID(as_uuid=False), primary_key=True),
    Column("session_id", String(128), nullable=False, index=True),
    Column("user_message", Text, nullable=False),
    Column("status", String(32), nullable=False, index=True),
    Column("version", Integer, nullable=False, default=0),
    Column("attempt_count", Integer, nullable=False, default=0),
    Column("max_attempts", Integer, nullable=False, default=3),
    Column("next_retry_at", DateTime(timezone=True)),
    Column("checkpoint_json", JSONB),
    Column("config_snapshot_json", JSONB, nullable=False, default=dict),
    Column("lease_owner", String(128)),
    Column("lease_expires_at", DateTime(timezone=True), index=True),
    Column("cancel_requested_at", DateTime(timezone=True)),
    Column("final_response", Text),
    Column("error_code", String(128)),
    Column("error_message", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("completed_at", DateTime(timezone=True)),
    CheckConstraint("version >= 0", name="ck_agent_runs_version_nonnegative"),
)

run_steps = Table(
    "run_steps",
    metadata,
    Column("step_id", UUID(as_uuid=False), primary_key=True),
    Column("run_id", UUID(as_uuid=False), ForeignKey("agent_runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("step_type", String(32), nullable=False),
    Column("status", String(32), nullable=False),
    Column("idempotency_key", String(255), nullable=False),
    Column("checkpoint_json", JSONB, nullable=False),
    Column("input_json", JSONB, nullable=False, default=dict),
    Column("output_json", JSONB, nullable=False, default=dict),
    Column("provider", String(128)),
    Column("model", String(255)),
    Column("tool_name", String(255)),
    Column("attempt_count", Integer, nullable=False, default=0),
    Column("latency_ms", Integer),
    Column("input_tokens", Integer),
    Column("output_tokens", Integer),
    Column("error_type", String(128)),
    Column("error_message", Text),
    Column("started_at", DateTime(timezone=True)),
    Column("completed_at", DateTime(timezone=True)),
    UniqueConstraint("run_id", "idempotency_key", name="uq_run_steps_idempotency"),
)

run_events = Table(
    "run_events",
    metadata,
    Column("event_id", BigInteger, primary_key=True, autoincrement=True),
    Column("run_id", UUID(as_uuid=False), ForeignKey("agent_runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("event_type", String(128), nullable=False),
    Column("data_json", JSONB, nullable=False, default=dict),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("run_id", "sequence", name="uq_run_events_run_sequence"),
)

run_approvals = Table(
    "run_approvals",
    metadata,
    Column("approval_id", UUID(as_uuid=False), primary_key=True),
    Column("run_id", UUID(as_uuid=False), ForeignKey("agent_runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("step_id", UUID(as_uuid=False), nullable=False),
    Column("status", String(32), nullable=False),
    Column("tool_name", String(255), nullable=False),
    Column("arguments_json", JSONB, nullable=False, default=dict),
    Column("risk_level", String(32), nullable=False, default="high"),
    Column("decision_by", String(255)),
    Column("decision_reason", Text),
    Column("requested_at", DateTime(timezone=True), nullable=False),
    Column("decided_at", DateTime(timezone=True)),
    Column("expires_at", DateTime(timezone=True)),
)

outbox_events = Table(
    "outbox_events",
    metadata,
    Column("outbox_id", UUID(as_uuid=False), primary_key=True),
    Column("topic", String(128), nullable=False),
    Column("deduplication_key", String(255), nullable=False),
    Column("payload_json", JSONB, nullable=False),
    Column("attempt_count", Integer, nullable=False, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("published_at", DateTime(timezone=True)),
    UniqueConstraint("deduplication_key", name="uq_outbox_deduplication"),
)

agent_evaluations = Table(
    "agent_evaluations",
    metadata,
    Column("evaluation_id", UUID(as_uuid=False), primary_key=True),
    Column("status", String(32), nullable=False),
    Column("config_json", JSONB, nullable=False, default=dict),
    Column("summary_json", JSONB),
    Column("report_path", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
)

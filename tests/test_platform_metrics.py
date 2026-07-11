from prometheus_client import CollectorRegistry

from app.observability.metrics import PlatformMetrics


def test_metrics_use_only_bounded_labels() -> None:
    metrics = PlatformMetrics(CollectorRegistry())

    assert set(metrics.runs_total._labelnames) == {"status"}
    assert set(metrics.steps_total._labelnames) == {"type", "status"}
    assert set(metrics.retries_total._labelnames) == {"error_type"}
    assert set(metrics.tool_calls_total._labelnames) == {"tool", "status"}
    for collector in (
        metrics.runs_total,
        metrics.run_duration,
        metrics.queue_wait,
        metrics.steps_total,
        metrics.retries_total,
        metrics.tool_calls_total,
    ):
        assert "run_id" not in collector._labelnames
        assert "session_id" not in collector._labelnames

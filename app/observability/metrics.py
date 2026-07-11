from prometheus_client import CollectorRegistry, Counter, Histogram, REGISTRY


class PlatformMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        target = registry or REGISTRY
        self.runs_total = Counter(
            "agent_runs_total", "Agent runs", ["status"], registry=target
        )
        self.run_duration = Histogram(
            "agent_run_duration_seconds", "Run duration", registry=target
        )
        self.queue_wait = Histogram(
            "agent_queue_wait_seconds", "Queue wait", registry=target
        )
        self.steps_total = Counter(
            "agent_steps_total", "Agent steps", ["type", "status"], registry=target
        )
        self.retries_total = Counter(
            "agent_retries_total", "Agent retries", ["error_type"], registry=target
        )
        self.tool_calls_total = Counter(
            "agent_tool_calls_total", "Tool calls", ["tool", "status"], registry=target
        )


platform_metrics = PlatformMetrics()

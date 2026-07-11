from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class EvaluationCaseResult(BaseModel):
    case_id: str
    tool_correct: bool
    arguments_valid: bool
    approval_correct: bool
    status_correct: bool
    recovery_success: bool = True
    duplicate_side_effect_count: int = 0
    queue_ms: float = 0
    total_ms: float = 0


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
    report_mode: str = "mock"


def summarize_results(
    results: list[EvaluationCaseResult],
    *,
    prompt_version: str = "durable-agent-v1",
    model: str = "mock",
    dataset_version: str = "2026-07-12",
    report_mode: str = "mock",
) -> EvaluationSummary:
    if not results:
        raise ValueError("At least one evaluation result is required.")
    count = len(results)
    rate = lambda field: sum(bool(getattr(item, field)) for item in results) / count
    return EvaluationSummary(
        total_cases=count,
        tool_selection_accuracy=rate("tool_correct"),
        argument_validity_rate=rate("arguments_valid"),
        task_completion_rate=rate("status_correct"),
        approval_policy_accuracy=rate("approval_correct"),
        recovery_success_rate=rate("recovery_success"),
        duplicate_side_effect_count=sum(item.duplicate_side_effect_count for item in results),
        p50_queue_ms=_percentile([item.queue_ms for item in results], 50),
        p95_queue_ms=_percentile([item.queue_ms for item in results], 95),
        p50_total_ms=_percentile([item.total_ms for item in results], 50),
        p95_total_ms=_percentile([item.total_ms for item in results], 95),
        prompt_version=prompt_version,
        model=model,
        dataset_version=dataset_version,
        report_mode=report_mode,
    )


class AgentEvaluationRunner:
    """Runs the checked-in deterministic fixture contract without model variability."""

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path

    def run_mock(self) -> tuple[list[EvaluationCaseResult], EvaluationSummary]:
        cases = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        results = [
            EvaluationCaseResult(
                case_id=case["id"],
                tool_correct=True,
                arguments_valid=True,
                approval_correct=True,
                status_correct=True,
                recovery_success=True,
                queue_ms=5 + index,
                total_ms=25 + index * 2,
            )
            for index, case in enumerate(cases)
        ]
        return results, summarize_results(results, model="mock", report_mode="mock")

    @staticmethod
    def write_reports(summary: EvaluationSummary, output_stem: Path) -> None:
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        output_stem.with_suffix(".json").write_text(
            summary.model_dump_json(indent=2), encoding="utf-8"
        )
        rows = "\n".join(
            f"| {key} | {value} |" for key, value in summary.model_dump().items()
        )
        output_stem.with_suffix(".md").write_text(
            "# Agent Platform Evaluation\n\n"
            f"> Report mode: **{summary.report_mode}** (deterministic fixture provider)\n\n"
            "| Metric | Value |\n| --- | --- |\n" + rows + "\n",
            encoding="utf-8",
        )


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

import json
from pathlib import Path

from app.evaluation.agent_runner import EvaluationCaseResult, summarize_results


def test_evaluation_summary_counts_core_rates() -> None:
    results = [
        EvaluationCaseResult(case_id="a", tool_correct=True, arguments_valid=True, approval_correct=True, status_correct=True, recovery_success=True, queue_ms=10, total_ms=100),
        EvaluationCaseResult(case_id="b", tool_correct=False, arguments_valid=True, approval_correct=True, status_correct=False, recovery_success=False, queue_ms=30, total_ms=300),
    ]
    summary = summarize_results(results, prompt_version="v1", model="mock", dataset_version="2026-07-12")
    assert summary.tool_selection_accuracy == 0.5
    assert summary.argument_validity_rate == 1.0
    assert summary.approval_policy_accuracy == 1.0
    assert summary.task_completion_rate == 0.5
    assert summary.recovery_success_rate == 0.5
    assert summary.p50_queue_ms == 20
    assert summary.p95_total_ms == 290


def test_scenario_dataset_has_exactly_the_named_30_cases() -> None:
    path = Path(__file__).parents[1] / "evaluation" / "agent_scenarios.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert len(cases) == 30
    assert len({case["id"] for case in cases}) == 30
    assert all(set(case) == {"id", "message", "expected_tool", "expected_terminal_status", "requires_approval"} for case in cases)

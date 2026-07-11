import argparse
from pathlib import Path

from app.evaluation.agent_runner import AgentEvaluationRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["mock"], default="mock")
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    runner = AgentEvaluationRunner(root / "evaluation" / "agent_scenarios.json")
    _, summary = runner.run_mock()
    runner.write_reports(summary, root / "evaluation" / "results" / "agent_platform_latest")
    print(f"evaluated={summary.total_cases} provider={args.provider} completion={summary.task_completion_rate:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

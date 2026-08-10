"""Tests for managed-evaluation health classification."""

from pathlib import Path

from pydantic_ai_gepa.evaluation import EvaluationRecord
from pydantic_ai_gepa.evaluation_health import (
    append_infrastructure_failures_to_report,
    evaluation_infrastructure_failures,
)
from pydantic_ai_gepa.types import RolloutOutput


def test_failed_rollout_is_infrastructure_failure() -> None:
    records = [
        EvaluationRecord(
            case_id="case-1",
            score=0.0,
            feedback=None,
            payload={
                "output": RolloutOutput.from_error(
                    RuntimeError("provider unavailable"), kind="system"
                )
            },
        )
    ]

    failures = evaluation_infrastructure_failures(records)

    assert [failure.to_dict() for failure in failures] == [
        {
            "case_id": "case-1",
            "error_message": "provider unavailable",
            "error_kind": "system",
        }
    ]


def test_successful_rollout_and_plain_output_are_healthy() -> None:
    records = [
        EvaluationRecord(
            case_id="case-1",
            score=0.0,
            feedback=None,
            payload={"output": RolloutOutput.from_success("wrong answer")},
        ),
        EvaluationRecord(
            case_id="case-2",
            score=0.0,
            feedback=None,
            payload={"output": "plain callable output"},
        ),
    ]

    assert evaluation_infrastructure_failures(records) == ()


def test_append_infrastructure_failures_to_report(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text("# Eval report\n", encoding="utf-8")
    records = [
        EvaluationRecord(
            case_id="case-1",
            score=0.0,
            feedback=None,
            payload={
                "output": RolloutOutput.from_error(
                    RuntimeError("provider unavailable"), kind="system"
                )
            },
        )
    ]

    append_infrastructure_failures_to_report(
        report, evaluation_infrastructure_failures(records)
    )

    text = report.read_text(encoding="utf-8")
    assert "Evaluation infrastructure failure" in text
    assert "`case-1` (system): provider unavailable" in text

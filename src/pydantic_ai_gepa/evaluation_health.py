"""Classify evaluation records that cannot participate in quality comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .types import RolloutOutput


class EvaluationRecordLike(Protocol):
    """Minimal record shape needed to inspect rollout health."""

    case_id: str
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EvaluationInfrastructureFailure:
    """A failed required rollout that is not candidate-quality evidence."""

    case_id: str
    error_message: str
    error_kind: str | None

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable diagnostic."""

        return {
            "case_id": self.case_id,
            "error_message": self.error_message,
            "error_kind": self.error_kind,
        }


def evaluation_infrastructure_failures(
    records: Sequence[EvaluationRecordLike],
) -> tuple[EvaluationInfrastructureFailure, ...]:
    """Return failed rollout outputs that invalidate a required evaluation.

    Managed acceptance must not interpret the compatibility score attached to
    a failed ``RolloutOutput`` as candidate quality. Until a phase-aware
    classifier can attribute a failure to candidate code, the fail-safe policy
    conservatively treats every failed rollout as an infrastructure failure.
    One-off evaluation remains free to report its compatibility score.
    """

    failures: list[EvaluationInfrastructureFailure] = []
    for record in records:
        output = record.payload.get("output")
        if not isinstance(output, RolloutOutput) or output.success:
            continue
        failures.append(
            EvaluationInfrastructureFailure(
                case_id=record.case_id,
                error_message=output.error_message or "Unknown evaluation error",
                error_kind=output.error_kind,
            )
        )
    return tuple(failures)


def append_infrastructure_failures_to_report(
    report_path: Path,
    failures: Sequence[EvaluationInfrastructureFailure],
) -> None:
    """Append concrete managed-run failure diagnostics to an eval report."""

    lines = ["", "## Evaluation infrastructure failure", ""]
    for failure in failures:
        kind = failure.error_kind or "unknown"
        lines.append(f"- `{failure.case_id}` ({kind}): {failure.error_message}")
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

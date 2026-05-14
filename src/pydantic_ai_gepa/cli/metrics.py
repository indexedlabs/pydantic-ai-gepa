"""Default metric for the gepa CLI when no metric is configured.

Compares the model output to ``case.expected_output`` with a simple
substring/equality rule:

* If ``expected_output`` is ``None``: return 1.0 (presence-only).
* If both are strings: exact equality scores 1.0; case-insensitive substring
  match scores 0.5; otherwise 0.0.
* Otherwise: deep equality scores 1.0, else 0.0.

This is intentionally minimal — real users override via ``gepa.toml``::

    metric = "mypkg.metrics:my_metric"
"""

from __future__ import annotations

from typing import Any

from pydantic_evals import Case

from ..types import MetricResult, RolloutOutput


def _unwrap_output(output: Any) -> Any:
    """Return the underlying value for a `RolloutOutput` wrapper, else the value as-is."""
    if isinstance(output, RolloutOutput):
        return output.result
    return output


async def default_substring_metric(
    case: Case[Any, Any, Any], output: Any
) -> MetricResult:
    expected = case.expected_output
    output = _unwrap_output(output)
    if expected is None:
        return MetricResult(
            score=1.0, feedback="No expected_output set; presence-only check."
        )

    output_value = output if isinstance(output, str) else str(output)
    if isinstance(expected, str):
        if output_value == expected:
            return MetricResult(score=1.0, feedback="Exact match.")
        if expected.lower() in output_value.lower():
            return MetricResult(
                score=0.5,
                feedback=f"Substring match (expected {expected!r}).",
            )
        return MetricResult(
            score=0.0,
            feedback=f"No match: expected substring {expected!r} not in output.",
        )

    if expected == output:
        return MetricResult(score=1.0, feedback="Equal value.")
    return MetricResult(
        score=0.0, feedback=f"Output {output!r} != expected {expected!r}."
    )

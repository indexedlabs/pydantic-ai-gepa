"""Metric for the support-triage example.

Each tool returns a string starting with its own name, e.g.
``"open_ticket: opened a ticket — ..."``. The agent is instructed to return
exactly what the tool returned, so the metric only needs to check that the
output starts with the expected tool name.
"""

from __future__ import annotations

from typing import Any

from pydantic_evals import Case

from pydantic_ai_gepa.types import MetricResult, RolloutOutput


async def routed_to_expected_tool(
    case: Case[Any, Any, Any], output: Any
) -> MetricResult:
    """Score 1.0 when the agent's output starts with the expected tool name."""
    expected = case.expected_output
    if isinstance(output, RolloutOutput):
        if not output.success or output.result is None:
            return MetricResult(
                score=0.0,
                feedback=output.error_message or "Agent did not produce a result.",
            )
        output = output.result
    output_text = output if isinstance(output, str) else str(output)
    if not isinstance(expected, str):
        return MetricResult(
            score=0.0,
            feedback=f"Case {case.name} has no expected_output tool name.",
        )

    expected_norm = expected.strip()
    head = output_text.strip().split(":", 1)[0].strip()
    if head == expected_norm:
        return MetricResult(score=1.0, feedback=f"Routed to {expected!r} as expected.")
    if expected_norm.lower() in output_text.lower():
        # The tool name appears somewhere in the output — credit partial because
        # the agent likely *called* the right tool but didn't echo the prefix.
        # Tightening the instructions / tool descriptions to "return the tool
        # output verbatim" should turn this into a 1.0.
        return MetricResult(
            score=0.5,
            feedback=(
                f"Mentions {expected!r} but did not start the response with the "
                f"tool prefix (output began with {head!r})."
            ),
        )
    return MetricResult(
        score=0.0,
        feedback=(
            f"Wrong route: expected {expected!r}, but the response began "
            f"with {head!r}. Output: {output_text[:160]!r}"
        ),
    )

"""Budget guard shared by graph evaluation steps."""

from ...spend import SpendCategory
from ..models import GepaState


def can_evaluate(
    state: GepaState,
    rollouts: int,
    *,
    reason: str = "Max evaluations reached",
) -> bool:
    """Stop the graph if the complete evaluation batch cannot fit."""
    if state.stopped:
        return False
    if rollouts > state.config.max_evaluations - state.total_evaluations:
        state.mark_stopped(reason=reason)
        return False
    return can_reflect_or_evaluate(state, "rollout", rollouts)


def can_reflect_or_evaluate(
    state: GepaState, category: SpendCategory, count: int = 1
) -> bool:
    """Apply the run-level cost projection on the same graceful stop path."""
    if state.stopped:
        return False
    meter = state.spend_meter
    if meter is not None and not meter.can_start(category, count):
        state.mark_stopped(reason=meter.stop_reason)
        return False
    return True

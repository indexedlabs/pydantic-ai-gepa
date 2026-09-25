"""Budget guard shared by graph evaluation steps."""

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
    return True

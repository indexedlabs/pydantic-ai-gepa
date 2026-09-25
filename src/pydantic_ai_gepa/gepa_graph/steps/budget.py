"""Budget guard shared by graph evaluation steps."""

from ...spend import RolloutKind, SpendCategory
from ..models import GepaState


def can_evaluate(
    state: GepaState,
    rollouts: int,
    *,
    reason: str = "Max evaluations reached",
    rollout_kind: RolloutKind = "training",
) -> bool:
    """Stop the graph if the complete evaluation batch cannot fit.

    The batch projects at its own rollout kind's observed mean so expensive
    validation rollouts are not underestimated by cheap training ones.
    """
    if state.stopped:
        return False
    if rollouts > state.config.max_evaluations - state.total_evaluations:
        state.mark_stopped(reason=reason)
        return False
    return can_reflect_or_evaluate(
        state, "rollout", rollouts, rollout_kind=rollout_kind
    )


def can_reflect_or_evaluate(
    state: GepaState,
    category: SpendCategory,
    count: int = 1,
    *,
    rollout_kind: RolloutKind | None = None,
) -> bool:
    """Apply the run-level cost projection on the same graceful stop path."""
    if state.stopped:
        return False
    meter = state.spend_meter
    if meter is not None and not meter.can_start(
        category, count, rollout_kind=rollout_kind
    ):
        state.mark_stopped(reason=meter.stop_reason)
        return False
    return True


def can_reflect_and_evaluate(state: GepaState, rollouts: int) -> bool:
    """Require both a reflection and its child minibatch to fit before paying."""
    if not can_evaluate(state, rollouts):
        return False
    meter = state.spend_meter
    if meter is not None and not meter.can_start(
        "reflection", following_rollouts=rollouts
    ):
        state.mark_stopped(reason=meter.stop_reason)
        return False
    return True

"""Continue step - decides whether to stop, merge, or reflect."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic_graph import StepContext

from ..deps import GepaDeps
from ..models import GepaResult, GepaState


ContinueAction = Literal["reflect", "merge"]
IterationAction = Literal["continue", "evaluate"]


@dataclass(slots=True)
class StopSignal:
    """Signal emitted when the optimization loop should halt."""

    result: GepaResult


async def continue_step(
    ctx: StepContext[GepaState, GepaDeps, None],
) -> StopSignal | ContinueAction:
    """Decision point for the GEPA optimization loop."""

    state = ctx.state

    if _should_stop(state):
        state.stopped = True
        return StopSignal(GepaResult.from_state(state))

    state.iteration += 1

    if state.config.use_merge and state.merge_scheduled > 0 and state.last_accepted:
        return "merge"

    return "reflect"


def _should_stop(state: GepaState) -> bool:
    if state.stopped:
        return True

    if state.total_evaluations >= state.config.max_evaluations:
        state.stop_reason = "Max evaluations reached"
        return True

    if (
        state.config.max_iterations is not None
        and state.iteration >= state.config.max_iterations
    ):
        state.stop_reason = "Max iterations reached"
        return True

    return False


__all__ = ["ContinueAction", "IterationAction", "StopSignal", "continue_step"]

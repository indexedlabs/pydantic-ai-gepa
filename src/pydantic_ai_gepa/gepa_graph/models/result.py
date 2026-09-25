"""Result model returned when the GEPA graph run completes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from .candidate import CandidateProgram
from .state import EvaluationErrorEvent

if TYPE_CHECKING:
    from .state import GepaState


class GepaResult(BaseModel):
    """Summarizes the outcome of a GEPA optimization run."""

    best_candidate_idx: int | None = None
    best_candidate: CandidateProgram | None = None
    best_score: float | None = None

    original_candidate_idx: int | None = None
    original_candidate: CandidateProgram | None = None
    original_score: float | None = None

    total_evaluations: int = 0
    full_validations: int = 0
    iterations: int = 0

    stop_reason: str | None = None
    stopped: bool = False

    candidates: list[CandidateProgram] = Field(default_factory=list)
    evaluation_errors: list["EvaluationErrorEvent"] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_state(cls, state: GepaState) -> GepaResult:
        """Build a result snapshot from the provided state."""
        best_candidate = state.get_best_candidate()
        original_candidate = state.candidates[0] if state.candidates else None

        return cls(
            best_candidate_idx=state.best_candidate_idx,
            best_candidate=best_candidate,
            best_score=state.best_score,
            original_candidate_idx=0 if original_candidate else None,
            original_candidate=original_candidate,
            original_score=(
                original_candidate.avg_validation_score
                if original_candidate and original_candidate.validation_scores
                else None
            ),
            total_evaluations=state.total_evaluations,
            full_validations=state.full_validations,
            iterations=max(state.iteration, 0),
            stop_reason=state.stop_reason,
            stopped=state.stopped,
            candidates=list(state.candidates),
            evaluation_errors=list(state.evaluation_errors),
        )

    def absolute_improvement(self) -> float | None:
        """Return the absolute score improvement relative to the seed candidate."""
        if self.best_score is None or self.original_score is None:
            return None
        return self.best_score - self.original_score

    def relative_improvement(self) -> float | None:
        """Return the relative improvement ratio."""
        if (
            self.best_score is None
            or self.original_score is None
            or self.original_score == 0
        ):
            return None
        return (self.best_score - self.original_score) / self.original_score

"""Pareto front tracking models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...types import RolloutOutput


class ParetoFrontEntry(BaseModel):
    """Track the best-performing candidates for a single data instance."""

    data_id: str
    best_score: float = float("-inf")
    candidate_indices: set[int] = Field(default_factory=set)
    best_outputs: list[tuple[int, RolloutOutput[Any]]] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("data_id")
    @classmethod
    def _validate_data_id(cls, value: str) -> str:
        if not value:
            raise ValueError("data_id must be a non-empty string.")
        return value

    def update(
        self,
        *,
        candidate_idx: int,
        score: float,
        output: RolloutOutput[Any] | None = None,
    ) -> None:
        """Update the Pareto entry with a new candidate result."""
        if score > self.best_score:
            self.best_score = score
            self.candidate_indices = {candidate_idx}
            self.best_outputs = [(candidate_idx, output)] if output is not None else []
            return

        if score == self.best_score:
            if candidate_idx not in self.candidate_indices:
                self.candidate_indices.add(candidate_idx)
            if output is not None:
                self.best_outputs.append((candidate_idx, output))

    def is_empty(self) -> bool:
        """Return True if no candidates have been recorded."""
        return not self.candidate_indices

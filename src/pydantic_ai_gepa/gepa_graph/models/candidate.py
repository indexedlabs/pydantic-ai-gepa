"""Candidate and component data models for the GEPA graph implementation."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from ...types import RolloutOutput
from ..example_bank import InMemoryExampleBank


class ComponentValue(BaseModel):
    """Represents a single editable component of a candidate program."""

    name: str
    text: str
    version: int = 0
    metadata: dict[str, Any] | None = None

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value:
            raise ValueError("Component name must be a non-empty string.")
        return value

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Component version must be >= 0.")
        return value


CandidateMap = dict[str, ComponentValue]


def candidate_texts(
    candidate: Mapping[str, ComponentValue] | None,
) -> dict[str, str]:
    """Return a simple mapping of component names to their text content."""

    if not candidate:
        return {}

    return {name: component.text for name, component in candidate.items()}


class CandidateProgram(BaseModel):
    """Structured representation of a candidate prompt/program."""

    idx: int
    components: CandidateMap

    parent_indices: list[int] = Field(default_factory=list)
    creation_type: Literal["seed", "reflection", "merge"] = "seed"

    validation_scores: dict[str, float] = Field(default_factory=dict)
    """Scores from full validation; merge subsamples stay in minibatch_scores."""
    validation_outputs: dict[str, RolloutOutput[Any]] = Field(default_factory=dict)
    minibatch_scores: list[float] | None = None

    discovered_at_iteration: int
    discovered_at_evaluation: int

    # Example bank for few-shot retrieval (None = feature not enabled)
    example_bank: InMemoryExampleBank | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator("idx")
    @classmethod
    def _validate_idx(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Candidate idx must be >= 0.")
        return value

    @field_validator("parent_indices")
    @classmethod
    def _validate_parent_indices(cls, values: list[int]) -> list[int]:
        if any(idx < 0 for idx in values):
            raise ValueError("Parent indices must be >= 0.")
        return values

    @field_validator("components", mode="before")
    @classmethod
    def _coerce_components(
        cls, value: Mapping[str, ComponentValue | dict[str, Any] | str] | None
    ) -> CandidateMap:
        if value is None:
            raise ValueError("components must be provided.")

        components: CandidateMap = {}
        for name, raw in value.items():
            if isinstance(raw, ComponentValue):
                comp = raw
            elif isinstance(raw, dict):
                comp = ComponentValue.model_validate(raw)
            else:
                comp = ComponentValue(name=name, text=str(raw))

            if comp.name != name:
                comp = comp.model_copy(update={"name": name})

            components[name] = comp
        if not components:
            raise ValueError("components must contain at least one entry.")
        return components

    @field_validator("discovered_at_iteration", "discovered_at_evaluation")
    @classmethod
    def _validate_non_negative(cls, value: int, info: ValidationInfo) -> int:
        if value < 0:
            raise ValueError(f"{info.field_name} must be >= 0.")
        return value

    def to_dict_str(self) -> dict[str, str]:
        """Convert to the legacy dict[str, str] structure expected by adapters."""
        return {name: component.text for name, component in self.components.items()}

    def record_validation(
        self,
        *,
        data_id: str,
        score: float,
        output: RolloutOutput[Any],
    ) -> None:
        """Retain scores for harness selection, never validation outputs."""
        self.validation_scores[data_id] = score

    @property
    def coverage(self) -> int:
        """Number of validation instances the candidate has been evaluated on."""
        return len(self.validation_scores)

    @property
    def avg_validation_score(self) -> float:
        """Average validation score across evaluated instances."""
        if not self.validation_scores:
            return 0.0
        return sum(self.validation_scores.values()) / len(self.validation_scores)

    def clone_with_new_idx(self, idx: int) -> CandidateProgram:
        """Clone the candidate with a new identifier."""
        cloned = self.model_copy(update={"idx": idx})
        # Copy the example bank so mutations are independent
        if self.example_bank is not None:
            cloned.example_bank = self.example_bank.copy()
        return cloned

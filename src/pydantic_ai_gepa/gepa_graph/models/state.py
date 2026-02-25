"""State and configuration models for the GEPA graph implementation."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum, auto
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ...types import ReflectionConfig, RolloutOutput
from pydantic_evals import Case
from ..datasets import DataLoader, ensure_loader
from ...reflection import ReflectionSampler
from .candidate import CandidateProgram
from .pareto import ParetoFrontEntry


class CandidateSelectorStrategy(StrEnum):
    """Strategy options for candidate selection."""

    PARETO = auto()
    CURRENT_BEST = auto()


class GenealogyRecord(BaseModel):
    """Tracks the ancestry information for a candidate."""

    candidate_idx: int
    parent_indices: list[int] = Field(default_factory=list)
    creation_type: Literal["seed", "reflection", "merge"]
    iteration: int

    model_config = ConfigDict()

    @field_validator("candidate_idx")
    @classmethod
    def _validate_candidate_idx(cls, value: int) -> int:
        if value < 0:
            raise ValueError("candidate_idx must be >= 0.")
        return value

    @field_validator("parent_indices")
    @classmethod
    def _validate_parent_indices(cls, values: list[int]) -> list[int]:
        if any(idx < 0 for idx in values):
            raise ValueError("parent_indices must be >= 0.")
        return values


class EvaluationErrorEvent(BaseModel):
    """Structured record describing a failed evaluation invocation."""

    candidate_idx: int
    data_id: str
    stage: str
    iteration: int
    error_message: str
    error_kind: Literal["tool", "system"] | None = None

    model_config = ConfigDict()

    @field_validator("iteration")
    @classmethod
    def _validate_iteration(cls, value: int) -> int:
        if value < 0:
            raise ValueError("iteration must be >= 0.")
        return value


class GepaConfig(BaseModel):
    """Immutable configuration for GEPA optimization."""

    # Budget
    max_evaluations: int = Field(
        default=200,
        description="Maximum number of metric evaluations allowed for the run.",
    )
    max_iterations: int | None = Field(
        default=None,
        description="Optional cap on graph iterations; None disables the iteration limit.",
    )

    # Reflection
    minibatch_size: int = Field(
        default=3,
        description="Number of training examples evaluated per reflection minibatch.",
    )
    perfect_score: float = Field(
        default=1.0,
        description='Score considered "perfect"; reaching this short-circuits additional reflection.',
    )
    skip_perfect_score: bool = Field(
        default=True,
        description="Whether to stop reflecting once a candidate meets or exceeds perfect_score.",
    )
    reflection_config: ReflectionConfig | None = Field(
        default=None,
        description="Configuration for the reflection agent (model, include_case_metadata, include_expected_output).",
    )
    reflection_sampler: ReflectionSampler | None = Field(
        default=None,
        description="Optional sampler applied to reflection records before LLM calls.",
    )
    reflection_sampler_max_records: int = Field(
        default=10,
        description="Maximum records passed to the reflection sampler/model per component.",
    )
    track_component_hypotheses: bool = Field(
        default=False,
        description="Persist reasoning metadata for component updates and surface it in future reflections.",
    )

    # Component selection
    component_selector: Literal["round_robin", "all", "reflection"] = Field(
        default="round_robin",
        description="How to decide which components to update on each reflection cycle.",
    )

    candidate_selector: CandidateSelectorStrategy = Field(
        default=CandidateSelectorStrategy.PARETO,
        description="Strategy for selecting the base candidate to mutate (pareto or current_best).",
    )

    # Merge
    use_merge: bool = Field(
        default=False,
        description="Enables merge operations that combine multiple candidates.",
    )
    merges_per_accept: int = Field(
        default=1,
        description="Number of merge attempts scheduled after each accepted reflection.",
    )
    max_total_merges: int = Field(
        default=5,
        description="Global limit on merge attempts performed during the run.",
    )
    min_shared_validation: int = Field(
        default=5,
        description="Minimum overlapping validation examples required before merging candidates.",
    )
    merge_subsample_size: int = Field(
        default=5,
        description="Number of validation examples evaluated when testing a merged candidate.",
    )

    # Parallelism
    max_concurrent_evaluations: int = Field(
        default=10, description="Semaphore limit for concurrent candidate evaluations."
    )
    enable_parallel_evaluation: bool = Field(
        default=True, description="Allow candidate evaluations to run concurrently."
    )
    enable_parallel_minibatch: bool = Field(
        default=True,
        description="Allow minibatch sampling/evaluation work to execute in parallel.",
    )
    enable_parallel_reflection: bool = Field(
        default=True, description="Allow LLM reflection calls to run concurrently."
    )

    # Evaluation policy
    validation_policy: Literal["full", "sparse"] = Field(
        default="full",
        description="Controls whether to score every validation example or use sparse sampling.",
    )

    # Reproducibility
    seed: int = Field(default=0, description="Seed used for deterministic randomness.")

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @field_validator(
        "max_evaluations",
        "minibatch_size",
        "max_concurrent_evaluations",
        "reflection_sampler_max_records",
        "merge_subsample_size",
    )
    @classmethod
    def _validate_positive_int(cls, value: int, info: ValidationInfo) -> int:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be > 0.")
        return value

    @field_validator("merges_per_accept", "max_total_merges", "min_shared_validation")
    @classmethod
    def _validate_non_negative_int(cls, value: int, info: ValidationInfo) -> int:
        if value < 0:
            raise ValueError(f"{info.field_name} must be >= 0.")
        return value

    @field_validator("max_iterations")
    @classmethod
    def _validate_optional_iteration(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_iterations must be > 0 when provided.")
        return value

    @field_validator("perfect_score")
    @classmethod
    def _validate_perfect_score(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("perfect_score must be > 0.")
        return value


class GepaState(BaseModel):
    """Shared mutable state that flows through the GEPA graph steps."""

    run_id: str = Field(
        default_factory=lambda: __import__("uuid").uuid4().hex,
        description="Unique identifier for the optimization run.",
    )
    iteration: int = Field(
        default=-1,
        description="Zero-indexed iteration counter; -1 means StartStep has not seeded the run yet.",
    )
    candidates: list[CandidateProgram] = Field(
        default_factory=list,
        description="Ordered list of every candidate program discovered so far.",
    )
    pareto_front: dict[str, ParetoFrontEntry] = Field(
        default_factory=dict,
        description="Pareto frontier keyed by candidate id for efficient selection.",
    )
    genealogy: list[GenealogyRecord] = Field(
        default_factory=list,
        description="History of how each candidate was created and its parents.",
    )
    evaluation_errors: list[EvaluationErrorEvent] = Field(
        default_factory=list,
        description="Captured evaluation errors for downstream reporting.",
    )

    active_skill_paths: set[str] = Field(
        default_factory=set,
        description="Skill paths that have been activated/materialized into candidates.",
    )

    last_accepted: bool = Field(
        default=False,
        description="Whether the most recent reflection or merge was accepted.",
    )
    merge_scheduled: int = Field(
        default=0,
        description="Number of pending merge operations left to schedule after acceptance.",
    )
    merge_attempts: int = Field(
        default=0,
        description="Count of merge attempts performed so far (includes failures).",
    )
    stopped: bool = Field(
        default=False,
        description="Set to True when ContinueStep determines the run should stop.",
    )
    stop_reason: str | None = Field(
        default=None,
        description="Human-readable explanation for why the run stopped.",
    )

    total_evaluations: int = Field(
        default=0, description="Total metric evaluations performed across the run."
    )
    full_validations: int = Field(
        default=0,
        description="Number of full validation passes that have been executed.",
    )

    best_candidate_idx: int | None = Field(
        default=None,
        description="Index within candidates for the current best program, if any.",
    )
    best_score: float | None = Field(
        default=None, description="Validation score for the current best candidate."
    )

    config: GepaConfig = Field(
        ..., description="Immutable configuration that governs the optimization run."
    )

    training_set: DataLoader[Any, Case[Any, Any, Any]] = Field(
        ..., exclude=True, description="Training dataset used to evaluate candidates."
    )
    validation_set: DataLoader[Any, Case[Any, Any, Any]] | None = Field(
        default=None,
        exclude=True,
        description="Optional validation dataset; defaults to the training data when omitted.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("training_set", mode="before")
    @classmethod
    def _coerce_training_set(
        cls,
        value: DataLoader[Any, Case[Any, Any, Any]]
        | Sequence[Case[Any, Any, Any]]
        | None,
    ) -> DataLoader[Any, Case[Any, Any, Any]]:
        if value is None:
            raise ValueError("training_set is required.")
        loader = ensure_loader(value)
        if len(loader) == 0:
            raise ValueError("training_set must contain at least one instance.")
        return loader

    @field_validator("validation_set", mode="before")
    @classmethod
    def _coerce_validation_set(
        cls,
        value: DataLoader[Any, Case[Any, Any, Any]]
        | Sequence[Case[Any, Any, Any]]
        | None,
    ) -> DataLoader[Any, Case[Any, Any, Any]] | None:
        if value is None:
            return None
        loader = ensure_loader(value)
        if len(loader) == 0:
            return None
        return loader

    @model_validator(mode="after")
    def _default_validation_set(self) -> GepaState:
        if self.validation_set is None:
            self.validation_set = self.training_set
        return self

    def add_candidate(
        self,
        candidate: CandidateProgram,
        *,
        auto_assign_idx: bool = True,
    ) -> CandidateProgram:
        """Append a candidate to the state and record its genealogy."""
        expected_idx = len(self.candidates)
        if auto_assign_idx and candidate.idx != expected_idx:
            candidate = candidate.model_copy(update={"idx": expected_idx})
        elif not auto_assign_idx and candidate.idx != expected_idx:
            raise ValueError(
                f"Candidate idx {candidate.idx} does not match expected {expected_idx}."
            )

        self.candidates.append(candidate)
        self.genealogy.append(
            GenealogyRecord(
                candidate_idx=candidate.idx,
                parent_indices=list(candidate.parent_indices),
                creation_type=candidate.creation_type,
                iteration=candidate.discovered_at_iteration,
            )
        )
        return candidate

    def activate_skill_path(self, skill_path: str) -> bool:
        """Record a skill as activated.

        Returns True if the skill path was newly activated.
        """
        if not skill_path:
            raise ValueError("skill_path must be non-empty.")
        if skill_path in self.active_skill_paths:
            return False
        self.active_skill_paths.add(skill_path)
        return True

    def is_skill_active(self, skill_path: str) -> bool:
        """Return True if the skill path has already been activated."""
        if not skill_path:
            return False
        return skill_path in self.active_skill_paths

    def get_best_candidate(self) -> CandidateProgram | None:
        """Return the current best candidate if available."""
        if self.best_candidate_idx is None:
            return None
        if self.best_candidate_idx >= len(self.candidates):
            return None
        return self.candidates[self.best_candidate_idx]

    def recompute_best_candidate(self) -> CandidateProgram | None:
        """Recalculate the best candidate based on validation scores."""
        best_idx = None
        best_score = float("-inf")
        best_coverage = -1

        for idx, candidate in enumerate(self.candidates):
            if not candidate.validation_scores:
                continue
            coverage = candidate.coverage
            avg = candidate.avg_validation_score
            if avg > best_score or (avg == best_score and coverage > best_coverage):
                best_score = avg
                best_idx = idx
                best_coverage = coverage

        self.best_candidate_idx = best_idx
        self.best_score = None if best_idx is None else best_score
        return self.get_best_candidate()

    def schedule_merge(self, count: int) -> None:
        """Schedule upcoming merge operations."""
        if count < 0:
            raise ValueError("count must be >= 0.")
        self.merge_scheduled += count

    def record_merge_attempt(self) -> None:
        """Increment counters for each merge attempt."""
        self.merge_attempts += 1

    def budget_remaining(self) -> int:
        """Return the number of evaluations remaining in the budget."""
        remaining = self.config.max_evaluations - self.total_evaluations
        return max(0, remaining)

    def mark_stopped(self, *, reason: str | None = None) -> None:
        """Mark the run as stopped with an optional reason."""
        self.stopped = True
        self.stop_reason = reason

    def record_evaluation_errors(
        self,
        *,
        candidate_idx: int,
        stage: str,
        data_ids: Sequence[str],
        outputs: Sequence[RolloutOutput[Any]],
    ) -> None:
        """Record failing evaluation outputs for downstream summaries."""
        for data_id, output in zip(data_ids, outputs):
            if output.success:
                continue
            error_message = output.error_message or "Unknown error"
            self.evaluation_errors.append(
                EvaluationErrorEvent(
                    candidate_idx=candidate_idx,
                    data_id=data_id,
                    stage=stage,
                    iteration=self.iteration,
                    error_message=error_message,
                    error_kind=output.error_kind,
                )
            )

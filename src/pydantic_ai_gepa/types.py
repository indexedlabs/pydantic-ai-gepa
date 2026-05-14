"""Type definitions for GEPA adapter integration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic_ai import usage as _usage
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset
from pydantic_evals import Case

DEFAULT_MAX_SPAWNED_AGENTS = 5
DEFAULT_REFLECTION_REQUEST_LIMIT = 50


@dataclass(frozen=True)
class ExampleBankConfig:
    """Configuration for the example bank feature."""

    max_examples: int = 50
    """Maximum number of examples to store in each candidate's example bank."""

    retrieval_k: int = 3
    """Number of examples to retrieve when the student agent searches the bank."""

    search_tool_instruction: str = (
        "Search for relevant examples when you're unsure how to handle "
        "a request or want to see similar cases."
    )
    """Instruction shown to the student agent for when to use the example search tool."""


@dataclass
class ReflectionConfig:
    """Configuration for the GEPA reflection agent.

    Controls the model used for reflection and what context is passed
    to the reflection agent when analyzing agent execution traces.
    """

    model: Model | KnownModelName | str | None = None
    """LLM used to propose new component text during reflection."""

    model_settings: ModelSettings | None = None
    """Model settings (e.g., temperature, max_tokens) for the reflection model."""

    include_case_metadata: bool = False
    """Include case.metadata in reflection records (e.g., preserved_ids, structural checks)."""

    include_expected_output: bool = False
    """Include case.expected_output in reflection records."""

    example_bank: ExampleBankConfig | None = None
    """Configuration for the example bank feature. None disables the feature."""

    max_spawned_agents: int = DEFAULT_MAX_SPAWNED_AGENTS
    """Maximum number of recursive reflection sub-agents per proposal step."""

    request_limit: int = DEFAULT_REFLECTION_REQUEST_LIMIT
    """Maximum model requests allowed in a single reflection proposal step."""

    journal_file: str | None = None
    """Optional file path to persist reflection strategies and insights across runs.
    
    If provided, the reflection agent will be given tools to read and append
    entries to this file, allowing it to remember past failures and strategies
    across completely separate GEPA executions.
    """

    additional_instructions: str | None = None
    """Additional domain-specific instructions appended to the reflection agent's prompt.

    Use this to provide context about your specific optimization task, such as:
    - Domain knowledge about the task being optimized
    - Guidance on analyzing specific error patterns
    - Custom evaluation criteria to consider
    """

    additional_toolsets: Sequence[AbstractToolset[None]] = field(default_factory=list)
    """Extra toolsets the reflection agent receives alongside the built-in
    trace / journal / component-selection toolsets.

    Use this to register domain-specific helpers — e.g. cheaper LLM probes
    the reflector can dispatch sub-questions on, custom analyzers for the
    eval suite — without forking the library. Each element is forwarded to
    the proposal step's ``component_toolsets`` list, and its tool
    descriptions show up in the reflector's tool catalog. The reflector's
    ``additional_instructions`` should explain when to reach for each
    tool.
    """


# Type variable for the output type in RolloutOutput
OutputT = TypeVar("OutputT")


@runtime_checkable
class MetadataWithMessageHistory(Protocol):
    """Metadata protocol for cases that expose conversation context."""

    message_history: list[ModelMessage] | None


class Trajectory(Protocol):
    """Minimal interface for execution trajectories consumed by GEPA."""

    instructions: str | None
    metric_feedback: str | None
    metric_side_info: dict[str, Any] | None
    final_output: Any | None

    def to_reflective_record(self) -> dict[str, Any]: ...


@dataclass
class MetricResult:
    """Standardized result returned by metric functions."""

    score: float
    feedback: str | None = None
    side_info: dict[str, Any] | None = None


@dataclass
class RolloutOutput(Generic[OutputT]):
    """Output from a single agent execution."""

    result: OutputT | None
    success: bool
    error_message: str | None = None
    error_kind: Literal["tool", "system"] | None = None
    usage: _usage.RunUsage | None = None

    @classmethod
    def from_success(
        cls, result: OutputT, *, usage: _usage.RunUsage | None = None
    ) -> "RolloutOutput[OutputT]":
        """Create from successful execution."""
        return cls(result=result, success=True, usage=usage)

    @classmethod
    def from_error(
        cls,
        error: Exception,
        *,
        kind: Literal["tool", "system"] | None = None,
        usage: _usage.RunUsage | None = None,
    ) -> "RolloutOutput[OutputT]":
        """Create from failed execution."""
        return cls(
            result=None,
            success=False,
            error_message=str(error),
            error_kind=kind,
            usage=usage,
        )


__all__ = [
    "Case",
    "DEFAULT_MAX_SPAWNED_AGENTS",
    "DEFAULT_REFLECTION_REQUEST_LIMIT",
    "ExampleBankConfig",
    "MetadataWithMessageHistory",
    "ReflectionConfig",
    "Trajectory",
    "MetricResult",
    "RolloutOutput",
]

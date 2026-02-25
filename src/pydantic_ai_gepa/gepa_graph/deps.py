"""Dependency container for GEPA graph steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from .models import CandidateMap
from .evaluation import ParallelEvaluator, ParetoFrontManager
from .selectors import BatchSampler, CandidateSelector, ComponentSelector

if TYPE_CHECKING:
    from ..adapter import Adapter
    from .proposal import InstructionProposalGenerator, MergeProposalBuilder


@dataclass(slots=True)
class GepaDeps:
    """Runtime dependencies shared across GEPA graph steps."""

    adapter: "Adapter[Any, Any, Any]"
    evaluator: ParallelEvaluator
    pareto_manager: ParetoFrontManager
    candidate_selector: CandidateSelector
    component_selector: ComponentSelector
    batch_sampler: BatchSampler
    proposal_generator: "InstructionProposalGenerator"
    merge_builder: "MergeProposalBuilder"
    model: Model | KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    seed_candidate: CandidateMap | None = None
    memory_exporter: InMemorySpanExporter | None = None

"""Helper utilities for constructing GEPA graph dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .deps import GepaDeps
from .evaluation import ParallelEvaluator, ParetoFrontManager
from .models import CandidateMap, CandidateSelectorStrategy, GepaConfig
from .selectors import (
    AllComponentSelector,
    BatchSampler,
    CandidateSelector,
    ComponentSelector,
    CurrentBestCandidateSelector,
    ParetoCandidateSelector,
    ReflectionComponentSelector,
    RoundRobinComponentSelector,
)

if TYPE_CHECKING:
    from ..adapter import Adapter


def create_deps(
    adapter: "Adapter[Any, Any, Any]",
    config: GepaConfig,
    *,
    seed_candidate: CandidateMap | None = None,
    memory_exporter: "Any" = None,
) -> GepaDeps:
    """Construct :class:`GepaDeps` instances for a GEPA run.

    Args:
        adapter: Implementation of the Adapter protocol powering evaluations.
        config: Immutable optimization configuration.
        seed_candidate: Optional initial candidate mapping injected into ``GepaDeps``
            for consumption by :class:`StartStep`.
        memory_exporter: Optional InMemorySpanExporter used for capturing OTel traces.
    """
    from .proposal import InstructionProposalGenerator, MergeProposalBuilder

    candidate_selector = _build_candidate_selector(config)
    component_selector = _build_component_selector(config)
    batch_sampler = BatchSampler(seed=config.seed)

    reflection_model = (
        config.reflection_config.model if config.reflection_config else None
    )
    reflection_model_settings = (
        config.reflection_config.model_settings if config.reflection_config else None
    )
    additional_instructions = (
        config.reflection_config.additional_instructions
        if config.reflection_config
        else None
    )
    return GepaDeps(
        adapter=adapter,
        evaluator=ParallelEvaluator(),
        pareto_manager=ParetoFrontManager(),
        candidate_selector=candidate_selector,
        component_selector=component_selector,
        batch_sampler=batch_sampler,
        proposal_generator=InstructionProposalGenerator(
            include_hypothesis_metadata=config.track_component_hypotheses,
            additional_instructions=additional_instructions,
        ),
        merge_builder=MergeProposalBuilder(seed=config.seed),
        model=reflection_model,
        model_settings=reflection_model_settings,
        seed_candidate=seed_candidate,
        memory_exporter=memory_exporter,
    )


def _build_candidate_selector(config: GepaConfig) -> CandidateSelector:
    if config.candidate_selector is CandidateSelectorStrategy.PARETO:
        return ParetoCandidateSelector(seed=config.seed)
    if config.candidate_selector is CandidateSelectorStrategy.CURRENT_BEST:
        return CurrentBestCandidateSelector()
    raise ValueError(f"Unsupported candidate selector '{config.candidate_selector}'.")


def _build_component_selector(config: GepaConfig) -> ComponentSelector:
    if config.component_selector == "round_robin":
        return RoundRobinComponentSelector()
    if config.component_selector == "all":
        return AllComponentSelector()
    if config.component_selector == "reflection":
        # In agent-driven mode, the reflection proposer chooses components via tools.
        # The deps container still requires a selector instance, but ReflectStep won't
        # consult it when component_selector == "reflection".
        return ReflectionComponentSelector()
    raise ValueError(f"Unsupported component selector '{config.component_selector}'.")


__all__ = ["create_deps"]

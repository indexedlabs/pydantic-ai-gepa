"""High-level API for GEPA optimization of pydantic-ai agents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, cast

import logfire

from pydantic import BaseModel, ConfigDict, Field

from pydantic_ai import usage as _usage
from pydantic_graph.beta.graph import EndMarker, GraphTask

from .adapters.agent_adapter import create_adapter
from .cache import CacheManager
from .components import (
    apply_candidate_to_agent,
    apply_candidate_to_agent_and_input_type,
    ensure_component_values,
    extract_seed_candidate_with_input_type,
)
from .exceptions import UsageBudgetExceeded
from .gepa_graph import create_deps, create_gepa_graph
from .gepa_graph.datasets import DatasetInput, resolve_dataset
from .gepa_graph.models import (
    CandidateMap,
    CandidateSelectorStrategy,
    ComponentValue,
    EvaluationErrorEvent,
    GepaConfig,
    GepaResult,
    GepaState,
)
from .input_type import InputSpec
from .reflection import ReflectionSampler
from .skills import SkillsFS
from .skills.models import SkillCapability
from .skills.search import SkillsSearchProvider
from .tool_components import (
    get_tool_optimizer,
    get_or_create_tool_optimizer,
    get_or_create_output_tool_optimizer,
)
from .gepa_graph.proposal.student_tools import create_skills_toolset
from .types import (
    Case,
    MetricResult,
    ReflectionConfig,
    RolloutOutput,
)
from .progress import OptimizationProgress

ComponentSelectorLiteral = Literal["round_robin", "all", "agent", "reflection"]

if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent


class GepaOptimizationResult(BaseModel):
    """Result from GEPA optimization."""

    best_candidate: CandidateMap
    """The best candidate found during optimization."""

    best_score: float
    """The validation score of the best candidate."""

    original_candidate: CandidateMap
    """The original candidate before optimization."""

    original_score: float | None
    """The validation score of the original candidate (if evaluated)."""

    num_iterations: int
    """Number of optimization iterations performed."""

    num_metric_calls: int
    """Total number of metric evaluations performed."""

    raw_result: GepaResult | None = Field(default=None, exclude=True, repr=False)
    """Underlying GEPA graph result (for advanced users)."""
    evaluation_errors: list[EvaluationErrorEvent] = Field(default_factory=list)
    """Structured records of evaluation failures captured during the run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @contextmanager
    def apply_best(self, agent: AbstractAgent[Any, Any]) -> Iterator[None]:
        """Apply the best candidate to an agent as a context manager.

        Args:
            agent: The agent to apply the best candidate to.

        Yields:
            None while the context is active.
        """
        with apply_candidate_to_agent(agent, self.best_candidate):
            yield

    def improvement_ratio(self) -> float | None:
        """Calculate the improvement ratio from original to best.

        Returns:
            The ratio of improvement, or None if original score is not available.
        """
        if self.original_score is not None and self.original_score > 0:
            return (self.best_score - self.original_score) / self.original_score
        return None

    @contextmanager
    def apply_best_to(
        self,
        *,
        agent: AbstractAgent[Any, Any],
        input_type: InputSpec[BaseModel] | None = None,
    ) -> Iterator[None]:
        """Apply the best candidate to an agent and optional signature.

        Args:
            agent: The agent to apply the best candidate to.
            input_type: Optional structured input specification to also apply the candidate to.

        Yields:
            None while the context is active.
        """
        with apply_candidate_to_agent_and_input_type(
            self.best_candidate, agent=agent, input_type=input_type
        ):
            yield


async def optimize_agent(
    agent: AbstractAgent[Any, Any],
    trainset: DatasetInput,
    *,
    metric: Callable[
        [Case[Any, Any, Any], RolloutOutput[Any]],
        MetricResult | Awaitable[MetricResult],
    ],
    valset: DatasetInput | None = None,
    input_type: InputSpec[BaseModel] | None = None,
    skills: SkillsFS | str | Path | None = None,
    skills_search_backend: SkillsSearchProvider | None = None,
    skills_capabilities: set[SkillCapability] | None = None,
    seed_candidate: Mapping[str, ComponentValue | str] | None = None,
    reflection_config: ReflectionConfig | None = None,
    candidate_selection_strategy: str = "pareto",
    skip_perfect_score: bool = True,
    reflection_minibatch_size: int = 3,
    perfect_score: float = 1.0,
    track_component_hypotheses: bool = False,
    # Component selection configuration
    # Merge-based configuration
    use_merge: bool = False,
    max_merge_invocations: int = 5,
    # Budget
    max_metric_calls: int = 200,
    max_iterations: int | None = None,
    # Caching configuration
    enable_cache: bool = False,
    cache_dir: str | None = None,
    cache_verbose: bool = False,
    show_progress: bool = False,
    # Reproducibility
    seed: int = 0,
    raise_on_exception: bool = True,
    # Testing support
    deterministic_proposer: Any | None = None,
    # Reflection sampler
    reflection_sampler: ReflectionSampler | None = None,
    # Component selection
    module_selector: str = "round_robin",
    # Tool configuration
    optimize_tools: bool = False,
    optimize_output_type: bool = False,
    agent_usage_limits: _usage.UsageLimits | None = None,
    gepa_usage_limits: _usage.UsageLimits | None = None,
) -> GepaOptimizationResult:
    """Optimizes a pydantic-ai agent (and optional signature inputs) using the GEPA graph backend.

    This is the main entry point for prompt optimization. It takes a pydantic-ai
    agent and a dataset, and returns optimized prompts.

    Args:
        agent: The pydantic-ai agent to optimize.
        trainset: Training dataset specification (sequence, DataLoader, or async factory).
        metric: Function that computes (score, feedback) for each instance.
                The feedback (second element of tuple) is optional but recommended.
                If provided, it will be used to guide the optimization process.
        valset: Optional validation dataset specification. Defaults to ``trainset`` when omitted.
        input_type: Optional structured input specification whose instructions and
            field descriptions should be optimized alongside the agent's prompts.
        skills: Optional skills (either a ``SkillsFS`` instance or a directory path)
            whose description/body components should be optimized and made available
            to the student agent via tools during evaluation.
        skills_search_backend: Optional search backend for skills discovery. When provided,
            it is passed through to the student toolset so searches can be backed by
            external indexes (e.g., TurboPuffer) and incorporate per-candidate overlays.
        skills_capabilities: Optional set of capabilities for skills tools (e.g. {SkillCapability.READ}).
            By default, GEPA restricts to READ capabilities to avoid untrusted script execution.

        reflection_config: Configuration for the reflection agent (model, include_case_metadata,
            include_expected_output, example_bank). When None, reflection runs with default settings.
        candidate_selection_strategy: Strategy for selecting candidates ('pareto' or 'current_best').
        skip_perfect_score: Whether to skip updating if perfect score achieved on minibatch.
        reflection_minibatch_size: Number of examples to use for reflection in each proposal.
        perfect_score: Score threshold treated as perfect when `skip_perfect_score` is enabled.
        track_component_hypotheses: When True, store the reflection hypothesis metadata with
            each component version and surface it to future reflection prompts.

        # Merge-based configuration
        use_merge: Whether to use the merge strategy for combining candidates.
        max_merge_invocations: Maximum number of merge invocations to perform.

        # Budget
        max_metric_calls: Maximum number of metric evaluations (budget).
        max_iterations: Optional cap on the number of GEPA loop iterations.

        # Caching configuration
        enable_cache: Whether to enable caching of metric results for resumable runs.
        cache_dir: Directory to store cache files. If None, uses '.gepa_cache' in current directory.
        cache_verbose: Whether to log cache hits and misses.

        show_progress: Display a Rich progress bar tied to the evaluation budget.
        # Reproducibility
        seed: Random seed for reproducibility.
        raise_on_exception: Whether to raise exceptions or continue on errors.

        # Testing support
        deterministic_proposer: For testing - a deterministic proposal function.

        # Reflection sampler
        reflection_sampler: Optional sampler for reflection records. If provided,
                               it will be called to sample records when needed. If None,
                               all reflection records are kept without sampling.

        # Tool configuration
        optimize_tools: Enable optimization of tool descriptions and parameter schemas
            for plain agents without requiring a SignatureAgent wrapper.
        optimize_output_type: Enable optimization of output tool descriptions and parameter schemas
            derived from the agent's output_type (via prepare_output_tools).
        agent_usage_limits: Optional UsageLimits applied to each individual agent run
            (e.g., cap tool calls per evaluation to prevent runaway tool loops). When None,
            no per-run usage limits are enforced.
        gepa_usage_limits: Optional UsageLimits applied cumulatively across the entire
            GEPA optimization run. When provided, GEPA stops once the aggregated usage
            exceeds this budget.

    Returns:
        GepaOptimizationResult with the best candidate and metadata.
    """
    train_loader = await resolve_dataset(trainset, name="trainset")
    val_loader = (
        await resolve_dataset(valset, name="valset") if valset is not None else None
    )

    if optimize_output_type:
        # Ensure output tool optimizer is installed before seed extraction
        try:
            get_or_create_output_tool_optimizer(agent)
        except Exception:
            logfire.debug(
                "Unable to install output tool optimizer; continuing without output optimization",
                exc_info=True,
            )

    skills_fs = _coerce_skills_fs(skills)

    extracted_seed_candidate = extract_seed_candidate_with_input_type(
        agent=agent,
        input_type=input_type,
        optimize_output_type=optimize_output_type,
    )
    if seed_candidate is None:
        normalized_seed_candidate = extracted_seed_candidate
    else:
        normalized_seed_candidate = ensure_component_values(seed_candidate)
        if sorted(extracted_seed_candidate.keys()) != sorted(
            normalized_seed_candidate.keys()
        ):
            raise ValueError(
                "Seed candidate keys do not match extracted seed candidate keys"
            )

    if (
        skills_fs is not None
        and not optimize_tools
        and get_tool_optimizer(agent) is None
    ):
        # When skills are enabled, the student agent gains default skills tools
        # (list/search/load). Install a restricted tool optimizer so these tool schemas
        # can be optimized even when optimize_tools=False.
        #
        # This is intentionally after seed normalization so user-provided seed candidates
        # don't have to anticipate tool:* keys (they can be hydrated later).
        manager = get_or_create_tool_optimizer(
            agent,
            allowed_tools={
                "list_skills",
                "search_skills",
                "load_skill",
                "load_skill_file",
            },
        )

        try:
            toolset = create_skills_toolset(
                skills_fs,
                search_backend=skills_search_backend,
                capabilities=skills_capabilities,
            )
            manager.record_model_request(
                function_tools=[tool.tool_def for tool in toolset.tools.values()],
            )
        except Exception:
            # If pre-seeding fails, the optimizer will still learn tool schemas from
            # actual model runs (when available).
            logfire.debug(
                "Failed to pre-seed skills tool components; continuing",
                exc_info=True,
            )

    cache_manager = None
    if enable_cache:
        cache_manager = CacheManager(
            cache_dir=cache_dir,
            enabled=True,
            verbose=cache_verbose,
        )

    adapter = create_adapter(
        agent=agent,
        metric=metric,
        input_type=input_type,
        skills_fs=skills_fs,
        skills_search_backend=skills_search_backend,
        skills_capabilities=skills_capabilities,
        cache_manager=cache_manager,
        optimize_tools=optimize_tools,
        optimize_output_type=optimize_output_type,
        agent_usage_limits=agent_usage_limits,
        gepa_usage_limits=gepa_usage_limits,
    )

    config = _build_gepa_config(
        max_metric_calls=max_metric_calls,
        max_iterations=max_iterations,
        reflection_minibatch_size=reflection_minibatch_size,
        skip_perfect_score=skip_perfect_score,
        perfect_score=perfect_score,
        module_selector=module_selector,
        seed_candidate=normalized_seed_candidate,
        candidate_selection_strategy=candidate_selection_strategy,
        use_merge=use_merge,
        max_merge_invocations=max_merge_invocations,
        seed=seed,
        reflection_config=reflection_config,
        reflection_sampler=reflection_sampler,
        track_component_hypotheses=track_component_hypotheses,
    )

    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    memory_exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(memory_exporter)
    provider = trace.get_tracer_provider()

    # logfire configures a global tracer provider.
    # Try getting logfire's provider if default has no add_span_processor
    if not hasattr(provider, "add_span_processor"):
        try:
            provider = logfire.get_tracer_provider()
        except Exception:
            pass

    if hasattr(provider, "add_span_processor"):
        provider.add_span_processor(processor)

    deps = create_deps(
        adapter,
        config,
        seed_candidate=normalized_seed_candidate,
        memory_exporter=memory_exporter,
    )
    if deterministic_proposer is not None:
        deps.proposal_generator = deterministic_proposer

    graph = create_gepa_graph(config=config)
    state = GepaState(
        config=config,
        training_set=train_loader,
        validation_set=val_loader,
    )

    gepa_result: GepaResult | None = None
    try:
        run_output = None
        with OptimizationProgress(
            total=config.max_evaluations,
            description="Optimizing agent",
            enabled=show_progress,
        ) as progress_bar:
            previous_node_name: str | None = None
            async with graph.iter(state=state, deps=deps) as run:
                async for event in run:
                    current_node_name = _describe_graph_event(graph, event)
                    progress_bar.update(
                        state.total_evaluations,
                        current_node=current_node_name,
                        previous_node=previous_node_name,
                        best_score=state.best_score,
                    )
                    if current_node_name:
                        previous_node_name = current_node_name
                run_output = run.output
            progress_bar.update(
                state.total_evaluations,
                best_score=state.best_score,
            )
        if run_output is None:
            raise RuntimeError("GEPA graph run did not produce a result.")
        gepa_result = run_output
    except UsageBudgetExceeded as exc:
        state.mark_stopped(reason="Usage budget exceeded")
        logfire.info(
            "Optimization stopped due to usage budget",
            exception=exc,
            total_evaluations=state.total_evaluations,
        )
        logfire.info(
            "Returning best-so-far candidate after usage budget exceeded",
            total_evaluations=state.total_evaluations,
        )
        gepa_result = GepaResult.from_state(state)
    except Exception as exc:
        if raise_on_exception:
            raise
        logfire.error(
            "Optimization failed",
            exception=exc,
        )
        logfire.error(
            "Optimization failed while returning fallback result",
            exception=exc,
        )
        return _fallback_result(normalized_seed_candidate)

    if gepa_result is None:
        raise RuntimeError("GEPA optimization did not produce a result.")

    best_candidate_model = gepa_result.best_candidate or state.get_best_candidate()
    if best_candidate_model is None:
        best_candidate_dict = normalized_seed_candidate
    else:
        best_candidate_dict = {
            name: component.model_copy()
            for name, component in best_candidate_model.components.items()
        }

    if gepa_result.original_candidate is not None:
        original_candidate_model = gepa_result.original_candidate
    elif state.candidates:
        original_candidate_model = state.candidates[0]
    else:
        original_candidate_model = None

    if original_candidate_model is None:
        original_candidate_dict = normalized_seed_candidate
    else:
        original_candidate_dict = {
            name: component.model_copy()
            for name, component in original_candidate_model.components.items()
        }

    result = GepaOptimizationResult(
        best_candidate=best_candidate_dict,
        best_score=gepa_result.best_score or 0.0,
        original_candidate=original_candidate_dict,
        original_score=gepa_result.original_score,
        num_iterations=gepa_result.iterations,
        num_metric_calls=gepa_result.total_evaluations,
        raw_result=gepa_result,
        evaluation_errors=gepa_result.evaluation_errors,
    )

    if cache_manager and cache_verbose:
        stats = cache_manager.get_cache_stats()
        logfire.info("Cache stats", stats=stats)

    return result


def _build_gepa_config(
    *,
    max_metric_calls: int,
    max_iterations: int | None,
    reflection_minibatch_size: int,
    skip_perfect_score: bool,
    perfect_score: float,
    module_selector: str,
    seed_candidate: CandidateMap,
    candidate_selection_strategy: str,
    use_merge: bool,
    max_merge_invocations: int,
    seed: int,
    reflection_config: ReflectionConfig | None,
    reflection_sampler: ReflectionSampler | None,
    track_component_hypotheses: bool,
) -> GepaConfig:
    component_selector = _resolve_component_selector(
        module_selector, len(seed_candidate)
    )
    candidate_selector: CandidateSelectorStrategy = _resolve_candidate_selector(
        candidate_selection_strategy
    )

    return GepaConfig(
        max_evaluations=max_metric_calls,
        max_iterations=max_iterations,
        minibatch_size=reflection_minibatch_size,
        perfect_score=float(perfect_score),
        skip_perfect_score=skip_perfect_score,
        component_selector=component_selector,
        candidate_selector=candidate_selector,
        use_merge=use_merge,
        max_total_merges=max_merge_invocations,
        seed=seed,
        reflection_config=reflection_config,
        reflection_sampler=reflection_sampler,
        track_component_hypotheses=track_component_hypotheses,
    )


def _resolve_component_selector(
    selector: str,
    component_count: int,
) -> Literal["round_robin", "all", "reflection"]:
    if selector not in {"round_robin", "all", "reflection"}:
        raise ValueError(
            "module_selector must be 'round_robin', 'all', or 'reflection'."
        )
    if selector == "round_robin" and component_count <= 1:
        return "all"
    return cast(Literal["round_robin", "all", "reflection"], selector)


def _resolve_candidate_selector(strategy: str) -> CandidateSelectorStrategy:
    try:
        return CandidateSelectorStrategy(strategy)
    except ValueError as error:
        raise ValueError(
            "candidate_selection_strategy must be 'pareto' or 'current_best'."
        ) from error


def _coerce_skills_fs(skills: SkillsFS | str | Path | None) -> SkillsFS | None:
    if skills is None:
        return None
    if isinstance(skills, SkillsFS):
        return skills
    return SkillsFS.from_disk(Path(skills))


def _describe_graph_event(
    graph: Any,
    event: EndMarker[GepaResult] | Sequence[GraphTask],
) -> str | None:
    if isinstance(event, EndMarker):
        return "End"

    node_ids = {task.node_id for task in event}
    if not node_ids:
        return None

    names = sorted(_node_label(graph, node_id) for node_id in node_ids)
    return ", ".join(names)


def _node_label(graph: Any, node_id) -> str:
    node = getattr(graph, "nodes", {}).get(node_id)
    if node is None:
        return str(node_id)
    label = getattr(node, "label", None)
    if label:
        return label
    node_identifier = getattr(node, "id", None)
    if node_identifier is not None:
        return str(node_identifier)
    if hasattr(node, "__class__"):
        return node.__class__.__name__
    return str(node_id)


def _fallback_result(
    seed_candidate: CandidateMap,
) -> GepaOptimizationResult:
    candidate_copy = {
        name: component.model_copy() for name, component in seed_candidate.items()
    }
    return GepaOptimizationResult(
        best_candidate=candidate_copy,
        best_score=0.0,
        original_candidate=dict(seed_candidate),
        original_score=None,
        num_iterations=0,
        num_metric_calls=0,
        raw_result=None,
        evaluation_errors=[],
    )

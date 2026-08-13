"""The built-in engine backed by the reflective GEPA graph."""

from __future__ import annotations

from typing import Any, cast

from pydantic_ai import usage as _usage

from ..adapters.agent_adapter import create_adapter
from ..exceptions import UsageBudgetExceeded
from ..gepa_graph import create_deps, create_gepa_graph
from ..gepa_graph.models import CandidateMap, GepaConfig, GepaResult, GepaState
from ..runner import _resolve_candidate_selector, _resolve_component_selector
from ..types import ReflectionConfig
from .base import (
    BudgetTracker,
    EngineConfig,
    EngineEvent,
    EngineResult,
    OptimizationTask,
)
from .registry import register_engine


class GepaEngine:
    """Run the existing reflective GEPA graph under a shared metric-call budget.

    ``engine_config`` supports the scalar GEPA knobs accepted by
    :func:`pydantic_ai_gepa.runner.optimize_agent`.  ``reflection_config`` and
    ``agent_usage_limits`` are intentionally only accepted as their already
    constructed Python objects; registry-based configuration is therefore
    limited to the scalar knobs.

    When ``EngineConfig.stop_at_score`` is set, it overrides ``perfect_score``
    and enables ``skip_perfect_score``.  This maps the engine-level threshold
    to the graph's existing early-stop behavior.
    """

    name = "gepa"

    def __init__(self, config: EngineConfig) -> None:
        """Capture GEPA-specific options supplied through ``engine_config``."""
        self._engine_config = config.engine_config

    async def run(
        self,
        task: OptimizationTask,
        config: EngineConfig,
        budget: BudgetTracker,
    ) -> EngineResult:
        """Optimize ``task`` by running the reflective graph unchanged."""
        budget.check()

        seed_candidate = await task.seed_candidate()
        remaining_budget = budget.remaining
        reflection_config = self._complex_option("reflection_config", ReflectionConfig)
        agent_usage_limits = self._complex_option(
            "agent_usage_limits", _usage.UsageLimits
        )

        adapter = create_adapter(
            agent=task.agent,
            metric=task.metric,
            input_type=task.input_type,
            skills_fs=task.skills_fs,
            skills_capabilities=task.skills_capabilities,
            case_factory=task.case_factory,
            cache_manager=None,
            optimize_tools=self._option("optimize_tools", False),
            optimize_output_type=self._option("optimize_output_type", False),
            agent_usage_limits=agent_usage_limits,
        )
        gepa_config = self._build_config(
            config=config,
            max_evaluations=min(config.max_metric_calls, remaining_budget),
            seed_candidate=seed_candidate,
            reflection_config=reflection_config,
        )
        state = GepaState(
            config=gepa_config,
            training_set=await task.train_loader(),
            validation_set=await task.val_loader(),
        )
        deps = create_deps(
            adapter,
            gepa_config,
            seed_candidate=seed_candidate,
            memory_exporter=None,
        )
        graph = create_gepa_graph(config=gepa_config)
        # Reserve before the graph can invoke a rollout. The graph's own cap
        # is identical; unused capacity is refunded after its exact telemetry
        # is known, so result accounting remains honest.
        reserved = gepa_config.max_evaluations
        budget.spend(reserved)

        try:
            run_output: GepaResult | None = None
            async with graph.iter(state=state, deps=deps) as run:
                async for _event in run:
                    pass
                run_output = run.output
            if run_output is None:
                raise RuntimeError("GEPA graph run did not produce a result.")
            gepa_result = run_output
        except UsageBudgetExceeded:
            state.mark_stopped(reason="Usage budget exceeded")
            gepa_result = GepaResult.from_state(state)

        num_metric_calls = state.total_evaluations
        history: list[EngineEvent] = []
        if num_metric_calls > reserved:
            raise RuntimeError(
                f"GEPA exceeded its pre-reserved slice ({num_metric_calls}>{reserved})."
            )
        budget.refund(reserved - num_metric_calls)

        best_candidate = _candidate_map(gepa_result.best_candidate, seed_candidate)
        history.append(
            EngineEvent(
                kind="summary",
                data={
                    "iterations": gepa_result.iterations,
                    "total_evaluations": num_metric_calls,
                    "original_score": gepa_result.original_score,
                    "stop_reason": gepa_result.stop_reason,
                },
            )
        )
        return EngineResult(
            engine=self.name,
            best_candidate=best_candidate,
            best_score=gepa_result.best_score or 0.0,
            num_metric_calls=num_metric_calls,
            history=history,
        )

    def _build_config(
        self,
        *,
        config: EngineConfig,
        max_evaluations: int,
        seed_candidate: CandidateMap,
        reflection_config: ReflectionConfig | None,
    ) -> GepaConfig:
        """Shape shared and engine-specific options into the graph configuration."""
        perfect_score = self._option("perfect_score", 1.0)
        skip_perfect_score = self._option("skip_perfect_score", True)
        if config.stop_at_score is not None:
            perfect_score = config.stop_at_score
            skip_perfect_score = True

        module_selector = self._option("module_selector", "round_robin")
        candidate_selection_strategy = self._option(
            "candidate_selection_strategy", "pareto"
        )
        return GepaConfig(
            max_evaluations=max_evaluations,
            max_iterations=config.max_iterations,
            minibatch_size=self._option("reflection_minibatch_size", 3),
            perfect_score=float(perfect_score),
            skip_perfect_score=skip_perfect_score,
            component_selector=_resolve_component_selector(
                module_selector, len(seed_candidate)
            ),
            candidate_selector=_resolve_candidate_selector(
                candidate_selection_strategy
            ),
            use_merge=self._option("use_merge", False),
            max_total_merges=self._option("max_merge_invocations", 5),
            seed=config.seed,
            reflection_config=reflection_config,
            track_component_hypotheses=self._option(
                "track_component_hypotheses", False
            ),
        )

    def _option(self, name: str, default: Any) -> Any:
        """Return a GEPA option, preserving pydantic validation at config creation."""
        return self._engine_config.get(name, default)

    def _complex_option(self, name: str, expected_type: type[Any]) -> Any | None:
        """Return a directly constructed complex option or reject JSON-like values."""
        value = self._engine_config.get(name)
        if value is None:
            return None
        if not isinstance(value, expected_type):
            raise TypeError(
                f"engine_config[{name!r}] must be a {expected_type.__name__} instance; "
                "registry-based engine configuration supports scalar GEPA options only."
            )
        return value


def _candidate_map(candidate: Any | None, fallback: CandidateMap) -> CandidateMap:
    """Copy a graph candidate into the engine's public candidate mapping."""
    if candidate is None:
        return fallback
    components = cast(CandidateMap, candidate.components)
    return {name: component.model_copy() for name, component in components.items()}


register_engine(GepaEngine.name, GepaEngine, replace=True)


__all__ = ["GepaEngine"]

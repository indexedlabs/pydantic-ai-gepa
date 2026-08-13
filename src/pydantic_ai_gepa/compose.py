"""Composable, budget-accounted optimization pipelines, including Omni."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from .acceptance import compare_candidate_samples
from .engines import (
    BudgetTracker,
    CandidateEvaluation,
    EngineConfig,
    EngineEvent,
    EngineResult,
    OptimizationTask,
    get_engine,
)
from .gepa_graph.models import CandidateMap


class FairVote(BaseModel):
    """Repeated matched samples and the selection decision for one candidate."""

    candidate_index: int
    samples: list[float]
    mean_score: float
    objective_scores: dict[str, float] = Field(default_factory=dict)
    per_case_scores: dict[str, float] = Field(default_factory=dict)
    per_case_objective_scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    selectable: bool = True


class PipelineResult(BaseModel):
    """Inspectable result of a composed run.

    ``total_metric_calls`` retains the historic optimization-only meaning.
    Comparison and reporting calls are separately visible and the sum is
    provided as ``accounted_metric_calls`` so no evaluator work is hidden.
    """

    results: list[EngineResult]
    best: EngineResult
    best_index: int
    fair_scores: list[float]
    total_metric_calls: int
    comparison_metric_calls: int = 0
    reporting_metric_calls: int = 0
    fair_votes: list[FairVote] = Field(default_factory=list)
    decision: dict[str, Any] = Field(default_factory=dict)
    test_score: float | None = None
    phases: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @property
    def accounted_metric_calls(self) -> int:
        return (
            self.total_metric_calls
            + self.comparison_metric_calls
            + self.reporting_metric_calls
        )


class OmniPlan(BaseModel):
    """Explicit two-phase Omni plan.

    Phase-one configurations each own their fixed matched slice via
    ``EngineConfig.max_metric_calls``.  The phase-two configuration creates a
    fresh engine instance from the fairly selected winner.  Comparison and
    reporting budget are separate named currencies, rather than free calls.
    """

    phase_one: list[EngineConfig]
    phase_two: EngineConfig
    phase_one_metric_calls: int
    phase_two_metric_calls: int
    comparison_metric_calls: int | None = None
    continuation_comparison_metric_calls: int | None = None
    reporting_metric_calls: int | None = None
    fair_vote_repetitions: int = Field(default=3, ge=1, le=5)
    fair_vote_max_repetitions: int = Field(default=5, ge=1, le=5)
    acceptance_confidence: float = Field(default=0.9, gt=0.0, lt=1.0)
    acceptance_min_delta: float = Field(default=0.0, ge=0.0)
    selection_mode: Literal["instance", "objective", "hybrid", "cartesian"] = "instance"

    def model_post_init(self, __context: Any) -> None:
        if not self.phase_one:
            raise ValueError("Omni phase_one must contain at least one engine.")
        if self.fair_vote_max_repetitions < self.fair_vote_repetitions:
            raise ValueError(
                "fair_vote_max_repetitions must be >= fair_vote_repetitions."
            )
        if not math.isfinite(self.acceptance_min_delta):
            raise ValueError("acceptance_min_delta must be finite.")
        if (
            sum(config.max_metric_calls for config in self.phase_one)
            != self.phase_one_metric_calls
        ):
            raise ValueError(
                "phase_one_metric_calls must equal the sum of phase-one engine slices; "
                "matched phase-one slices are explicit."
            )
        if self.phase_two.max_metric_calls != self.phase_two_metric_calls:
            raise ValueError(
                "phase_two_metric_calls must equal phase_two.max_metric_calls."
            )
        if len({config.max_metric_calls for config in self.phase_one}) != 1:
            raise ValueError("All phase-one engine slices must be matched.")
        if len({config.engine for config in self.phase_one}) != len(self.phase_one):
            raise ValueError("Phase-one Omni engines must be distinct families.")


class _SeededTask:
    """Delegate a task while supplying a fixed seed candidate to an engine."""

    def __init__(self, task: OptimizationTask, seed: CandidateMap) -> None:
        self._task = task
        self._seed = seed

    async def seed_candidate(self) -> CandidateMap:
        return self._seed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._task, name)


class _EngineTaskView:
    """A capability view that prevents engines from accessing held-out data."""

    def __init__(self, task: OptimizationTask) -> None:
        self.__task = task

    async def seed_candidate(self) -> CandidateMap:
        return await self.__task.seed_candidate()

    async def train_loader(self) -> Any:
        return await self.__task.train_loader()

    async def val_loader(self) -> Any:
        return await self.__task.val_loader()

    @property
    def agent(self) -> Any:
        return self.__task.agent

    @property
    def metric(self) -> Any:
        return self.__task.metric

    @property
    def input_type(self) -> Any:
        return self.__task.input_type

    @property
    def skills_fs(self) -> Any:
        return self.__task.skills_fs

    @property
    def skills_capabilities(self) -> Any:
        return self.__task.skills_capabilities

    @property
    def case_factory(self) -> Any:
        return self.__task.case_factory

    @property
    def concurrency(self) -> int:
        return self.__task.concurrency

    async def evaluate(
        self, candidate: CandidateMap, **kwargs: Any
    ) -> CandidateEvaluation:
        if kwargs.pop("dataset", "validation") != "validation":
            raise PermissionError("Engines cannot access the reporting-only test_set.")
        return await self.__task.evaluate(candidate, **kwargs)

    @property
    def test_set(self) -> None:
        return None


SelectionRule = Callable[[Sequence[FairVote]], int]


async def optimize_parallel(
    task: OptimizationTask,
    configs: Sequence[EngineConfig],
    *,
    max_metric_calls: int,
) -> list[EngineResult]:
    """Run engines concurrently in pre-reserved isolated budget slices."""
    budget = BudgetTracker(max_metric_calls)
    return await _run_parallel(task, configs, budget, legacy_fallback=True)


async def optimize_best_of(
    task: OptimizationTask,
    configs: Sequence[EngineConfig],
    *,
    max_metric_calls: int,
    comparison_metric_calls: int | None = None,
    fair_vote_repetitions: int = 1,
    fair_vote_max_repetitions: int = 1,
    selection_mode: Literal[
        "instance", "objective", "hybrid", "cartesian"
    ] = "instance",
    selection_rule: SelectionRule | None = None,
    acceptance_confidence: float = 0.9,
    acceptance_min_delta: float = 0.0,
) -> PipelineResult:
    """Parallel exploration plus a charged repeated matched comparison."""
    await _require_comparison_budget(
        task,
        candidates=len(configs),
        repetitions=fair_vote_max_repetitions,
        requested=comparison_metric_calls,
    )
    budget = BudgetTracker(max_metric_calls)
    results = await _run_parallel(task, configs, budget, legacy_fallback=True)
    winner, votes, comparison_budget, decision = await _select_fair_winner(
        task,
        results,
        metric_calls=comparison_metric_calls,
        repetitions=fair_vote_repetitions,
        max_repetitions=fair_vote_max_repetitions,
        mode=selection_mode,
        selection_rule=selection_rule,
        confidence=acceptance_confidence,
        min_delta=acceptance_min_delta,
        baseline_index=None,
    )
    return _pipeline_result(results, winner, votes, budget, comparison_budget, decision)


async def optimize_sequential(
    task: OptimizationTask,
    configs: Sequence[EngineConfig],
    *,
    max_metric_calls: int,
    comparison_metric_calls: int | None = None,
) -> PipelineResult:
    """Run fresh sequential stages while never adopting a regressing seed."""
    if not configs:
        raise ValueError("At least one engine config is required.")
    budget = BudgetTracker(max_metric_calls)
    comparison_budget = BudgetTracker(
        await _require_comparison_budget(
            task,
            candidates=len(configs),
            repetitions=1,
            requested=comparison_metric_calls,
            include_seed=True,
        )
    )
    seed = await task.seed_candidate()
    seed_evaluation = await task.evaluate(seed, budget=comparison_budget)
    seed_score = seed_evaluation.score
    adopted_result = EngineResult(
        engine="seed",
        best_candidate=seed,
        best_score=seed_score,
        num_metric_calls=0,
        history=[EngineEvent(kind="seed", data={"fair_score": seed_score})],
    )
    adopted_index = -1
    results: list[EngineResult] = []
    votes: list[FairVote] = []
    phases: list[dict[str, Any]] = []
    for index, config in enumerate(configs):
        # A stage slice is reserved independently. It cannot steal capacity
        # allocated to later stages, which keeps accounting deterministic.
        if budget.remaining == 0:
            phases.append(
                {"stage": index, "engine": config.engine, "skipped": "budget_exhausted"}
            )
            break
        slice_size = min(config.max_metric_calls, budget.remaining)
        local = budget.reserve_slice(slice_size)
        try:
            stage_task = cast(
                OptimizationTask,
                _SeededTask(cast(OptimizationTask, _EngineTaskView(task)), seed),
            )
            result = await get_engine(config.engine, config).run(
                stage_task, config, local
            )
            _reconcile_engine_result(config, result, local)
        finally:
            budget.release_slice(local)
        results.append(result)
        evaluation = await task.evaluate(
            result.best_candidate, budget=comparison_budget
        )
        vote = FairVote(
            candidate_index=index,
            samples=[evaluation.score],
            mean_score=evaluation.score,
            objective_scores=evaluation.objective_scores,
            selectable=evaluation.selectable,
        )
        votes.append(vote)
        adopted = evaluation.selectable and evaluation.score >= seed_score
        phases.append(
            {
                "stage": index,
                "engine": config.engine,
                "adopted": adopted,
                "seed_score": seed_score,
                "score": evaluation.score,
            }
        )
        if adopted:
            seed, seed_score, adopted_result, adopted_index = (
                result.best_candidate,
                evaluation.score,
                result,
                index,
            )
    return PipelineResult(
        results=results,
        best=adopted_result,
        best_index=adopted_index,
        fair_scores=[vote.mean_score for vote in votes],
        total_metric_calls=budget.spent,
        comparison_metric_calls=comparison_budget.spent,
        fair_votes=votes,
        decision={"kind": "monotonic_sequential", "seed_score": seed_score},
        phases=phases,
    )


async def optimize_vote(
    task: OptimizationTask,
    configs: Sequence[EngineConfig],
    *,
    max_metric_calls: int,
    comparison_metric_calls: int | None = None,
    fair_vote_repetitions: int = 1,
    fair_vote_max_repetitions: int = 1,
) -> PipelineResult:
    """Alias for a repeated, matched, charged cross-engine vote."""
    return await optimize_best_of(
        task,
        configs,
        max_metric_calls=max_metric_calls,
        comparison_metric_calls=comparison_metric_calls,
        fair_vote_repetitions=fair_vote_repetitions,
        fair_vote_max_repetitions=fair_vote_max_repetitions,
    )


async def optimize_omni(
    task: OptimizationTask,
    plan: OmniPlan,
    *,
    selection_rule: SelectionRule | None = None,
) -> PipelineResult:
    """Run the first-class Omni explore → fair-vote → fresh-continuation flow."""
    # Validate every named non-optimization evaluator budget before any engine
    # or rollout can start. An underfunded fair vote/report is a plan error,
    # never a partially optimized run.
    await _require_comparison_budget(
        task,
        candidates=len(plan.phase_one) + 1,
        repetitions=plan.fair_vote_max_repetitions,
        requested=plan.comparison_metric_calls,
    )
    await _require_comparison_budget(
        task,
        candidates=2,
        repetitions=plan.fair_vote_max_repetitions,
        requested=plan.continuation_comparison_metric_calls,
    )
    if task.test_set is not None:
        test_case_count = len(await task._test_cases_for_reporting())
        if (
            plan.reporting_metric_calls is not None
            and plan.reporting_metric_calls < test_case_count
        ):
            raise ValueError(
                "reporting_metric_calls cannot fund the held-out test_set."
            )
    phase_one_budget = BudgetTracker(plan.phase_one_metric_calls)
    phase_one = await _run_parallel(task, plan.phase_one, phase_one_budget)
    seed = await task.seed_candidate()
    seed_result = EngineResult(
        engine="seed",
        best_candidate=seed,
        best_score=0.0,
        num_metric_calls=0,
        history=[EngineEvent(kind="seed")],
    )
    comparable = [seed_result, *phase_one]
    winner_index, votes, comparison_budget, decision = await _select_fair_winner(
        task,
        comparable,
        metric_calls=plan.comparison_metric_calls,
        repetitions=plan.fair_vote_repetitions,
        max_repetitions=plan.fair_vote_max_repetitions,
        mode=plan.selection_mode,
        selection_rule=selection_rule,
        confidence=plan.acceptance_confidence,
        min_delta=plan.acceptance_min_delta,
        baseline_index=0,
    )
    seed_result = seed_result.model_copy(update={"best_score": votes[0].mean_score})
    comparable[0] = seed_result
    if (
        votes[winner_index].mean_score < votes[0].mean_score
        or not votes[winner_index].selectable
    ):
        winner_index = 0
        decision = {**decision, "seed_monotonic_override": True}
    winner = comparable[winner_index]
    # A fresh registry lookup (rather than reusing phase one) is intentional.
    phase_two_budget = BudgetTracker(plan.phase_two_metric_calls)
    seeded_task = cast(
        OptimizationTask,
        _SeededTask(
            cast(OptimizationTask, _EngineTaskView(task)), winner.best_candidate
        ),
    )
    continuation = await get_engine(plan.phase_two.engine, plan.phase_two).run(
        seeded_task, plan.phase_two, phase_two_budget
    )
    _reconcile_engine_result(plan.phase_two, continuation, phase_two_budget)
    (
        continuation_winner_index,
        continuation_votes,
        continuation_budget,
        continuation_decision,
    ) = await _select_fair_winner(
        task,
        [winner, continuation],
        metric_calls=plan.continuation_comparison_metric_calls,
        repetitions=plan.fair_vote_repetitions,
        max_repetitions=plan.fair_vote_max_repetitions,
        mode=plan.selection_mode,
        selection_rule=None,
        confidence=plan.acceptance_confidence,
        min_delta=plan.acceptance_min_delta,
        baseline_index=0,
    )
    # A continuation is adopted only after the shared uncertainty-aware
    # acceptance primitive cleared the configured practical delta.
    adopted_continuation = (
        continuation_winner_index == 1
        and continuation_votes[1].selectable
        and continuation_decision.get("acceptance", {}).get("verdict") == "accepted"
    )
    final = continuation if adopted_continuation else winner
    test_case_count = (
        len(await task._test_cases_for_reporting()) if task.test_set is not None else 0
    )
    final_budget = (
        BudgetTracker(plan.reporting_metric_calls or test_case_count)
        if test_case_count
        else None
    )
    test_score: float | None = None
    reporting_calls = 0
    if test_case_count:
        assert final_budget is not None
        test = await task.evaluate(
            final.best_candidate, budget=final_budget, dataset="test"
        )
        test_score = test.score
        reporting_calls = test.num_cases
    return PipelineResult(
        results=[*comparable, continuation],
        best=final,
        best_index=len(comparable) if adopted_continuation else winner_index,
        fair_scores=[vote.mean_score for vote in votes],
        total_metric_calls=phase_one_budget.spent + phase_two_budget.spent,
        comparison_metric_calls=comparison_budget.spent + continuation_budget.spent,
        reporting_metric_calls=reporting_calls,
        fair_votes=votes,
        decision={
            **decision,
            "phase_two_engine": plan.phase_two.engine,
            "phase_two_seeded_from": winner_index,
            "phase_two_adopted": adopted_continuation,
            "continuation_winner": continuation_winner_index,
            "continuation_vote": continuation_decision,
            "continuation_samples": [vote.model_dump() for vote in continuation_votes],
        },
        test_score=test_score,
        phases=[
            {
                "phase": "explore",
                "budget": plan.phase_one_metric_calls,
                "engines": [config.engine for config in plan.phase_one],
            },
            {
                "phase": "compare",
                "budget": comparison_budget.max_metric_calls,
                "winner": winner_index,
            },
            {
                "phase": "continue",
                "budget": plan.phase_two_metric_calls,
                "engine": plan.phase_two.engine,
                "fresh_instance": True,
            },
            {
                "phase": "continuation_compare",
                "budget": continuation_budget.max_metric_calls,
            },
        ],
    )


async def optimize_adaptive_sequential(
    task: OptimizationTask,
    configs: Sequence[EngineConfig],
    *,
    max_metric_calls: int,
    plateau_min_improvement: float = 0.0,
    comparison_metric_calls: int | None = None,
    patience: int = 1,
    min_evaluations_per_engine: int = 1,
    max_switches: int | None = None,
    cycle: bool = False,
    max_slices: int | None = None,
) -> PipelineResult:
    """Run bounded fresh slices and switch only after observed plateaus."""
    if not configs:
        raise ValueError("At least one engine config is required.")
    if patience < 1 or min_evaluations_per_engine < 1:
        raise ValueError("patience and min_evaluations_per_engine must be positive.")
    if not math.isfinite(plateau_min_improvement) or plateau_min_improvement < 0:
        raise ValueError("plateau_min_improvement must be finite and non-negative.")
    if max_switches is not None and max_switches < 0:
        raise ValueError("max_switches must be non-negative.")
    if max_slices is not None and max_slices < 1:
        raise ValueError("max_slices must be positive.")
    budget = BudgetTracker(max_metric_calls)
    # One seed comparison plus one result comparison per possible slice.
    # Defaults are driven by affordable fresh engine slices, not the number of
    # engine families.  A first engine may improve for many slices before it
    # plateaus; hard caps remain explicit through ``max_slices``.
    smallest_slice = min(config.max_metric_calls for config in configs)
    slice_limit = max_slices or max(1, max_metric_calls // smallest_slice)
    possible_slices = slice_limit
    cap = await _comparison_cap(task, possible_slices, 1, None, include_seed=True)
    if comparison_metric_calls is not None and comparison_metric_calls < cap:
        raise ValueError(
            "comparison_metric_calls cannot fund all scheduled adaptive comparisons."
        )
    comparison = BudgetTracker(comparison_metric_calls or cap)
    seed = await task.seed_candidate()
    incumbent = await task.evaluate(seed, budget=comparison)
    best = EngineResult(
        engine="seed",
        best_candidate=seed,
        best_score=incumbent.score,
        num_metric_calls=0,
    )
    results: list[EngineResult] = []
    votes: list[FairVote] = []
    phases: list[dict[str, Any]] = []
    index = 0
    switches = 0
    plateau_rounds = 0
    calls_for_engine = 0
    while (
        budget.remaining > 0
        and len(results) < slice_limit
        and (cycle or index < len(configs))
    ):
        config = configs[index % len(configs)]
        slice_size = min(config.max_metric_calls, budget.remaining)
        if slice_size < min_evaluations_per_engine:
            break
        local = budget.reserve_slice(slice_size)
        try:
            seeded = cast(
                OptimizationTask,
                _SeededTask(
                    cast(OptimizationTask, _EngineTaskView(task)), best.best_candidate
                ),
            )
            result = await get_engine(config.engine, config).run(seeded, config, local)
            _reconcile_engine_result(config, result, local)
        finally:
            budget.release_slice(local)
        results.append(result)
        consumed = slice_size - local.remaining
        observed = await task.evaluate(result.best_candidate, budget=comparison)
        vote = FairVote(
            candidate_index=len(results) - 1,
            samples=[observed.score],
            mean_score=observed.score,
            objective_scores=observed.objective_scores,
            per_case_scores=observed.per_case_scores,
            per_case_objective_scores=observed.per_case_objective_scores,
            selectable=observed.selectable,
        )
        votes.append(vote)
        improved = (
            observed.selectable
            and observed.score > incumbent.score + plateau_min_improvement
        )
        if improved:
            best = result
            incumbent = observed
            plateau_rounds = 0
        else:
            plateau_rounds += 1
        # A driver that spends no metric calls has made no measurable search
        # progress. Count it as a plateau so bounded scheduling cannot keep
        # selecting it forever.
        no_progress = consumed == 0
        calls_for_engine += consumed
        if no_progress:
            plateau_rounds = max(plateau_rounds, patience)
        switch = plateau_rounds >= patience and (
            calls_for_engine >= min_evaluations_per_engine or no_progress
        )
        phases.append(
            {
                "stage": len(results) - 1,
                "engine": config.engine,
                "slice_metric_calls": slice_size,
                "improved": improved,
                "no_progress": no_progress,
                "plateau": switch,
                "incumbent_score": incumbent.score,
            }
        )
        if switch:
            # ``max_switches`` counts transitions that actually occur, not a
            # final plateau observation that merely terminates the scheduler.
            if max_switches is not None and switches >= max_switches:
                break
            index += 1
            switches += 1
            plateau_rounds = 0
            calls_for_engine = 0
        elif not cycle:
            # Continue the active engine only when it improved; its next slice
            # is a fresh instance seeded from the global incumbent.
            index = index
        else:
            index = index
        if not cycle and index >= len(configs):
            break
    best_index = -1 if best.engine == "seed" else results.index(best)
    return PipelineResult(
        results=results,
        best=best,
        best_index=best_index,
        fair_scores=[vote.mean_score for vote in votes],
        total_metric_calls=budget.spent,
        comparison_metric_calls=comparison.spent,
        fair_votes=votes,
        phases=phases,
        decision={
            "kind": "adaptive_sequential",
            "switches": switches,
            "patience": patience,
            "plateau_min_improvement": plateau_min_improvement,
        },
    )


async def _run_parallel(
    task: OptimizationTask,
    configs: Sequence[EngineConfig],
    budget: BudgetTracker,
    *,
    legacy_fallback: bool = False,
) -> list[EngineResult]:
    if not configs:
        raise ValueError("At least one engine config is required.")
    requested = sum(config.max_metric_calls for config in configs)
    if requested > budget.remaining and legacy_fallback:
        # Pre-Omni callers did not declare slices. Preserve their surface but
        # serialize it: a mutable shared budget is never handed to concurrent
        # engines, and each result is reconciled with actual consumption.
        engines = [get_engine(config.engine, config) for config in configs]
        results: list[EngineResult] = []
        engine_task = cast(OptimizationTask, _EngineTaskView(task))
        for engine, config in zip(engines, configs):
            before = budget.spent
            result = await engine.run(engine_task, config, budget)
            if result.num_metric_calls != budget.spent - before:
                raise RuntimeError(
                    f"Engine {config.engine!r} reported {result.num_metric_calls} calls, "
                    f"but consumed {budget.spent - before}."
                )
            results.append(result)
        return results
    if requested > budget.remaining:
        raise ValueError(
            f"Engine slices request {requested} metric calls but only {budget.remaining} are available."
        )
    slices = [budget.reserve_slice(config.max_metric_calls) for config in configs]
    engines = [get_engine(config.engine, config) for config in configs]
    engine_task = cast(OptimizationTask, _EngineTaskView(task))
    tasks = [
        asyncio.create_task(engine.run(engine_task, config, local))
        for engine, config, local in zip(engines, configs, slices)
    ]
    try:
        results = list(await asyncio.gather(*tasks))
        for config, result, local in zip(configs, results, slices):
            _reconcile_engine_result(config, result, local)
        return results
    finally:
        # ``gather`` propagates a sibling exception immediately.  Do not
        # release a local slice while a cancelled sibling could still spend it.
        unfinished_tasks = [pending for pending in tasks if not pending.done()]
        for pending in unfinished_tasks:
            pending.cancel()
        if unfinished_tasks:
            await asyncio.gather(*unfinished_tasks, return_exceptions=True)
        for local in slices:
            budget.release_slice(local)


def _reconcile_engine_result(
    config: EngineConfig, result: EngineResult, budget: BudgetTracker
) -> None:
    if result.num_metric_calls != budget.spent:
        raise RuntimeError(
            f"Engine {config.engine!r} reported {result.num_metric_calls} calls, "
            f"but consumed {budget.spent} from its allocated budget."
        )


async def _select_fair_winner(
    task: OptimizationTask,
    results: Sequence[EngineResult],
    *,
    metric_calls: int | None,
    repetitions: int,
    max_repetitions: int,
    mode: str,
    selection_rule: SelectionRule | None,
    confidence: float = 0.9,
    min_delta: float = 0.0,
    baseline_index: int | None = None,
) -> tuple[int, list[FairVote], BudgetTracker, dict[str, Any]]:
    if not results:
        raise ValueError("At least one engine result is required.")
    if not 1 <= repetitions <= max_repetitions <= 5:
        raise ValueError(
            "Fair voting requires 1 <= repetitions <= max_repetitions <= 5."
        )
    required = await _comparison_cap(task, len(results), max_repetitions, None)
    if metric_calls is not None and metric_calls < required:
        raise ValueError(
            f"Fair comparison requires {required} metric calls for {max_repetitions} matched rounds; got {metric_calls}."
        )
    budget = BudgetTracker(metric_calls if metric_calls is not None else required)
    evaluations: list[list[CandidateEvaluation]] = [[] for _ in results]
    # Round-robin evaluation provides matched repetitions: every candidate sees
    # the same immutable validation case order before anyone gets a tiebreak.
    rounds = 0
    for _ in range(repetitions):
        for index, result in enumerate(results):
            evaluations[index].append(
                await task.evaluate(result.best_candidate, budget=budget)
            )
        rounds += 1
    votes = _votes(evaluations)
    provisional, kind = _provisional_winner(votes, mode, selection_rule)
    if baseline_index is None:
        acceptance: dict[str, Any] = {"verdict": "not_applicable"}
        continue_rounds = _requires_tiebreak(votes)
    else:
        if not 0 <= baseline_index < len(votes):
            raise ValueError("baseline_index must reference a fair-vote candidate.")
        if not votes[baseline_index].selectable:
            raise ValueError(
                "The frozen fair-vote baseline must be selectable on every matched round."
            )
        acceptance = _acceptance_for_candidate(
            votes, baseline_index, provisional, confidence, min_delta
        )
        continue_rounds = (
            provisional != baseline_index and acceptance["verdict"] == "inconclusive"
        )
    while continue_rounds and rounds < max_repetitions:
        for index, result in enumerate(results):
            evaluations[index].append(
                await task.evaluate(result.best_candidate, budget=budget)
            )
        rounds += 1
        votes = _votes(evaluations)
        provisional, kind = _provisional_winner(votes, mode, selection_rule)
        if baseline_index is None:
            continue_rounds = _requires_tiebreak(votes)
        else:
            if not votes[baseline_index].selectable:
                raise ValueError(
                    "The frozen fair-vote baseline must be selectable on every matched round."
                )
            acceptance = _acceptance_for_candidate(
                votes, baseline_index, provisional, confidence, min_delta
            )
            continue_rounds = (
                provisional != baseline_index
                and acceptance["verdict"] == "inconclusive"
            )
    winner = (
        provisional
        if baseline_index is None
        or provisional == baseline_index
        or acceptance["verdict"] == "accepted"
        else baseline_index
    )
    return (
        winner,
        votes,
        budget,
        {
            "kind": kind,
            "repetitions": rounds,
            "matched_case_order": True,
            "baseline_index": baseline_index,
            "provisional_winner": provisional,
            "acceptance": acceptance,
            "selectable_candidates": [
                vote.candidate_index for vote in votes if vote.selectable
            ],
        },
    )


def _provisional_winner(
    votes: Sequence[FairVote], mode: str, selection_rule: SelectionRule | None
) -> tuple[int, str]:
    """Select a Pareto/custom provisional winner before baseline acceptance."""
    if selection_rule is not None:
        winner = selection_rule(votes)
        if not 0 <= winner < len(votes) or not votes[winner].selectable:
            raise ValueError("selection_rule must choose a selectable candidate index.")
        return winner, "custom"
    return _select_vote(votes, mode), mode


def _acceptance_for_candidate(
    votes: Sequence[FairVote],
    baseline_index: int,
    candidate_index: int,
    confidence: float,
    min_delta: float,
) -> dict[str, Any]:
    """Run the one shared acceptance primitive against the durable baseline."""
    if candidate_index == baseline_index:
        return {
            "verdict": "baseline",
            "confidence": confidence,
            "min_delta": min_delta,
        }
    return compare_candidate_samples(
        votes[baseline_index].samples,
        votes[candidate_index].samples,
        confidence=confidence,
        min_delta=min_delta,
    ).to_dict()


def _votes(evaluations: Sequence[Sequence[CandidateEvaluation]]) -> list[FairVote]:
    for samples in evaluations:
        _require_matching_coordinates(samples)
    if evaluations:
        coordinate_sets = [
            _coordinate_set(samples[0]) for samples in evaluations if samples
        ]
        if len(coordinate_sets) != len(evaluations) or any(
            coordinates != coordinate_sets[0] for coordinates in coordinate_sets[1:]
        ):
            raise ValueError(
                "Matched fair-vote candidates must expose identical scalar and objective coordinates."
            )
    return [
        FairVote(
            candidate_index=index,
            samples=[evaluation.score for evaluation in samples],
            mean_score=sum(evaluation.score for evaluation in samples) / len(samples),
            objective_scores=_mean_objectives(samples),
            per_case_scores=_mean_per_case_scores(samples),
            per_case_objective_scores=_mean_per_case_objectives(samples),
            selectable=all(evaluation.selectable for evaluation in samples),
        )
        for index, samples in enumerate(evaluations)
    ]


def _require_matching_coordinates(samples: Sequence[CandidateEvaluation]) -> None:
    if not samples:
        return
    first = _coordinate_set(samples[0])
    for sample in samples[1:]:
        if _coordinate_set(sample) != first:
            raise ValueError(
                "Matched fair-vote samples must expose identical scalar and objective coordinates."
            )


def _coordinate_set(
    evaluation: CandidateEvaluation,
) -> tuple[frozenset[str], frozenset[str], frozenset[tuple[str, str]]]:
    return (
        frozenset(evaluation.per_case_scores),
        frozenset(evaluation.objective_scores),
        frozenset(
            (case_id, name)
            for case_id, values in evaluation.per_case_objective_scores.items()
            for name in values
        ),
    )


def _mean_objectives(samples: Sequence[CandidateEvaluation]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for sample in samples:
        for name, value in sample.objective_scores.items():
            values.setdefault(name, []).append(value)
    return {name: sum(numbers) / len(numbers) for name, numbers in values.items()}


def _mean_per_case_scores(samples: Sequence[CandidateEvaluation]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for sample in samples:
        for case_id, score in sample.per_case_scores.items():
            values.setdefault(case_id, []).append(score)
    return {case_id: sum(scores) / len(scores) for case_id, scores in values.items()}


def _mean_per_case_objectives(
    samples: Sequence[CandidateEvaluation],
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = {}
    for sample in samples:
        for case_id, objectives in sample.per_case_objective_scores.items():
            for name, score in objectives.items():
                values.setdefault(case_id, {}).setdefault(name, []).append(score)
    return {
        case_id: {
            name: sum(scores) / len(scores) for name, scores in objectives.items()
        }
        for case_id, objectives in values.items()
    }


def _requires_tiebreak(votes: Sequence[FairVote]) -> bool:
    selectable = [vote for vote in votes if vote.selectable]
    if len(selectable) < 2:
        return False
    ordered = sorted(selectable, key=lambda vote: vote.mean_score, reverse=True)
    return abs(ordered[0].mean_score - ordered[1].mean_score) < 1e-12


def _select_vote(votes: Sequence[FairVote], mode: str) -> int:
    selectable = [vote for vote in votes if vote.selectable]
    if not selectable:
        raise ValueError("No selectable candidate was produced by the fair comparison.")
    if mode != "instance" and any(
        _frontier_coordinates(vote, mode) is None for vote in selectable
    ):
        raise ValueError(
            f"{mode} selection requires complete objective scores for every selectable candidate."
        )
    coordinates = [_frontier_coordinates(vote, mode) for vote in selectable]
    if any(coordinate is None for coordinate in coordinates) or any(
        coordinate is None or set(coordinate) != set(coordinates[0] or {})
        for coordinate in coordinates[1:]
    ):
        raise ValueError(
            "Fair-vote candidates must have identical frontier coordinates."
        )
    nondominated = [
        vote
        for vote in selectable
        if not any(
            _dominates(other, vote, mode) for other in selectable if other is not vote
        )
    ]
    return max(
        nondominated, key=lambda vote: (vote.mean_score, -vote.candidate_index)
    ).candidate_index


def _dominates(left: FairVote, right: FairVote, mode: str) -> bool:
    coordinates_left = _frontier_coordinates(left, mode)
    coordinates_right = _frontier_coordinates(right, mode)
    if (
        coordinates_left is None
        or coordinates_right is None
        or set(coordinates_left) != set(coordinates_right)
    ):
        return False
    return (
        bool(coordinates_left)
        and all(
            coordinates_left[key] >= coordinates_right[key] for key in coordinates_left
        )
        and any(
            coordinates_left[key] > coordinates_right[key] for key in coordinates_left
        )
    )


def _frontier_coordinates(vote: FairVote, mode: str) -> dict[str, float] | None:
    if mode not in {"instance", "objective", "hybrid", "cartesian"}:
        raise ValueError(
            "selection mode must be instance, objective, hybrid, or cartesian."
        )
    coordinates: dict[str, float] = {}
    if mode in {"instance", "hybrid"}:
        coordinates.update(
            {
                f"case:{case_id}": score
                for case_id, score in vote.per_case_scores.items()
            }
        )
    if mode in {"objective", "hybrid"}:
        if not vote.objective_scores:
            return None
        coordinates.update(
            {
                f"objective:{name}": score
                for name, score in vote.objective_scores.items()
            }
        )
    if mode == "cartesian":
        for case_id, objectives in vote.per_case_objective_scores.items():
            for name, score in objectives.items():
                coordinates[f"case:{case_id}:objective:{name}"] = score
        if not coordinates:
            return None
    return coordinates or None


async def _comparison_cap(
    task: OptimizationTask,
    candidates: int,
    repetitions: int,
    requested: int | None,
    *,
    include_seed: bool = False,
) -> int:
    """Resolve the named comparison budget from the fixed validation set."""
    if requested is not None:
        return requested
    return len(await task._validation_cases()) * (
        candidates * repetitions + int(include_seed)
    )


async def _require_comparison_budget(
    task: OptimizationTask,
    *,
    candidates: int,
    repetitions: int,
    requested: int | None,
    include_seed: bool = False,
) -> int:
    """Resolve and preflight a full matched comparison before work begins."""
    required = await _comparison_cap(
        task, candidates, repetitions, None, include_seed=include_seed
    )
    if requested is not None and requested < required:
        raise ValueError(
            f"comparison_metric_calls cannot fund {required} required matched metric calls; got {requested}."
        )
    return requested if requested is not None else required


def _pipeline_result(
    results: list[EngineResult],
    index: int,
    votes: list[FairVote],
    budget: BudgetTracker,
    comparison_budget: BudgetTracker,
    decision: dict[str, Any],
) -> PipelineResult:
    return PipelineResult(
        results=results,
        best=results[index],
        best_index=index,
        fair_scores=[vote.mean_score for vote in votes],
        total_metric_calls=budget.spent,
        comparison_metric_calls=comparison_budget.spent,
        fair_votes=votes,
        decision=decision,
    )


__all__ = [
    "FairVote",
    "OmniPlan",
    "PipelineResult",
    "SelectionRule",
    "optimize_adaptive_sequential",
    "optimize_best_of",
    "optimize_omni",
    "optimize_parallel",
    "optimize_sequential",
    "optimize_vote",
]

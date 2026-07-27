"""Helpers for composing optimization engines under one shared budget."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from .engines import (
    BudgetTracker,
    EngineConfig,
    EngineEvent,
    EngineResult,
    OptimizationTask,
    get_engine,
)
from .gepa_graph.models import CandidateMap


class PipelineResult(BaseModel):
    """Results of a composed engine run and its fairly selected winner."""

    results: list[EngineResult]
    best: EngineResult
    best_index: int
    fair_scores: list[float]
    total_metric_calls: int

    model_config = ConfigDict(frozen=True)


class _SeededTask:
    """Delegate a task while supplying a fixed seed candidate to an engine."""

    def __init__(self, task: OptimizationTask, seed: CandidateMap) -> None:
        self._task = task
        self._seed = seed

    async def seed_candidate(self) -> CandidateMap:
        """Return the candidate adopted by the preceding pipeline stage."""
        return self._seed

    def __getattr__(self, name: str) -> Any:
        """Forward all non-seed task operations to the wrapped task."""
        return getattr(self._task, name)


async def optimize_parallel(
    task: OptimizationTask,
    configs: Sequence[EngineConfig],
    *,
    max_metric_calls: int,
) -> list[EngineResult]:
    """Run configured engines concurrently against one shared metric budget.

    Failures propagate from :func:`asyncio.gather`: a failed engine makes the
    pipeline fail loudly instead of silently returning a partial comparison.
    """
    budget = BudgetTracker(max_metric_calls)
    return await _run_parallel(task, configs, budget)


async def optimize_best_of(
    task: OptimizationTask,
    configs: Sequence[EngineConfig],
    *,
    max_metric_calls: int,
) -> PipelineResult:
    """Run engines in parallel and choose the highest fair valset score.

    Fair evaluations are intentionally uncharged, matching coding-agent final
    evaluation semantics and ensuring engines are compared on the same valset.
    """
    budget = BudgetTracker(max_metric_calls)
    results = await _run_parallel(task, configs, budget)
    best_index, fair_scores = await _select_fair_winner(task, results)
    return _pipeline_result(results, best_index, fair_scores, budget)


async def optimize_sequential(
    task: OptimizationTask,
    configs: Sequence[EngineConfig],
    *,
    max_metric_calls: int,
) -> PipelineResult:
    """Run engines in order, seeding each from the last non-regressing result.

    Each stage and its incoming seed are evaluated on the valset without
    charging the optimization budget.  A lower-scoring stage is not adopted as
    the next seed, which keeps the chain monotonic under the shared evaluator.
    """
    if not configs:
        raise ValueError("At least one engine config is required.")

    budget = BudgetTracker(max_metric_calls)
    seed = await task.seed_candidate()
    seed_score = (await task.evaluate(seed, budget=None)).score
    adopted_result = EngineResult(
        engine="seed",
        best_candidate=seed,
        best_score=seed_score,
        num_metric_calls=0,
        history=[EngineEvent(kind="seed", data={"fair_score": seed_score})],
    )
    adopted_index = -1
    results: list[EngineResult] = []
    fair_scores: list[float] = []

    for index, config in enumerate(configs):
        stage_task: OptimizationTask
        if index == 0:
            stage_task = task
        else:
            stage_task = cast(OptimizationTask, _SeededTask(task, seed))
        result = await get_engine(config.engine, config).run(stage_task, config, budget)
        results.append(result)
        fair_score = (await task.evaluate(result.best_candidate, budget=None)).score
        fair_scores.append(fair_score)
        if fair_score >= seed_score:
            seed = result.best_candidate
            seed_score = fair_score
            adopted_result = result
            adopted_index = index

    return PipelineResult(
        results=results,
        best=adopted_result,
        best_index=adopted_index,
        fair_scores=fair_scores,
        total_metric_calls=budget.spent,
    )


async def optimize_vote(
    task: OptimizationTask,
    configs: Sequence[EngineConfig],
    *,
    max_metric_calls: int,
) -> PipelineResult:
    """Run engines in parallel and select a winner by an uncharged valset vote.

    This shares the fair-selection implementation with ``optimize_best_of``;
    the APIs differ in intent, with this helper emphasizing the re-evaluation
    vote as the selection mechanism.
    """
    budget = BudgetTracker(max_metric_calls)
    results = await _run_parallel(task, configs, budget)
    best_index, fair_scores = await _select_fair_winner(task, results)
    return _pipeline_result(results, best_index, fair_scores, budget)


async def _run_parallel(
    task: OptimizationTask,
    configs: Sequence[EngineConfig],
    budget: BudgetTracker,
) -> list[EngineResult]:
    """Construct and concurrently run engines in configuration order."""
    engines = [get_engine(config.engine, config) for config in configs]
    return list(
        await asyncio.gather(
            *(
                engine.run(task, config, budget)
                for engine, config in zip(engines, configs)
            )
        )
    )


async def _select_fair_winner(
    task: OptimizationTask, results: Sequence[EngineResult]
) -> tuple[int, list[float]]:
    """Return the earliest result with the highest uncharged full-valset score."""
    if not results:
        raise ValueError("At least one engine result is required.")
    fair_scores = [
        (await task.evaluate(result.best_candidate, budget=None)).score
        for result in results
    ]
    return max(
        range(len(fair_scores)), key=lambda index: fair_scores[index]
    ), fair_scores


def _pipeline_result(
    results: list[EngineResult],
    best_index: int,
    fair_scores: list[float],
    budget: BudgetTracker,
) -> PipelineResult:
    """Build a public result without exposing the mutable budget tracker."""
    return PipelineResult(
        results=results,
        best=results[best_index],
        best_index=best_index,
        fair_scores=fair_scores,
        total_metric_calls=budget.spent,
    )


__all__ = [
    "PipelineResult",
    "optimize_best_of",
    "optimize_parallel",
    "optimize_sequential",
    "optimize_vote",
]

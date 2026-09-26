"""A small reference engine that selects the best of sampled candidates."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

from ..gepa_graph.models import CandidateMap
from .base import (
    BudgetExhausted,
    BudgetTracker,
    EngineConfig,
    EngineEvent,
    EngineResult,
    OptimizationTask,
    check_cost_budget_support,
)
from .registry import register_engine

Proposer = Callable[[CandidateMap], Awaitable[CandidateMap]]


class BestOfNEngine:
    """Sample variants of the seed and keep the highest-scoring candidate.

    ``engine_config['propose']`` is a required async callback accepting the
    seed :class:`CandidateMap` and returning one derived candidate.  The seed
    itself is always evaluated as candidate zero, so a completed run cannot
    select a candidate worse than its evaluated seed.
    """

    name = "best_of_n"

    def __init__(self, config: EngineConfig) -> None:
        """Read the required proposal callback and the variant count."""
        propose = config.engine_config.get("propose")
        if not callable(propose):
            raise TypeError(
                "engine_config['propose'] must be a callable async callback that "
                "accepts a CandidateMap seed and returns a CandidateMap."
            )
        n = config.engine_config.get("n", 4)
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("engine_config['n'] must be a positive integer.")

        self._propose = cast(Proposer, propose)
        self._n = n

    async def run(
        self,
        task: OptimizationTask,
        config: EngineConfig,
        budget: BudgetTracker,
    ) -> EngineResult:
        """Evaluate the seed and ``n`` derived candidates under ``budget``."""
        check_cost_budget_support(self, config)
        budget.check()
        starting_spend = budget.spent
        seed = await task.seed_candidate()
        candidates = [seed]
        for _ in range(self._n):
            # The proposer gets its own copy: mutating it in place must not
            # change candidate zero, whose score may be reused below.
            candidates.append(await self._propose(_copy_candidate(seed)))

        # A composition helper may already have scored the seed; reuse that
        # aggregate instead of spending this slice's budget on it again.
        seed_scorer = getattr(task, "seed_validation_score", None)
        pre_scored_seed = await seed_scorer() if seed_scorer is not None else None

        best_candidate = seed
        best_score: float | None = None
        scores: list[float | None] = []
        history: list[EngineEvent] = []
        for index, candidate in enumerate(candidates):
            if index == 0 and pre_scored_seed is not None:
                scores.append(pre_scored_seed.score)
                if best_score is None or pre_scored_seed.score > best_score:
                    best_candidate = candidate
                    best_score = pre_scored_seed.score
                continue
            try:
                evaluation = await task.evaluate(candidate, budget=budget)
            except BudgetExhausted:
                scores.append(None)
                history.append(
                    EngineEvent(
                        kind="budget_overshoot",
                        message="Candidate evaluation exceeded the shared metric-call budget.",
                        data={
                            "candidate_index": index,
                            "budget_remaining": budget.remaining,
                        },
                    )
                )
                break

            scores.append(evaluation.score)
            if best_score is None or evaluation.score > best_score:
                best_candidate = candidate
                best_score = evaluation.score

        history.append(
            EngineEvent(
                kind="summary",
                data={
                    "num_variants": self._n,
                    "candidate_scores": scores
                    + [None] * (len(candidates) - len(scores)),
                    "evaluated_candidates": len(scores) - scores.count(None),
                },
            )
        )
        return EngineResult(
            engine=self.name,
            best_candidate=best_candidate,
            best_score=best_score,
            num_metric_calls=budget.spent - starting_spend,
            history=history,
        )


def _copy_candidate(candidate: CandidateMap) -> CandidateMap:
    """Isolate the evaluated seed from proposer-side mutation."""
    return {
        name: component.model_copy(deep=True) for name, component in candidate.items()
    }


register_engine(BestOfNEngine.name, BestOfNEngine, replace=True)


__all__ = ["BestOfNEngine", "Proposer"]

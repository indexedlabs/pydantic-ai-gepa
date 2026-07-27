"""Coverage for public optimization-engine composition helpers."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa.compose import (
    optimize_best_of,
    optimize_parallel,
    optimize_sequential,
    optimize_vote,
)
from pydantic_ai_gepa.engines import (
    BudgetTracker,
    EngineConfig,
    EngineResult,
    OptimizationTask,
    register_engine,
    unregister_engine,
)
from pydantic_ai_gepa.gepa_graph.models import CandidateMap, ComponentValue
from pydantic_ai_gepa.types import MetricResult, RolloutOutput

_ENGINE_NAME = "compose_test_engine"


def _candidate(text: str) -> CandidateMap:
    return {"instructions": ComponentValue(name="instructions", text=text)}


def _task() -> OptimizationTask:
    agent = Agent(TestModel(custom_output_text="response"), instructions="seed")
    cases = [Case(name="case", inputs="input", expected_output="response")]

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        override = agent._override_instructions.get()
        active = override.value if override is not None else agent._instructions
        return MetricResult(score=float("correct" in str(active)))

    return OptimizationTask(agent=agent, trainset=cases, metric=metric, valset=cases)


class _FakeEngine:
    name = _ENGINE_NAME

    def __init__(self, config: EngineConfig) -> None:
        self._config = config

    async def run(
        self,
        task: OptimizationTask,
        config: EngineConfig,
        budget: BudgetTracker,
    ) -> EngineResult:
        options = config.engine_config
        seen = options.get("seen_seeds")
        if isinstance(seen, list):
            seen.append((await task.seed_candidate())["instructions"].text)
        spend = options.get("spend", 0)
        budget.spend(spend)
        return EngineResult(
            engine=self.name,
            best_candidate=options["candidate"],
            best_score=options.get("reported_score", 0.0),
            num_metric_calls=spend,
        )


@pytest.fixture(autouse=True)
def _registered_fake_engine() -> Generator[None, None, None]:
    unregister_engine(_ENGINE_NAME)
    register_engine(_ENGINE_NAME, _FakeEngine, replace=True)
    yield
    unregister_engine(_ENGINE_NAME)


def _config(
    candidate: CandidateMap,
    *,
    reported_score: float = 0.0,
    spend: int = 0,
    seen_seeds: list[str] | None = None,
) -> EngineConfig:
    options: dict[str, Any] = {
        "candidate": candidate,
        "reported_score": reported_score,
        "spend": spend,
    }
    if seen_seeds is not None:
        options["seen_seeds"] = seen_seeds
    return EngineConfig(engine=_ENGINE_NAME, engine_config=options)


@pytest.mark.asyncio
async def test_optimize_parallel_preserves_order_and_shares_its_budget() -> None:
    first = _config(_candidate("first"), spend=1)
    second = _config(_candidate("second"), spend=2)

    results = await optimize_parallel(_task(), [first, second], max_metric_calls=3)

    assert [result.best_candidate["instructions"].text for result in results] == [
        "first",
        "second",
    ]
    assert sum(result.num_metric_calls for result in results) == 3


@pytest.mark.asyncio
async def test_optimize_best_of_uses_fair_valset_scores_not_reported_scores() -> None:
    low_actual = _config(_candidate("wrong"), reported_score=99.0)
    high_actual = _config(_candidate("correct"), reported_score=-1.0)

    result = await optimize_best_of(
        _task(), [low_actual, high_actual], max_metric_calls=2
    )

    assert result.best_index == 1
    assert result.best.best_candidate == result.results[1].best_candidate
    assert result.fair_scores == [0.0, 1.0]
    assert [item.best_score for item in result.results] == [99.0, -1.0]
    assert result.total_metric_calls == 0


@pytest.mark.asyncio
async def test_optimize_sequential_chains_seeds_and_rejects_regressions() -> None:
    seen_seeds: list[str] = []
    configs = [
        _config(_candidate("correct first"), seen_seeds=seen_seeds),
        _config(_candidate("wrong second"), seen_seeds=seen_seeds),
        _config(_candidate("correct third"), seen_seeds=seen_seeds),
    ]

    result = await optimize_sequential(_task(), configs, max_metric_calls=2)

    assert seen_seeds == ["seed", "correct first", "correct first"]
    assert result.fair_scores == [1.0, 0.0, 1.0]
    assert result.best_index == 2
    assert result.best.best_candidate == result.results[2].best_candidate


@pytest.mark.asyncio
async def test_optimize_vote_selects_the_highest_valset_score() -> None:
    result = await optimize_vote(
        _task(),
        [_config(_candidate("wrong")), _config(_candidate("correct"))],
        max_metric_calls=2,
    )

    assert result.best_index == 1
    assert result.best.best_candidate["instructions"].text == "correct"
    assert result.fair_scores == [0.0, 1.0]

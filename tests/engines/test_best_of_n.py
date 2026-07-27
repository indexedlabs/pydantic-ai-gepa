"""Coverage for the reference best-of-N optimization engine."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa.engines import (
    BestOfNEngine,
    BudgetTracker,
    EngineConfig,
    OptimizationTask,
    get_engine,
)
from pydantic_ai_gepa.gepa_graph.models import CandidateMap, ComponentValue
from pydantic_ai_gepa.types import MetricResult, RolloutOutput


def _candidate(text: str) -> CandidateMap:
    return {"instructions": ComponentValue(name="instructions", text=text)}


def _task(*, case_count: int = 1, instructions: str = "seed") -> OptimizationTask:
    agent = Agent(TestModel(custom_output_text="response"), instructions=instructions)
    cases = [
        Case(name=f"case-{index}", inputs="input", expected_output="response")
        for index in range(case_count)
    ]

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        override = agent._override_instructions.get()
        active = override.value if override is not None else agent._instructions
        return MetricResult(score=float("correct" in str(active)))

    return OptimizationTask(agent=agent, trainset=cases, metric=metric, valset=cases)


def test_registry_constructs_best_of_n_engine() -> None:
    async def propose(seed: CandidateMap) -> CandidateMap:
        return seed

    config = EngineConfig(engine="best_of_n", engine_config={"propose": propose})

    assert isinstance(get_engine("best_of_n", config), BestOfNEngine)


@pytest.mark.asyncio
async def test_best_of_n_selects_the_highest_scoring_variant() -> None:
    variants = iter([_candidate("still wrong"), _candidate("correct")])

    async def propose(seed: CandidateMap) -> CandidateMap:
        return next(variants)

    config = EngineConfig(
        engine="best_of_n",
        engine_config={"n": 2, "propose": propose},
    )
    budget = BudgetTracker(3)

    result = await get_engine("best_of_n", config).run(_task(), config, budget)

    assert result.best_candidate["instructions"].text == "correct"
    assert result.best_score == 1.0
    assert result.num_metric_calls == budget.spent == 3
    assert result.history[-1].data["candidate_scores"] == [0.0, 0.0, 1.0]


@pytest.mark.asyncio
async def test_best_of_n_never_selects_a_candidate_worse_than_the_seed() -> None:
    async def propose(seed: CandidateMap) -> CandidateMap:
        return _candidate("wrong")

    config = EngineConfig(
        engine="best_of_n",
        engine_config={"n": 2, "propose": propose},
    )

    result = await get_engine("best_of_n", config).run(
        _task(instructions="correct seed"), config, BudgetTracker(3)
    )

    assert result.best_candidate["instructions"].text == "correct seed"
    assert result.best_score == 1.0


@pytest.mark.asyncio
async def test_best_of_n_stops_evaluating_on_budget_overshoot() -> None:
    async def propose(seed: CandidateMap) -> CandidateMap:
        return _candidate("correct")

    config = EngineConfig(
        engine="best_of_n",
        engine_config={"n": 2, "propose": propose},
    )
    budget = BudgetTracker(3)

    result = await get_engine("best_of_n", config).run(
        _task(case_count=2), config, budget
    )

    assert result.best_candidate["instructions"].text == "seed"
    assert result.num_metric_calls == budget.spent == 2
    assert [event.kind for event in result.history].count("budget_overshoot") == 1
    assert result.history[-1].data["candidate_scores"] == [0.0, None, None]


@pytest.mark.parametrize("propose", [None, "not callable", object()])
def test_best_of_n_requires_a_callable_proposer(propose: object) -> None:
    config = EngineConfig(engine="best_of_n", engine_config={"propose": propose})

    with pytest.raises(TypeError, match=r"engine_config\['propose'\].*callable"):
        BestOfNEngine(config)

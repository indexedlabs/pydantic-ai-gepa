"""Coverage for the caller-proposed coding-agent optimization engine."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa.engines import (
    BudgetTracker,
    CodingAgentEngine,
    EngineConfig,
    OptimizationTask,
    ReflectionContext,
    get_engine,
)
from pydantic_ai_gepa.gepa_graph.models import CandidateMap, ComponentValue
from pydantic_ai_gepa.types import MetricResult, RolloutOutput


def _task(*, case_count: int = 3) -> OptimizationTask:
    """Build a TestModel task whose metric observes the applied instructions."""
    agent = Agent(TestModel(custom_output_text="response"), instructions="seed")
    cases = [
        Case(name=f"case-{index}", inputs="input", expected_output="response")
        for index in range(case_count)
    ]

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        override = agent._override_instructions.get()
        instructions = override.value if override is not None else agent._instructions
        correct = "correct" in "\n".join(str(item) for item in instructions)
        return MetricResult(
            score=float(correct),
            feedback="correct" if correct else f"Add correct guidance for {case.name}.",
        )

    return OptimizationTask(agent=agent, trainset=cases, metric=metric, valset=cases)


def _candidate(text: str) -> CandidateMap:
    return {"instructions": ComponentValue(name="instructions", text=text)}


def test_registry_constructs_coding_agent_engine() -> None:
    """Importing the engines package registers the built-in coding-agent engine."""

    async def propose(context: ReflectionContext) -> CandidateMap:
        return context.candidate

    config = EngineConfig(engine="coding_agent", engine_config={"propose": propose})

    assert isinstance(get_engine("coding_agent", config), CodingAgentEngine)


@pytest.mark.asyncio
async def test_coding_agent_engine_accepts_an_improving_proposal() -> None:
    """A strictly better proposal is retained and receives the fair valset score."""
    contexts: list[ReflectionContext] = []

    async def propose(context: ReflectionContext) -> CandidateMap:
        contexts.append(context)
        return _candidate("correct")

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=9,
        engine_config={
            "propose": propose,
            "minibatch_size": 3,
            "max_proposals_per_run": 1,
        },
    )
    budget = BudgetTracker(9)

    result = await get_engine("coding_agent", config).run(_task(), config, budget)

    assert result.best_candidate["instructions"].text == "correct"
    assert result.best_score == 1.0
    assert budget.spent == result.num_metric_calls == 9
    assert [event.kind for event in result.history].count("accepted") == 1
    assert contexts[0].minibatch_records
    assert contexts[0].report.startswith("# Eval report")


@pytest.mark.asyncio
async def test_coding_agent_engine_rejects_equal_or_worse_proposals() -> None:
    """Only a strict minibatch improvement can replace the current best candidate."""

    async def propose(context: ReflectionContext) -> CandidateMap:
        return _candidate("still wrong")

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=9,
        engine_config={
            "propose": propose,
            "minibatch_size": 3,
            "max_proposals_per_run": 1,
        },
    )

    result = await get_engine("coding_agent", config).run(
        _task(), config, BudgetTracker(9)
    )

    assert result.best_candidate["instructions"].text == "seed"
    assert result.best_score == 0.0
    assert any(event.kind == "rejected" for event in result.history)


@pytest.mark.asyncio
async def test_coding_agent_engine_stops_after_a_budget_overshoot() -> None:
    """An oversized proposal evaluation keeps the already-confirmed seed candidate."""

    async def propose(context: ReflectionContext) -> CandidateMap:
        return _candidate("correct")

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=4,
        engine_config={
            "propose": propose,
            "minibatch_size": 3,
            "max_proposals_per_run": 1,
        },
    )
    budget = BudgetTracker(4)

    result = await get_engine("coding_agent", config).run(_task(), config, budget)

    assert result.best_candidate["instructions"].text == "seed"
    assert result.num_metric_calls == budget.spent == 3
    assert any(event.kind == "budget_overshoot" for event in result.history)


@pytest.mark.parametrize("propose", [None, "not callable", object()])
def test_coding_agent_engine_requires_a_callable_proposer(propose: object) -> None:
    """The proposal callback is a required engine-specific dependency."""
    config = EngineConfig(engine="coding_agent", engine_config={"propose": propose})

    with pytest.raises(TypeError, match=r"engine_config\['propose'\].*callable"):
        CodingAgentEngine(config)


@pytest.mark.asyncio
async def test_coding_agent_engine_minibatches_are_deterministic_for_a_seed() -> None:
    """Equal seeds produce the same sampled minibatch history and metric usage."""

    async def propose(context: ReflectionContext) -> CandidateMap:
        return _candidate("still wrong")

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=11,
        seed=17,
        engine_config={
            "propose": propose,
            "minibatch_size": 3,
            "max_proposals_per_run": 1,
        },
    )
    first = await get_engine("coding_agent", config).run(
        _task(case_count=5), config, BudgetTracker(11)
    )
    second = await get_engine("coding_agent", config).run(
        _task(case_count=5), config, BudgetTracker(11)
    )

    assert first.num_metric_calls == second.num_metric_calls == 11
    assert [event.model_dump() for event in first.history] == [
        event.model_dump() for event in second.history
    ]

"""Coverage for the caller-proposed coding-agent optimization engine."""

from __future__ import annotations

from collections.abc import Awaitable
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
async def test_coding_agent_engine_classifies_equal_proposal_as_equivalent() -> None:
    """An unchanged deterministic proposal is not treated as a failed hypothesis."""

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
    assert any(event.kind == "equivalent" for event in result.history)


@pytest.mark.asyncio
async def test_coding_agent_engine_repeats_matched_case_evaluations() -> None:
    """Configured repetitions are charged and exposed in comparison history."""

    async def propose(context: ReflectionContext) -> CandidateMap:
        return _candidate("correct")

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=7,
        engine_config={
            "propose": propose,
            "minibatch_size": 1,
            "max_proposals_per_run": 1,
            "acceptance_repetitions": 3,
            "acceptance_max_repetitions": 3,
        },
    )
    budget = BudgetTracker(7)

    result = await get_engine("coding_agent", config).run(
        _task(case_count=1), config, budget
    )

    accepted = next(event for event in result.history if event.kind == "accepted")
    assert accepted.data["baseline_sample_count"] == 3
    assert accepted.data["candidate_sample_count"] == 3
    assert accepted.data["verdict"] == "accepted"
    assert result.num_metric_calls == budget.spent == 7


@pytest.mark.asyncio
async def test_coding_agent_engine_preserves_noisy_overlap_as_inconclusive() -> None:
    """A positive sample mean inside rollout variance is not accepted or rejected."""
    agent = Agent(TestModel(custom_output_text="response"), instructions="seed")
    case = Case(name="case-noisy", inputs="input", expected_output="response")
    samples = {
        "seed": iter([0.40, 0.60, 0.50, 0.50]),
        "proposal": iter([0.45, 0.65, 0.55]),
    }

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        del case, output
        override = agent._override_instructions.get()
        instructions = override.value if override is not None else agent._instructions
        key = (
            "proposal"
            if "proposal" in "\n".join(str(item) for item in instructions)
            else "seed"
        )
        return MetricResult(score=next(samples[key]), feedback="noisy")

    task = OptimizationTask(agent=agent, trainset=[case], metric=metric, valset=[case])

    async def propose(context: ReflectionContext) -> CandidateMap:
        return _candidate("proposal")

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=7,
        engine_config={
            "propose": propose,
            "minibatch_size": 1,
            "max_proposals_per_run": 1,
            "acceptance_repetitions": 3,
            "acceptance_max_repetitions": 3,
        },
    )

    result = await get_engine("coding_agent", config).run(
        task, config, BudgetTracker(7)
    )

    inconclusive = next(
        event for event in result.history if event.kind == "inconclusive"
    )
    assert inconclusive.data["delta"] == pytest.approx(0.05)
    assert inconclusive.data["lower_bound"] < 0.0
    assert inconclusive.data["upper_bound"] > 0.0
    assert result.best_candidate["instructions"].text == "seed"


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


@pytest.mark.asyncio
async def test_coding_agent_engine_stops_cleanly_when_first_minibatch_is_unaffordable() -> (
    None
):
    """A preflight failure must not leak a provider call or raise from run()."""

    metric_calls = 0
    proposer_calls = 0
    task = _task(case_count=2)
    original_metric = task.metric

    def counted_metric(
        case: Case[str, str, Any], output: RolloutOutput[Any]
    ) -> MetricResult | Awaitable[MetricResult]:
        nonlocal metric_calls
        metric_calls += 1
        return original_metric(case, output)

    task.metric = counted_metric

    async def propose(context: ReflectionContext) -> CandidateMap:
        del context
        nonlocal proposer_calls
        proposer_calls += 1
        return _candidate("correct")

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=1,
        engine_config={"propose": propose, "minibatch_size": 2},
    )
    budget = BudgetTracker(1)

    result = await get_engine("coding_agent", config).run(task, config, budget)

    assert result.best_candidate["instructions"].text == "seed"
    assert result.num_metric_calls == budget.spent == 0
    assert metric_calls == proposer_calls == 0
    assert any(event.kind == "budget_exhausted" for event in result.history)


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

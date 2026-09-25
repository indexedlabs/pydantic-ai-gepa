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
from pydantic_ai_gepa.engines.base import CandidateEvaluation
from pydantic_ai_gepa.engines.coding_agent_engine import (
    _CandidatePoolEntry,
    _select_pareto_parent,
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


@pytest.mark.asyncio
async def test_coding_agent_engine_returns_unscored_seed_when_budget_is_insufficient():
    async def propose(context: ReflectionContext) -> CandidateMap:
        pytest.fail("An unscored seed should not reach proposal generation")

    task = _task(case_count=3)
    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=2,
        engine_config={"propose": propose},
    )
    budget = BudgetTracker(2)

    result = await CodingAgentEngine(config).run(task, config, budget)

    assert result.best_candidate == await task.seed_candidate()
    assert result.best_score is None
    assert result.num_metric_calls == budget.spent == 0


def test_pareto_parent_selection_handles_partial_coordinate_overlap() -> None:
    pool = [
        _CandidatePoolEntry(
            _candidate("first"),
            CandidateEvaluation(
                score=0.5,
                records=[],
                side_info={},
                num_cases=2,
                per_case_scores={"shared": 1.0, "first-only": 0.0},
            ),
            0,
        ),
        _CandidatePoolEntry(
            _candidate("second"),
            CandidateEvaluation(
                score=0.5,
                records=[],
                side_info={},
                num_cases=2,
                per_case_scores={"shared": 0.0, "second-only": 1.0},
            ),
            1,
        ),
    ]

    assert _select_pareto_parent(pool, seed=0, epoch=0) in pool


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

    # Seed validation + selection batch + 3 baseline + 3 proposal repetitions
    # of 3 cases + proposal validation: 3 + 3 + 9 + 9 + 3.
    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=27,
        engine_config={
            "propose": propose,
            "minibatch_size": 3,
            "max_proposals_per_run": 1,
        },
    )
    budget = BudgetTracker(27)

    result = await get_engine("coding_agent", config).run(_task(), config, budget)

    assert result.best_candidate["instructions"].text == "correct"
    assert result.best_score == 1.0
    assert budget.spent == result.num_metric_calls == 27
    assert [event.kind for event in result.history].count("accepted") == 1
    assert contexts[0].minibatch_records
    assert contexts[0].report.startswith("# Eval report")


@pytest.mark.asyncio
async def test_coding_agent_engine_keeps_validation_evidence_out_of_reflection() -> (
    None
):
    """Validation feedback and cases influence selection without entering context."""

    agent = Agent(TestModel(custom_output_text="response"), instructions="seed")
    train = Case(name="train-visible", inputs="train", expected_output="response")
    validation = Case(
        name="validation-secret", inputs="validation", expected_output="response"
    )

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        del output
        override = agent._override_instructions.get()
        instructions = override.value if override is not None else agent._instructions
        correct = "correct" in "\n".join(str(item) for item in instructions)
        return MetricResult(
            score=float(correct),
            feedback=(
                "VALIDATION SECRET" if case.name == "validation-secret" else "train fix"
            ),
            side_info={
                "evidence": (
                    "VALIDATION SIDE INFO"
                    if case.name == "validation-secret"
                    else "training side info"
                )
            },
        )

    task = OptimizationTask(
        agent=agent, trainset=[train], valset=[validation], metric=metric
    )
    contexts: list[ReflectionContext] = []

    async def propose(context: ReflectionContext) -> CandidateMap:
        contexts.append(context)
        return _candidate("correct")

    # Seed validation + selection batch + 3 matched repetitions + proposal
    # validation, one case each: 1 + 1 + 3 + 3 + 1.
    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=9,
        engine_config={
            "propose": propose,
            "minibatch_size": 1,
            "max_proposals_per_run": 1,
        },
    )
    result = await get_engine("coding_agent", config).run(
        task, config, BudgetTracker(9)
    )

    assert result.best_candidate["instructions"].text == "correct"
    assert [record.case_id for record in contexts[0].minibatch_records] == [
        "train-visible"
    ]
    for secret in ("validation-secret", "VALIDATION SECRET", "VALIDATION SIDE INFO"):
        assert secret not in repr(contexts)
        assert secret not in repr(result.history)
    assert "train fix" in contexts[0].report
    assert "training side info" in repr(contexts[0].minibatch_records)
    validation_events = [
        event for event in result.history if "validation_score" in event.data
    ]
    assert [event.data["validation_score"] for event in validation_events] == [0.0, 1.0]
    assert all(
        "validation_case_scores" not in event.data for event in validation_events
    )
    assert result.history[-1].data["validation_evaluations"] == 2


@pytest.mark.asyncio
async def test_coding_agent_engine_does_not_adopt_validation_regression() -> None:
    """A training improvement cannot replace a stronger validation incumbent."""

    agent = Agent(TestModel(custom_output_text="response"), instructions="seed")
    train = Case(name="train", inputs="train", expected_output="response")
    validation = Case(
        name="validation", inputs="validation", expected_output="response"
    )

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        del output
        override = agent._override_instructions.get()
        instructions = "\n".join(
            str(item)
            for item in (
                override.value if override is not None else agent._instructions
            )
        )
        proposal = "proposal" in instructions
        score = proposal if case.name == "train" else not proposal
        return MetricResult(score=float(score), feedback=f"feedback for {case.name}")

    task = OptimizationTask(
        agent=agent, trainset=[train], valset=[validation], metric=metric
    )

    async def propose(context: ReflectionContext) -> CandidateMap:
        return _candidate("proposal")

    # Seed validation + selection batch + 3 matched repetitions + proposal
    # validation, one case each: 1 + 1 + 3 + 3 + 1.
    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=9,
        engine_config={
            "propose": propose,
            "minibatch_size": 1,
            "max_proposals_per_run": 1,
        },
    )
    result = await get_engine("coding_agent", config).run(
        task, config, BudgetTracker(9)
    )

    assert result.best_candidate["instructions"].text == "seed"
    assert result.best_score == 1.0
    assert any(event.kind == "validation_rejected" for event in result.history)


@pytest.mark.asyncio
async def test_coding_agent_engine_skips_validation_for_non_improvement() -> None:
    """Only a training-improved proposal consumes another validation pass."""

    async def propose(context: ReflectionContext) -> CandidateMap:
        return _candidate("still wrong")

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=21,
        engine_config={
            "propose": propose,
            "minibatch_size": 3,
            "max_proposals_per_run": 1,
        },
    )
    result = await get_engine("coding_agent", config).run(
        _task(), config, BudgetTracker(21)
    )

    assert result.history[-1].data["validation_evaluations"] == 1
    # Seed validation + selection batch + 2 affordable matched repetitions:
    # 3 + 3 + 6 + 6; the budget no longer reaches a third repetition.
    assert result.num_metric_calls == 18


@pytest.mark.asyncio
async def test_coding_agent_engine_classifies_equal_proposal_as_equivalent() -> None:
    """An unchanged deterministic proposal is not treated as a failed hypothesis."""

    async def propose(context: ReflectionContext) -> CandidateMap:
        return _candidate("still wrong")

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=21,
        engine_config={
            "propose": propose,
            "minibatch_size": 3,
            "max_proposals_per_run": 1,
        },
    )

    result = await get_engine("coding_agent", config).run(
        _task(), config, BudgetTracker(21)
    )

    assert result.best_candidate["instructions"].text == "seed"
    assert result.best_score == 0.0
    assert any(event.kind == "equivalent" for event in result.history)


@pytest.mark.asyncio
async def test_coding_agent_engine_repeats_matched_case_evaluations() -> None:
    """Configured repetitions are charged and exposed in comparison history."""

    async def propose(context: ReflectionContext) -> CandidateMap:
        return _candidate("correct")

    # Seed validation + selection batch + 3 matched repetitions + proposal
    # validation, one case each: 1 + 1 + 3 + 3 + 1.
    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=9,
        engine_config={
            "propose": propose,
            "minibatch_size": 1,
            "max_proposals_per_run": 1,
            "acceptance_repetitions": 3,
            "acceptance_max_repetitions": 3,
        },
    )
    budget = BudgetTracker(9)

    result = await get_engine("coding_agent", config).run(
        _task(case_count=1), config, budget
    )

    accepted = next(event for event in result.history if event.kind == "accepted")
    assert accepted.data["baseline_sample_count"] == 3
    assert accepted.data["candidate_sample_count"] == 3
    assert accepted.data["verdict"] == "accepted"
    assert result.num_metric_calls == budget.spent == 9


@pytest.mark.asyncio
async def test_coding_agent_engine_preserves_noisy_overlap_as_inconclusive() -> None:
    """A positive sample mean inside rollout variance is not accepted or rejected."""
    agent = Agent(TestModel(custom_output_text="response"), instructions="seed")
    case = Case(name="case-noisy", inputs="input", expected_output="response")
    # Seed samples: validation, failure-selecting batch, three fresh baselines
    # with mean 0.50 so the delta stays 0.05.
    samples = {
        "seed": iter([0.50, 0.40, 0.60, 0.50, 0.40]),
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
        max_metric_calls=8,
        engine_config={
            "propose": propose,
            "minibatch_size": 1,
            "max_proposals_per_run": 1,
            "acceptance_repetitions": 3,
            "acceptance_max_repetitions": 3,
        },
    )

    result = await get_engine("coding_agent", config).run(
        task, config, BudgetTracker(8)
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
    assert any(event.kind == "budget_exhausted" for event in result.history)


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
async def test_coding_agent_engine_minibatches_are_deterministic_for_a_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal seeds produce the same sampled minibatch history and metric usage."""
    monkeypatch.setattr(
        "pydantic_ai_gepa.engines.coding_agent_engine.perf_counter", lambda: 0.0
    )

    async def propose(context: ReflectionContext) -> CandidateMap:
        return _candidate("still wrong")

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=23,
        seed=17,
        engine_config={
            "propose": propose,
            "minibatch_size": 3,
            "max_proposals_per_run": 1,
        },
    )
    first = await get_engine("coding_agent", config).run(
        _task(case_count=5), config, BudgetTracker(23)
    )
    second = await get_engine("coding_agent", config).run(
        _task(case_count=5), config, BudgetTracker(23)
    )

    # Seed validation (5) + selection batch (3) + 2 affordable matched
    # repetitions (6 + 6); the remaining 3 calls buy no third repetition.
    assert first.num_metric_calls == second.num_metric_calls == 20
    assert [event.model_dump() for event in first.history] == [
        event.model_dump() for event in second.history
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "engine_limit,shared_limit", [(5, 20), (20, 5), (8, 20), (20, 8)]
)
async def test_coding_agent_stops_before_under_sampled_comparison(
    engine_limit: int,
    shared_limit: int,
) -> None:
    """Fewer than two matched batches buys only validation and selection.

    The failure-selecting batch is charged before the engine checks that
    paired fresh repetitions are affordable, so the proposer stays uncalled.
    """
    task = _task(case_count=2)
    original_metric = task.metric
    metric_calls = 0
    proposer_calls = 0

    def counted_metric(
        case: Case[str, str, Any], output: RolloutOutput[Any]
    ) -> MetricResult | Awaitable[MetricResult]:
        nonlocal metric_calls
        metric_calls += 1
        return original_metric(case, output)

    task.metric = counted_metric

    async def propose(context: ReflectionContext) -> CandidateMap:
        nonlocal proposer_calls
        proposer_calls += 1
        return _candidate("correct")

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=engine_limit,
        engine_config={"propose": propose, "minibatch_size": 2},
    )
    budget = BudgetTracker(shared_limit)
    result = await get_engine("coding_agent", config).run(task, config, budget)

    assert result.best_candidate == _candidate("seed")
    assert result.num_metric_calls == budget.spent == metric_calls == 4
    assert proposer_calls == 0
    event = next(event for event in result.history if event.kind == "budget_exhausted")
    assert event.data["reason_code"] == "insufficient_acceptance_repetitions"
    assert event.data["affordable_repetitions"] < 2
    assert result.history[-1].data["stop_reason"] == "budget_exhausted"
    assert result.history[-1].data["iterations"] == 1


@pytest.mark.asyncio
async def test_coding_agent_can_accept_two_affordable_matched_repetitions() -> None:
    """Budget truncation may retain two samples, which can establish an effect."""

    async def propose(context: ReflectionContext) -> CandidateMap:
        return _candidate("correct")

    # Seed validation + selection batch + 2 matched repetitions + proposal
    # validation, two cases each: 2 + 2 + 4 + 4 + 2.
    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=14,
        engine_config={
            "propose": propose,
            "minibatch_size": 2,
            "max_proposals_per_run": 1,
        },
    )
    result = await get_engine("coding_agent", config).run(
        _task(case_count=2), config, BudgetTracker(14)
    )
    assert result.best_candidate == _candidate("correct")
    accepted = next(event for event in result.history if event.kind == "accepted")
    assert accepted.data["baseline_sample_count"] == 2
    assert accepted.data["candidate_sample_count"] == 2
    assert result.num_metric_calls == 14


@pytest.mark.asyncio
async def test_coding_agent_engine_excludes_the_failure_selected_batch() -> None:
    """The batch that triggers reflection is charged but never used as evidence."""
    agent = Agent(TestModel(custom_output_text="response"), instructions="seed")
    train = Case(name="train-case", inputs="input", expected_output="response")
    validation = Case(
        name="validation-case", inputs="input", expected_output="response"
    )
    # Seed draws on the training case, in call order: the low
    # failure-selecting batch, then the fresh baseline repetitions.
    # The proposal matches the fresh baseline mean.
    samples = {
        "seed": iter([0.0, 0.7, 0.7]),
        "proposal": iter([0.7, 0.7]),
    }

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        del output
        if case.name == "validation-case":
            return MetricResult(score=0.5, feedback="validation")
        override = agent._override_instructions.get()
        instructions = override.value if override is not None else agent._instructions
        key = (
            "proposal"
            if "proposal" in "\n".join(str(item) for item in instructions)
            else "seed"
        )
        return MetricResult(score=next(samples[key]), feedback=f"{key} feedback")

    task = OptimizationTask(
        agent=agent, trainset=[train], valset=[validation], metric=metric
    )
    contexts: list[ReflectionContext] = []

    async def propose(context: ReflectionContext) -> CandidateMap:
        contexts.append(context)
        return _candidate("proposal")

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=6,
        engine_config={
            "propose": propose,
            "minibatch_size": 1,
            "max_proposals_per_run": 1,
            "acceptance_repetitions": 2,
            "acceptance_max_repetitions": 2,
        },
    )
    result = await get_engine("coding_agent", config).run(
        task, config, BudgetTracker(6)
    )

    comparison = next(
        event for event in result.history if "selection_score" in event.data
    )
    assert comparison.data["selection_score"] == 0.0
    assert comparison.data["baseline_samples"] == [0.7, 0.7]
    assert comparison.data["baseline_score"] == pytest.approx(0.7)
    # The selected batch still drives the reflection context.
    assert [record.score for record in contexts[0].minibatch_records] == [0.0]
    assert "train-case" in contexts[0].report
    assert result.num_metric_calls == 6


@pytest.mark.asyncio
async def test_coding_agent_engine_reserves_paired_repetitions_before_proposing() -> (
    None
):
    """A failure selects the minibatch, but too few affordable paired
    repetitions stop the run before the proposer is ever called."""
    proposer_calls = 0

    async def propose(context: ReflectionContext) -> CandidateMap:
        del context
        nonlocal proposer_calls
        proposer_calls += 1
        return _candidate("correct")

    # Two-case validation (2) and selection batch (2) leave 4 calls: only one
    # paired baseline/proposal repetition of the two-case minibatch.
    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=8,
        engine_config={
            "propose": propose,
            "minibatch_size": 2,
            "max_proposals_per_run": 1,
        },
    )
    budget = BudgetTracker(8)

    result = await get_engine("coding_agent", config).run(
        _task(case_count=2), config, budget
    )

    assert proposer_calls == 0
    assert result.num_metric_calls == budget.spent == 4
    event = next(event for event in result.history if event.kind == "budget_exhausted")
    assert event.data["stage"] == "baseline_minibatch"
    assert event.data["reason_code"] == "insufficient_acceptance_repetitions"
    assert event.data["affordable_repetitions"] == 1
    assert result.history[-1].data["stop_reason"] == "budget_exhausted"
    assert result.history[-1].data["iterations"] == 1
    assert result.history[-1].data["proposals"] == 0

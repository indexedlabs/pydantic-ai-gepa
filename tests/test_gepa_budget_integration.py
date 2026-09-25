"""Real GEPA engine and composer accounting with deterministic local models."""

from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa.compose import optimize_adaptive_sequential, optimize_sequential
from pydantic_ai_gepa.engines import (
    BudgetTracker,
    EngineConfig,
    GepaEngine,
    OptimizationTask,
)
from pydantic_ai_gepa.gepa_graph.proposal import (
    InstructionProposalGenerator,
    ProposalResult,
)
from pydantic_ai_gepa.types import MetricResult, ReflectionConfig, RolloutOutput


@pytest.fixture
def counted_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []
    agent = Agent(TestModel(custom_output_text="ok"), instructions="seed")

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        calls.append(case.name)
        override = agent._override_instructions.get()
        active = override.value if override is not None else agent._instructions
        return MetricResult(score=0.85 if "improved" in str(active) else 0.4)

    async def propose(self, **kwargs):
        return ProposalResult(
            texts={"instructions": "improved"}, component_metadata={}, reasoning=None
        )

    monkeypatch.setattr(InstructionProposalGenerator, "propose_texts", propose)
    train = [Case(name=f"train-{i}", inputs="hello") for i in range(2)]
    validation = [Case(name=f"val-{i}", inputs="hello") for i in range(3)]
    return OptimizationTask(
        agent=agent, trainset=train, valset=validation, metric=metric
    ), calls


def _config(budget: int, *, target: float | None = None) -> EngineConfig:
    return EngineConfig(
        engine="gepa",
        max_metric_calls=budget,
        max_iterations=10,
        stop_at_score=target,
        engine_config={
            "reflection_minibatch_size": 2,
            "reflection_config": ReflectionConfig(model=TestModel()),
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("slice_size", "spent"), [(2, 0), (4, 3), (6, 5), (8, 7), (10, 10)]
)
async def test_engine_returns_and_refunds_unused_budget(
    counted_task, slice_size: int, spent: int
) -> None:
    task, calls = counted_task
    config = _config(100)
    budget = BudgetTracker(slice_size)

    result = await GepaEngine(config).run(task, config, budget)

    assert result.num_metric_calls == budget.spent == len(calls) == spent
    assert budget.remaining == slice_size - spent
    assert result.best_candidate["instructions"].text == (
        "improved" if spent == 10 else "seed"
    )
    if spent == 0:
        assert result.best_score is None
    else:
        assert result.best_score == pytest.approx(0.85 if spent == 10 else 0.4)
    summary = result.history[-1].data
    assert summary["total_evaluations"] == spent
    assert summary["stop_reason"].startswith("Max evaluations reached")
    if spent == 0:
        assert summary["original_score"] is None
        assert "cannot cover the validation set" in summary["stop_reason"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("target", "spent"), [(0.4, 3), (0.8, 10)])
async def test_engine_target_stops_and_refunds(
    counted_task, target: float, spent: int
) -> None:
    task, calls = counted_task
    config = _config(100, target=target)
    budget = BudgetTracker(100)

    result = await GepaEngine(config).run(task, config, budget)

    assert result.best_score is not None
    assert result.best_score >= target
    assert result.num_metric_calls == budget.spent == len(calls) == spent
    assert result.history[-1].data["stop_reason"] == "Target score reached"
    assert result.history[-1].data["iterations"] < config.max_iterations


@pytest.mark.asyncio
async def test_engine_preserves_measured_zero_score(counted_task) -> None:
    task, calls = counted_task

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        calls.append(case.name)
        return MetricResult(score=0.0)

    task.metric = metric
    config = _config(3)
    budget = BudgetTracker(3)

    result = await GepaEngine(config).run(task, config, budget)

    assert result.best_score == 0.0
    assert result.history[-1].data["original_score"] == 0.0
    assert result.num_metric_calls == budget.spent == len(calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize(("slice_size", "spent"), [(2, 0), (6, 5), (8, 7)])
async def test_composition_charges_only_spent_gepa_rollouts(
    counted_task,
    adaptive: bool,
    slice_size: int,
    spent: int,
) -> None:
    task, calls = counted_task
    configs = [_config(slice_size)]
    if adaptive:
        result = await optimize_adaptive_sequential(
            task,
            configs,
            max_metric_calls=20,
            max_slices=1,
        )
    else:
        result = await optimize_sequential(task, configs, max_metric_calls=20)

    assert len(result.results) == 1
    stage = result.results[0]
    assert stage.best_candidate["instructions"].text == "seed"
    assert stage.history[-1].data["stop_reason"].startswith("Max evaluations reached")
    assert result.total_metric_calls == stage.num_metric_calls == spent
    assert result.comparison_metric_calls == 6  # Full seed and result comparisons.
    assert result.accounted_metric_calls == len(calls) == spent + 6

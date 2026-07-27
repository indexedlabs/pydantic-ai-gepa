"""End-to-end coverage for the graph-backed GEPA optimization engine."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa.engines import (
    BudgetExhausted,
    BudgetTracker,
    EngineConfig,
    GepaEngine,
    OptimizationTask,
    get_engine,
)
from pydantic_ai_gepa.types import MetricResult, RolloutOutput


def _task() -> OptimizationTask:
    """Create a tiny task whose seed reaches the graph's perfect-score stop."""
    agent = Agent(TestModel(custom_output_text="ok"), instructions="Reply with ok.")
    cases = [Case(name="case", inputs="hello", expected_output="ok")]

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        score = float(output.result == case.expected_output)
        return MetricResult(score=score, feedback="correct" if score else "incorrect")

    return OptimizationTask(agent=agent, trainset=cases, metric=metric)


def test_registry_constructs_gepa_engine() -> None:
    """Package import registers the built-in graph-backed engine."""
    config = EngineConfig(engine="gepa", max_metric_calls=10)

    engine = get_engine("gepa", config)

    assert isinstance(engine, GepaEngine)


@pytest.mark.asyncio
async def test_gepa_engine_runs_graph_and_spends_shared_budget() -> None:
    """The engine returns the graph result and records its metric-call use."""
    config = EngineConfig(engine="gepa", max_metric_calls=10)
    budget = BudgetTracker(10)

    result = await get_engine("gepa", config).run(_task(), config, budget)

    assert result.engine == "gepa"
    assert result.best_candidate
    assert result.num_metric_calls > 0
    assert result.num_metric_calls <= budget.spent
    assert budget.spent == result.num_metric_calls
    assert any(event.kind == "summary" for event in result.history)


@pytest.mark.asyncio
async def test_gepa_engine_checks_an_exhausted_budget_before_running() -> None:
    """A fresh graph run cannot start after the shared budget is exhausted."""
    config = EngineConfig(engine="gepa", max_metric_calls=1)
    budget = BudgetTracker(1)
    budget.spend(1)

    with pytest.raises(BudgetExhausted):
        await get_engine("gepa", config).run(_task(), config, budget)

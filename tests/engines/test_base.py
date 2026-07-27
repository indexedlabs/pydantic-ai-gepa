"""Tests for shared optimization engine contracts."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa.engines.base import (
    BudgetExhausted,
    BudgetTracker,
    EngineConfig,
    OptimizationTask,
)
from pydantic_ai_gepa.evaluation import EvaluationRecord
from pydantic_ai_gepa.types import MetricResult, RolloutOutput


def test_budget_tracker_enforces_shared_limit() -> None:
    """The tracker reports consumption and refuses overspending."""
    budget = BudgetTracker(3)

    assert budget.spent == 0
    assert budget.remaining == 3
    assert not budget.exhausted

    budget.spend(2)

    assert budget.spent == 2
    assert budget.remaining == 1
    assert not budget.exhausted

    with pytest.raises(BudgetExhausted):
        budget.spend(2)
    assert budget.spent == 2

    budget.spend(1)
    assert budget.exhausted
    with pytest.raises(BudgetExhausted):
        budget.check()


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_metric_calls", 0), ("max_iterations", 0)],
)
def test_engine_config_requires_positive_limits(field: str, value: int) -> None:
    """Engine limits reject zero values."""
    overrides: dict[str, Any] = {field: value}
    with pytest.raises(ValidationError):
        EngineConfig(engine="test", **overrides)


@pytest.mark.asyncio
async def test_optimization_task_seeds_and_evaluates_cases() -> None:
    """The task shares evaluation, candidate extraction, and budget accounting."""
    agent = Agent(TestModel(custom_output_text="ok"), instructions="Reply with ok.")
    cases = [
        Case(name="passing", inputs="one", expected_output="ok"),
        Case(name="failing", inputs="two", expected_output="not ok"),
    ]

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        score = float(output.result == case.expected_output)
        return MetricResult(
            score=score,
            feedback="correct" if score else "incorrect",
            side_info={"expected": case.expected_output},
        )

    task = OptimizationTask(agent=agent, trainset=cases, metric=metric)
    seed = await task.seed_candidate()
    budget = BudgetTracker(2)

    result = await task.evaluate(seed, budget=budget)

    assert "instructions" in seed
    assert await task.seed_candidate() is seed
    assert result.score == 0.5
    assert result.num_cases == 2
    assert budget.spent == 2
    assert result.side_info == {"feedback": ["incorrect"]}


@pytest.mark.asyncio
async def test_optimization_task_aggregates_payload_side_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Payload side information is retained with the case that produced it."""
    agent = Agent(TestModel(custom_output_text="ok"), instructions="Reply with ok.")
    task = OptimizationTask(
        agent=agent,
        trainset=[Case(name="case", inputs="hello", expected_output="ok")],
        metric=lambda case, output: MetricResult(score=1.0),
    )

    async def fake_evaluate_candidate_dataset(**kwargs: Any) -> list[EvaluationRecord]:
        assert len(kwargs["dataset"]) == 1
        return [
            EvaluationRecord(
                case_id="case",
                score=0.0,
                feedback="try again",
                payload={"side_info": {"hint": "be concise"}},
            )
        ]

    monkeypatch.setattr(
        "pydantic_ai_gepa.engines.base.evaluate_candidate_dataset",
        fake_evaluate_candidate_dataset,
    )

    result = await task.evaluate(await task.seed_candidate())

    assert result.side_info == {
        "cases": [{"case_id": "case", "side_info": {"hint": "be concise"}}],
        "feedback": ["try again"],
    }

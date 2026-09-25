"""Dollar-budget engine wiring and subscription proposal accounting."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits
from pydantic_evals import Case

from pydantic_ai_gepa.engines import (
    AutonomousResearchEngine,
    BestOfNEngine,
    BudgetTracker,
    CodingAgentEngine,
    EngineConfig,
    GepaEngine,
    OptimizationTask,
    get_engine,
)
from pydantic_ai_gepa.types import MetricResult


def _task(*, score: float = 0.0) -> OptimizationTask:
    return OptimizationTask(
        agent=Agent(TestModel(custom_output_text="ok"), instructions="Reply with ok."),
        trainset=[Case(name="case", inputs="hello", expected_output="ok")],
        metric=lambda case, output: MetricResult(score=score),
    )


@pytest.mark.parametrize("cap", [0, -1, float("inf"), float("-inf"), float("nan")])
def test_engine_config_requires_positive_finite_dollar_cap(cap: float) -> None:
    with pytest.raises(ValidationError):
        EngineConfig(engine="gepa", max_token_cost=cap)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "engine_type", [BestOfNEngine, CodingAgentEngine, AutonomousResearchEngine]
)
async def test_nonmetering_engines_reject_cap_before_work(engine_type: Any) -> None:
    async def unexpected_callback(*args: Any) -> Any:
        pytest.fail("Dollar caps must be rejected before any user callback runs")

    config = EngineConfig(
        engine=engine_type.name,
        max_token_cost=1.0,
        engine_config={"propose": unexpected_callback, "driver": unexpected_callback},
    )
    with pytest.raises(ValueError, match="cannot meter dollars.*max_token_cost"):
        get_engine(config.engine, config)

    engine = engine_type(config)
    budget = BudgetTracker(10)
    with pytest.raises(ValueError, match="cannot meter dollars.*max_token_cost"):
        await engine.run(_task(), config, budget)
    assert budget.spent == 0


@pytest.mark.asyncio
async def test_gepa_engine_enforces_total_usage_limit() -> None:
    config = EngineConfig(
        engine="gepa",
        engine_config={"gepa_usage_limits": UsageLimits(request_limit=0)},
    )
    result = await GepaEngine(config).run(_task(score=1), config, BudgetTracker(10))
    summary = next(event.data for event in result.history if event.kind == "summary")
    assert summary["stop_reason"] == "Usage budget exceeded"
    assert "spend_report" in summary


@pytest.mark.asyncio
async def test_gepa_engine_rejects_unconstructed_usage_limits() -> None:
    config = EngineConfig(
        engine="gepa", engine_config={"gepa_usage_limits": {"request_limit": 1}}
    )
    with pytest.raises(TypeError, match="gepa_usage_limits.*UsageLimits instance"):
        await GepaEngine(config).run(_task(), config, BudgetTracker(10))


@pytest.mark.asyncio
async def test_gepa_engine_honors_cap_and_reports_actual_spend() -> None:
    config = EngineConfig(
        engine="gepa",
        max_token_cost=0.1,
        engine_config={"price_fn": lambda response: 0.25},
    )
    result = await get_engine("gepa", config).run(_task(), config, BudgetTracker(10))
    summary = next(event.data for event in result.history if event.kind == "summary")
    assert result.best_candidate["instructions"].text == "Reply with ok."
    assert summary["stop_reason"] == "Cost budget reached"
    spend = summary["spend_report"]
    assert spend["max_token_cost"] == 0.1
    assert spend["total_dollars"] == spend["rollout_dollars"] == 0.25
    assert spend["stopped_by_cost"] is True


@pytest.mark.asyncio
async def test_gepa_engine_reports_spend_without_cap() -> None:
    config = EngineConfig(
        engine="gepa",
        stop_at_score=1,
        engine_config={"price_fn": lambda response: 0.25},
    )
    result = await GepaEngine(config).run(_task(score=1), config, BudgetTracker(10))
    summary = next(event.data for event in result.history if event.kind == "summary")
    spend = summary["spend_report"]
    assert spend["max_token_cost"] is None
    assert spend["total_dollars"] == spend["rollout_dollars"] == 0.25
    assert spend["stopped_by_cost"] is False


@pytest.mark.asyncio
async def test_gepa_engine_rejects_noncallable_price_override() -> None:
    config = EngineConfig(engine="gepa", engine_config={"price_fn": {"test": 0.1}})
    with pytest.raises(TypeError, match="price_fn.*Callable"):
        await GepaEngine(config).run(_task(), config, BudgetTracker(10))


@pytest.mark.asyncio
async def test_coding_agent_reports_proposal_count_and_wall_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(
        "pydantic_ai_gepa.engines.coding_agent_engine.perf_counter", lambda: next(ticks)
    )

    async def propose(context: Any) -> Any:
        return context.candidate

    config = EngineConfig(
        engine="coding_agent",
        max_iterations=1,
        engine_config={"propose": propose, "minibatch_size": 1},
    )
    result = await CodingAgentEngine(config).run(_task(), config, BudgetTracker(10))
    summary = next(event.data for event in result.history if event.kind == "summary")
    assert summary["proposals"] == 1
    assert summary["proposal_wall_times_seconds"] == [2.5]
    assert summary["proposal_wall_time_seconds"] == 2.5


@pytest.mark.asyncio
async def test_coding_agent_reports_zero_proposals_when_seed_is_unaffordable() -> None:
    async def unexpected_propose(context: Any) -> Any:
        pytest.fail("Cannot propose without a baseline")

    task = _task()
    task.valset = [Case(name=f"case-{i}", inputs="hello") for i in range(2)]
    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=1,
        engine_config={"propose": unexpected_propose},
    )
    result = await CodingAgentEngine(config).run(task, config, BudgetTracker(1))
    summary = next(event.data for event in result.history if event.kind == "summary")
    assert summary["proposals"] == 0
    assert summary["proposal_wall_times_seconds"] == []
    assert summary["proposal_wall_time_seconds"] == 0

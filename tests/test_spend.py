"""Offline dollar accounting and graceful optimization budget stops."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage
from pydantic_evals import Case

from pydantic_ai_gepa.components import extract_seed_candidate
from pydantic_ai_gepa.gepa_graph.proposal.instruction import (
    ComponentUpdate,
    InstructionProposalOutput,
    TrajectoryAnalysis,
)
from pydantic_ai_gepa.gepa_graph.proposal.trace_tools import create_trace_toolset
from pydantic_ai_gepa.runner import optimize_agent
from pydantic_ai_gepa.spend import (
    COST_STOP_REASON,
    CostBudgetExceeded,
    SpendCapability,
    SpendMeter,
)
from pydantic_ai_gepa.types import MetricResult, ReflectionConfig, RolloutOutput


def _response(model_name: str = "custom-test") -> ModelResponse:
    return ModelResponse(
        parts=[TextPart("ok")],
        model_name=model_name,
        provider_name="openai" if model_name.startswith("gpt-") else "custom",
        usage=RequestUsage(input_tokens=100, output_tokens=50),
    )


def test_catalog_prices_known_response_offline() -> None:
    meter = SpendMeter()
    meter.record("rollout", _response("gpt-4o"))

    report = meter.report()
    assert report.total_dollars > 0
    assert report.rollout_dollars == report.total_dollars
    assert report.by_model["gpt-4o"].input_tokens == 100
    assert report.by_model["gpt-4o"].output_tokens == 50
    assert report.unpriced_usage == {}


def test_override_and_catalog_fallback_price_individual_models() -> None:
    meter = SpendMeter(
        price_fn=lambda response: 0.125
        if response.model_name == "custom-test"
        else None
    )
    meter.record("rollout", _response())
    meter.record("reflection", _response("gpt-4o"))

    report = meter.report()
    assert report.by_model["custom-test"].dollars == 0.125
    assert report.by_model["gpt-4o"].dollars > 0
    assert report.total_dollars == report.rollout_dollars + report.reflection_dollars


@pytest.mark.parametrize("cap", [0, -1, float("inf"), float("nan")])
def test_invalid_caps_are_rejected(cap: float) -> None:
    with pytest.raises(ValueError, match="max_token_cost"):
        SpendMeter(cap)


def test_concurrent_responses_are_counted_once_and_snapshots_are_detached() -> None:
    meter = SpendMeter(price_fn=lambda response: 0.001)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: meter.record("rollout", _response()), range(100)))

    report = meter.report()
    assert report.by_model["custom-test"].requests == 100
    assert report.by_model["custom-test"].input_tokens == 10_000
    assert report.total_dollars == pytest.approx(0.1)
    meter.record("rollout", _response())
    assert report.by_category["rollout"]["custom-test"].requests == 100


def test_projection_uses_completed_steps_instead_of_response_count() -> None:
    meter = SpendMeter(max_token_cost=0.5, price_fn=lambda response: 0.1)
    assert meter.can_start("reflection")
    with meter.step("reflection"):
        for _ in range(3):
            meter.record("reflection", _response())

    # One three-response step costs 0.3, so the next step cannot fit.
    assert not meter.can_start("reflection")
    assert meter.report().total_dollars == pytest.approx(0.3)
    assert meter.report().stop_reason == COST_STOP_REASON


def test_rollout_projection_reserves_the_complete_batch() -> None:
    meter = SpendMeter(max_token_cost=0.35, price_fn=lambda response: 0.1)
    with meter.step("rollout"):
        meter.record("rollout", _response())
    assert meter.can_start("rollout", 2)
    assert not meter.can_start("rollout", 3)
    assert meter.report().total_dollars == 0.1


def test_unknown_price_is_recorded_before_failing_closed() -> None:
    meter = SpendMeter(max_token_cost=1)
    with pytest.raises(CostBudgetExceeded, match="custom-test"):
        meter.record("reflection", _response())
    report = meter.report()
    assert report.stopped_by_cost
    assert report.unpriced_usage["custom-test"].requests == 1
    assert report.unpriced_usage["custom-test"].input_tokens == 100
    assert not meter.can_start("rollout")


@pytest.mark.asyncio
async def test_capability_counts_tool_rounds_and_child_agent_responses() -> None:
    meter = SpendMeter(price_fn=lambda response: 0.05)
    child = Agent(
        TestModel(custom_output_text="child result", model_name="child-test"),
        capabilities=[SpendCapability(meter, "reflection")],
    )
    parent_requests = 0

    async def parent_model(messages, info):
        nonlocal parent_requests
        parent_requests += 1
        if parent_requests == 1:
            return ModelResponse(parts=[ToolCallPart("ask_child", {})])
        return ModelResponse(parts=[TextPart("done")])

    parent = Agent(
        FunctionModel(parent_model, model_name="parent-test"),
        capabilities=[SpendCapability(meter, "reflection")],
    )

    @parent.tool_plain
    async def ask_child() -> str:
        return (await child.run("help")).output

    await parent.run("delegate")
    report = meter.report()
    assert report.by_model["parent-test"].requests == 2
    assert report.by_model["child-test"].requests == 1
    assert report.reflection_dollars == pytest.approx(0.15)
    assert report.rollout_dollars == 0


@pytest.mark.asyncio
async def test_backstop_refuses_another_tool_round_after_overshoot() -> None:
    meter = SpendMeter(max_token_cost=0.01, price_fn=lambda response: 0.02)
    calls = 0

    async def fake_model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(
        FunctionModel(fake_model), capabilities=[SpendCapability(meter, "rollout")]
    )
    with pytest.raises(CostBudgetExceeded):
        await agent.run("first")
    with pytest.raises(CostBudgetExceeded):
        await agent.run("second")
    assert calls == 1
    assert meter.report().total_dollars == 0.02


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [None, 0.01])
async def test_trace_toolset_meters_spawned_child_and_propagates_cost_stop(
    cap: float | None,
) -> None:
    meter = SpendMeter(max_token_cost=cap, price_fn=lambda response: 0.02)

    async def child_model(messages, info):
        return ModelResponse(
            parts=[TextPart("child finding")],
            usage=RequestUsage(input_tokens=10, output_tokens=3),
        )

    toolset = create_trace_toolset(
        "spend-test",
        0,
        reflection_model=FunctionModel(child_model, model_name="trace-child-test"),
        spend_meter=meter,
    )
    spawn_agent = cast(Any, toolset.tools["spawn_agent"].function)
    if cap is None:
        assert await spawn_agent("inspect a trace") == "child finding"
    else:
        # The trace tool's generic error-to-text handler must not hide this stop.
        with pytest.raises(CostBudgetExceeded, match=COST_STOP_REASON):
            await spawn_agent("inspect a trace")

    report = meter.report()
    assert report.by_model["trace-child-test"].requests == 1
    assert report.by_model["trace-child-test"].input_tokens == 10
    assert report.by_model["trace-child-test"].output_tokens == 3
    assert report.reflection_dollars == 0.02
    assert report.rollout_dollars == 0
    assert report.stopped_by_cost is (cap is not None)


def _optimization_inputs() -> dict[str, Any]:
    agent = Agent(
        TestModel(custom_output_text="ok", model_name="student-test"),
        instructions="Original instructions",
    )
    reflection_output = InstructionProposalOutput(
        reasoning=TrajectoryAnalysis(
            pattern_discovery="Some answers are incomplete",
            creative_hypothesis="More specific instructions improve answers",
            experimental_approach="Use explicit instructions",
        ),
        updated_components=[
            ComponentUpdate(
                component_name="instructions", optimized_value="Better instructions"
            )
        ],
    )

    def metric(case: Case, output: RolloutOutput) -> MetricResult:
        return MetricResult(score=0.5, feedback="Needs improvement")

    return {
        "agent": agent,
        "trainset": [Case(name="train", inputs="training prompt")],
        "valset": [Case(name="validation", inputs="validation prompt")],
        "metric": metric,
        "reflection_config": ReflectionConfig(
            model=TestModel(
                call_tools=[],
                model_name="reflector-test",
                custom_output_args=reflection_output.model_dump(mode="python"),
            )
        ),
        "max_metric_calls": 20,
        "reflection_minibatch_size": 1,
        "seed": 0,
    }


@pytest.mark.asyncio
async def test_projection_stops_optimization_with_best_so_far_and_spend() -> None:
    inputs = _optimization_inputs()
    seed = extract_seed_candidate(inputs["agent"])
    result = await optimize_agent(
        **inputs, max_token_cost=0.35, price_fn=lambda response: 0.1
    )

    assert result.best_candidate == seed
    assert result.best_score == 0.5
    assert result.raw_result is not None
    assert result.raw_result.stopped
    assert result.raw_result.stop_reason == COST_STOP_REASON
    report = result.spend_report
    assert report.stopped_by_cost
    assert report.max_token_cost == 0.35
    assert report.reflection_dollars > 0
    assert report.rollout_dollars > 0
    assert report.total_dollars <= 0.35
    assert report.total_dollars == pytest.approx(0.3)
    assert result.raw_result.spend_report == report


@pytest.mark.asyncio
async def test_expensive_reflection_backstop_preserves_seed_and_actual_overshoot() -> (
    None
):
    inputs = _optimization_inputs()
    seed = extract_seed_candidate(inputs["agent"])
    result = await optimize_agent(
        **inputs,
        max_token_cost=0.25,
        price_fn=lambda response: 0.5
        if response.model_name == "reflector-test"
        else 0.01,
    )

    assert result.best_candidate == seed
    assert result.best_score == 0.5
    assert result.raw_result is not None
    assert result.raw_result.stop_reason == COST_STOP_REASON
    assert result.spend_report.reflection_dollars == 0.5
    assert result.spend_report.rollout_dollars == pytest.approx(0.02)
    assert result.spend_report.total_dollars == pytest.approx(0.52)
    assert result.spend_report.stopped_by_cost


@pytest.mark.asyncio
async def test_initial_validation_drains_and_prices_in_flight_requests_before_return() -> (
    None
):
    all_started = asyncio.Event()
    first_priced = asyncio.Event()
    started = 0
    completed: list[int] = []
    priced: list[str | None] = []

    async def delayed_student(messages, info):
        nonlocal started
        index = started
        started += 1
        if started == 3:
            all_started.set()
        await all_started.wait()
        if index:
            await first_priced.wait()
            # Keep these requests in flight when the first response exceeds cap.
            await asyncio.sleep(0.01)
        completed.append(index)
        return ModelResponse(
            parts=[TextPart("ok")],
            usage=RequestUsage(input_tokens=10, output_tokens=3),
        )

    def price_response(response: ModelResponse) -> float:
        priced.append(response.model_name)
        first_priced.set()
        return 0.2

    inputs = _optimization_inputs()
    inputs["agent"] = Agent(
        FunctionModel(delayed_student, model_name="concurrent-student-test"),
        instructions="Original instructions",
    )
    inputs["valset"] = [
        Case(name=f"validation-{index}", inputs=f"prompt {index}") for index in range(3)
    ]
    result = await asyncio.wait_for(
        optimize_agent(**inputs, max_token_cost=0.1, price_fn=price_response),
        timeout=5,
    )

    assert sorted(completed) == [0, 1, 2]
    assert priced == ["concurrent-student-test"] * 3
    assert result.raw_result is not None
    assert result.raw_result.stop_reason == COST_STOP_REASON
    assert result.spend_report.total_dollars == pytest.approx(0.6)
    assert result.spend_report.rollout_dollars == pytest.approx(0.6)
    assert result.spend_report.reflection_dollars == 0
    assert result.spend_report.by_model["concurrent-student-test"].requests == 3
    assert result.spend_report.by_model["concurrent-student-test"].input_tokens == 30
    snapshot = result.spend_report.model_dump()
    await asyncio.sleep(0.02)
    assert started == 3
    assert len(priced) == 3
    assert result.spend_report.model_dump() == snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [None, 1.0])
async def test_optimization_records_unpriced_usage_and_enforces_unknown_model_policy(
    cap: float | None,
) -> None:
    inputs = _optimization_inputs()
    inputs["max_metric_calls"] = 1
    result = await optimize_agent(**inputs, max_token_cost=cap)

    report = result.spend_report
    assert report.unpriced_usage["student-test"].requests == 1
    assert report.unpriced_usage["student-test"].input_tokens > 0
    assert report.total_dollars == 0
    assert report.stopped_by_cost is (cap is not None)
    if cap is not None:
        assert report.stop_reason is not None
        assert "student-test" in report.stop_reason
        assert result.raw_result is not None
        assert result.raw_result.stop_reason == report.stop_reason
    else:
        assert report.stop_reason is None
        assert result.best_score == 0.5

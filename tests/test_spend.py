"""Offline dollar accounting and graceful optimization budget stops."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast
from pathlib import Path

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
    use_rollout_kind,
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
async def test_optimize_agent_cache_hits_are_free_without_diluting_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    start_module = importlib.import_module("pydantic_ai_gepa.gepa_graph.steps.start")
    from pydantic_ai_gepa.cache import CacheManager

    meters = []

    def new_meter(*args, **kwargs):
        meter = SpendMeter(*args, **kwargs)
        meters.append(meter)
        return meter

    monkeypatch.setattr(start_module, "SpendMeter", new_meter)
    model_calls = []

    async def student(messages, info):
        model_calls.append("student")
        return ModelResponse(
            parts=[TextPart("ok")], usage=RequestUsage(input_tokens=10, output_tokens=3)
        )

    hits = {"rollout": 0, "metric": 0}

    def track_hit(method, kind):
        def tracked(*args, **kwargs):
            result = method(*args, **kwargs)
            if result is not None:
                hits[kind] += 1
            return result

        return tracked

    monkeypatch.setattr(
        CacheManager,
        "get_cached_agent_run",
        track_hit(CacheManager.get_cached_agent_run, "rollout"),
    )
    monkeypatch.setattr(
        CacheManager,
        "get_cached_metric_result",
        track_hit(CacheManager.get_cached_metric_result, "metric"),
    )
    inputs = _optimization_inputs()
    inputs["agent"] = Agent(
        FunctionModel(student, model_name="cache-student"),
        instructions="Original instructions",
    )
    inputs.update(
        enable_cache=True,
        cache_rollouts=True,
        cache_metric_results=True,
        cache_metric_identity="spend-test-v1",
        cache_dir=str(tmp_path / "cache"),
        max_iterations=1,
        max_token_cost=10,
        price_fn=lambda response: 0.1
        if response.model_name == "cache-student"
        else 0.01,
    )
    first = await optimize_agent(**inputs)
    first_calls = len(model_calls)
    assert first_calls > 0
    assert first.spend_report.rollout_dollars == pytest.approx(first_calls * 0.1)
    assert first.spend_report.by_model["cache-student"].requests == first_calls
    assert not first.spend_report.stopped_by_cost

    hits.update(rollout=0, metric=0)
    model_calls.clear()
    second = await optimize_agent(**inputs)
    report = second.spend_report
    assert hits["rollout"] > 0 and hits["metric"] > 0
    # Held-out validation still bypasses cache; only fresh model calls cost money.
    assert 0 < len(model_calls) < first_calls
    assert report.by_model["cache-student"].requests == len(model_calls)
    assert report.by_model["cache-student"].input_tokens == len(model_calls) * 10
    assert report.rollout_dollars == pytest.approx(len(model_calls) * 0.1)
    assert report.total_dollars == pytest.approx(
        report.rollout_dollars + report.reflection_dollars
    )
    assert not report.stopped_by_cost
    assert report.rollout_dollars < first.spend_report.rollout_dollars
    # Cached steps must not dilute the observed $0.10 paid-rollout mean.
    meter = meters[-1]
    assert meter.can_start("rollout", 1)
    unaffordable = int((10 - report.total_dollars) / 0.1) + 2
    assert not meter.can_start("rollout", unaffordable)


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
async def test_unobserved_validation_batch_runs_one_rollout_before_cap_binds() -> None:
    started: list[int] = []
    priced: list[str | None] = []

    async def student(messages, info):
        started.append(len(started))
        return ModelResponse(
            parts=[TextPart("ok")],
            usage=RequestUsage(input_tokens=10, output_tokens=3),
        )

    def price_response(response: ModelResponse) -> float:
        priced.append(response.model_name)
        return 0.2

    inputs = _optimization_inputs()
    inputs["agent"] = Agent(
        FunctionModel(student, model_name="concurrent-student-test"),
        instructions="Original instructions",
    )
    seed = extract_seed_candidate(inputs["agent"])
    inputs["valset"] = [
        Case(name=f"validation-{index}", inputs=f"prompt {index}") for index in range(3)
    ]
    result = await asyncio.wait_for(
        optimize_agent(**inputs, max_token_cost=0.1, price_fn=price_response),
        timeout=5,
    )

    # An unobserved rollout kind runs one rollout at a time: the first $0.20
    # validation rollout is the kind's first observation and crosses the $0.10
    # cap by itself (the stated one-rollout margin); the rest of the batch is
    # never admitted, so nothing else is priced.
    assert started == [0]
    assert priced == ["concurrent-student-test"]
    assert result.raw_result is not None
    assert result.raw_result.stop_reason == COST_STOP_REASON
    assert result.spend_report.total_dollars == pytest.approx(0.2)
    assert result.spend_report.rollout_dollars == pytest.approx(0.2)
    assert result.spend_report.reflection_dollars == 0
    assert result.spend_report.by_model["concurrent-student-test"].requests == 1
    assert result.spend_report.stopped_by_cost
    assert result.best_score is None
    assert result.best_candidate == seed
    snapshot = result.spend_report.model_dump()
    await asyncio.sleep(0.02)
    assert started == [0]
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


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_price", [float("nan"), float("inf"), -0.1, "invalid"])
async def test_bad_price_override_stops_optimization_and_records_usage(
    bad_price,
) -> None:
    result = await optimize_agent(
        **_optimization_inputs(),
        max_token_cost=0.35,
        price_fn=lambda response: bad_price,
    )
    report = result.spend_report
    assert report.stopped_by_cost
    assert "student-test" in report.stop_reason
    assert report.unpriced_usage["student-test"].requests == 1
    assert report.by_model["student-test"].input_tokens > 0
    assert "ValueError" in report.stop_reason or "TypeError" in report.stop_reason
    assert result.raw_result.stop_reason == report.stop_reason


@pytest.mark.asyncio
async def test_raising_price_override_records_reflector_and_stops_gracefully() -> None:
    result = await optimize_agent(
        **_optimization_inputs(),
        max_token_cost=0.35,
        price_fn=lambda response: {"student-test": 0.1}[response.model_name],
    )
    report = result.spend_report
    assert report.stopped_by_cost
    assert "reflector-test" in report.stop_reason
    assert "KeyError" in report.stop_reason
    assert report.unpriced_usage["reflector-test"].requests == 1
    assert report.by_model["reflector-test"].output_tokens > 0
    assert report.rollout_dollars == pytest.approx(0.2)
    assert result.best_score == 0.5


@pytest.mark.parametrize("capped", [True, False])
def test_pricing_failure_is_unpriced_even_for_catalog_model(capped: bool) -> None:
    error = RuntimeError("pricing unavailable")

    def broken_price(response):
        raise error

    meter = SpendMeter(max_token_cost=1 if capped else None, price_fn=broken_price)
    if capped:
        with pytest.raises(CostBudgetExceeded, match="RuntimeError") as caught:
            meter.record("rollout", _response("gpt-4o"))
        assert caught.value.__cause__ is error
    else:
        meter.record("rollout", _response("gpt-4o"))
    report = meter.report()
    assert report.stopped_by_cost == capped
    assert report.unpriced_usage["gpt-4o"].requests == 1
    assert report.total_dollars == 0


@pytest.mark.asyncio
async def test_joint_projection_does_not_buy_unaffordable_reflection_and_child() -> (
    None
):
    result = await optimize_agent(
        **_optimization_inputs(), max_token_cost=0.65, price_fn=lambda response: 0.1
    )
    report = result.spend_report
    assert report.stopped_by_cost
    # After one rejected proposal and the next parent batch, $0.15 remains.
    # Reflection and child evaluation each fit separately, but need $0.20 together.
    assert report.total_dollars == pytest.approx(0.5)
    assert report.by_model["reflector-test"].requests == 1
    assert report.by_model["student-test"].requests == 4
    assert result.raw_result.stop_reason == COST_STOP_REASON


@pytest.mark.asyncio
async def test_empty_rollout_observations_ignore_concurrent_paid_responses() -> None:
    meter = SpendMeter(max_token_cost=0.19, price_fn=lambda response: 0.1)
    capability = SpendCapability(meter, "rollout")
    setup_started = asyncio.Event()
    paid_finished = asyncio.Event()

    async def failed_setup():
        setup_started.set()
        await paid_finished.wait()
        raise RuntimeError("setup failed before a model request")

    async def paid_run():
        await setup_started.wait()
        with meter.step("rollout"):
            meter.record("rollout", _response())
        paid_finished.set()

    results = await asyncio.gather(
        capability.wrap_run(cast(Any, None), handler=failed_setup),
        paid_run(),
        return_exceptions=True,
    )
    assert isinstance(results[0], RuntimeError)
    # An additional empty/failed step must not dilute the $0.10 mean either.
    with meter.step("rollout"):
        pass
    assert not meter.can_start("rollout")
    assert meter.report().total_dollars == 0.1


@pytest.mark.asyncio
async def test_bad_price_override_without_cap_keeps_all_usage_unpriced() -> None:
    result = await optimize_agent(
        **_optimization_inputs(),
        max_iterations=1,
        price_fn=lambda response: float("nan"),
    )
    report = result.spend_report
    assert not report.stopped_by_cost
    assert report.total_dollars == 0
    assert report.unpriced_usage["student-test"].requests > 0
    assert report.unpriced_usage["reflector-test"].requests > 0


@pytest.mark.asyncio
async def test_expensive_validation_probe_ends_within_one_rollout_of_cap() -> None:
    """The reviewer's OTTO-4763 probe on the Python API.

    Four validation rollouts at $0.50 each; training rollouts and reflection
    cost $0.005; cap $0.60 with enough concurrency to start the whole batch.
    """
    from pydantic_ai_gepa._validation import validation_active

    def price_response(response: ModelResponse) -> float:
        if response.model_name == "reflector-test":
            return 0.005
        return 0.5 if validation_active() else 0.005

    inputs = _optimization_inputs()
    seed = extract_seed_candidate(inputs["agent"])
    inputs["valset"] = [
        Case(name=f"validation-{index}", inputs=f"prompt {index}") for index in range(4)
    ]
    result = await optimize_agent(
        **inputs, max_token_cost=0.60, price_fn=price_response
    )

    report = result.spend_report
    assert report.stopped_by_cost
    assert report.stop_reason == COST_STOP_REASON
    assert result.raw_result is not None
    assert result.raw_result.stop_reason == COST_STOP_REASON
    # The first $0.50 validation rollout observes its kind; the other three are
    # never admitted. The run ends under the cap; the stated margin is one
    # rollout at the kind's observed high: 0.60 + 0.50.
    assert report.by_model["student-test"].requests == 1
    assert report.rollout_dollars == pytest.approx(0.50)
    assert report.total_dollars == pytest.approx(0.50)
    assert report.total_dollars <= 0.60 + 0.50
    # OTTO-4707: the partly validated seed is never picked and reports no score.
    assert result.best_score is None
    assert result.best_candidate == seed


@pytest.mark.asyncio
@pytest.mark.parametrize("case_count,cap", [(11, 1.00), (4, 0.60)])
async def test_cheap_first_validation_case_stays_within_one_rollout_of_cap(
    case_count: int, cap: float
) -> None:
    """The reviewer's OTTO-4840 probe on the Python API.

    The first validation case costs $0.01 and every other validation case
    $0.50 (training rollouts and reflection $0.005), at the default
    ``max_concurrent_evaluations=10``. The admission gate ramps a kind's
    in-flight limit with its observation count, so the cheap first case
    admits only one more rollout; its $0.50 becomes the kind's observed high
    and the run stops within one rollout of the cap instead of starting the
    whole batch on the $0.01 estimate.
    """
    from pydantic_ai_gepa._validation import validation_active

    cheap_case_pending = True

    def price_response(response: ModelResponse) -> float:
        nonlocal cheap_case_pending
        if response.model_name == "reflector-test" or not validation_active():
            return 0.005
        if cheap_case_pending:
            # The unobserved validation kind runs its first rollout alone, so
            # the cheap case is priced first deterministically.
            cheap_case_pending = False
            return 0.01
        return 0.50

    async def student(messages, info):
        # Fake latency so every admitted in-flight rollout issues its request
        # before the first priced response can stop the run.
        await asyncio.sleep(0.01)
        return ModelResponse(
            parts=[TextPart("ok")],
            usage=RequestUsage(input_tokens=10, output_tokens=3),
        )

    inputs = _optimization_inputs()
    seed = extract_seed_candidate(inputs["agent"])
    inputs["agent"] = Agent(
        FunctionModel(student, model_name="student-test"),
        instructions="Original instructions",
    )
    inputs["valset"] = [
        Case(name=f"validation-{index}", inputs=f"prompt {index}")
        for index in range(case_count)
    ]
    result = await optimize_agent(**inputs, max_token_cost=cap, price_fn=price_response)

    report = result.spend_report
    assert report.stopped_by_cost
    assert report.stop_reason == COST_STOP_REASON
    assert result.raw_result is not None
    assert result.raw_result.stop_reason == COST_STOP_REASON
    # The $0.01 first case observes the kind; the ramp admits one more
    # rollout, whose $0.50 sets the kind's high. No further rollout fits in
    # the remaining headroom, so the run stops at $0.51.
    assert report.by_model["student-test"].requests == 2
    assert report.rollout_dollars == pytest.approx(0.51)
    assert report.total_dollars == pytest.approx(0.51)
    assert report.total_dollars <= cap + 0.50
    # OTTO-4707: the partly validated seed is never picked and reports no score.
    assert result.best_score is None
    assert result.best_candidate == seed


@pytest.mark.asyncio
async def test_uncapped_cheap_first_validation_keeps_full_concurrency() -> None:
    """OTTO-4840: without a cap the same cheap-then-expensive run is unchanged.

    No cap means no waiting and no serialization: the whole validation batch
    overlaps up to the ``max_concurrent_evaluations`` semaphore and every
    rollout is charged, exactly as before the admission ramp.
    """
    from pydantic_ai_gepa._validation import validation_active

    cheap_case_pending = True

    def price_response(response: ModelResponse) -> float:
        nonlocal cheap_case_pending
        if response.model_name == "reflector-test" or not validation_active():
            return 0.005
        if cheap_case_pending:
            cheap_case_pending = False
            return 0.01
        return 0.50

    in_flight = 0
    peak = 0

    async def student(messages, info):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.01)
            return ModelResponse(
                parts=[TextPart("ok")],
                usage=RequestUsage(input_tokens=10, output_tokens=3),
            )
        finally:
            in_flight -= 1

    inputs = _optimization_inputs()
    inputs["agent"] = Agent(
        FunctionModel(student, model_name="student-test"),
        instructions="Original instructions",
    )
    inputs["valset"] = [
        Case(name=f"validation-{index}", inputs=f"prompt {index}")
        for index in range(11)
    ]
    # End the run right after the 11-case seed validation.
    inputs["max_metric_calls"] = 11
    result = await optimize_agent(**inputs, price_fn=price_response)

    report = result.spend_report
    assert not report.stopped_by_cost
    assert report.stop_reason is None
    # The full batch overlaps up to the default max_concurrent_evaluations=10
    # semaphore, and all of it is charged.
    assert peak == 10
    assert report.by_model["student-test"].requests == 11
    assert report.rollout_dollars == pytest.approx(0.01 + 10 * 0.50)
    assert report.total_dollars == pytest.approx(5.01)
    assert result.best_score == 0.5


@pytest.mark.asyncio
async def test_per_kind_projection_is_not_diluted_by_cheap_training_rollouts() -> None:
    """Validation is first observed after many cheap training rollouts.

    A blended mean ($0.017) projects a 4-case validation batch at $0.07 and
    admits it; the per-kind guard sees the observed $0.50 validation cost.
    """
    meter = SpendMeter(
        max_token_cost=0.80, price_fn=lambda response: 0.005, max_concurrent=4
    )
    for _ in range(40):
        with meter.step("rollout", kind="training"):
            meter.record("rollout", _response())

    assert meter.can_start("rollout", 4, rollout_kind="validation")
    capability = SpendCapability(meter, "rollout")
    started = 0

    async def expensive_rollout() -> None:
        nonlocal started
        started += 1
        meter.record("rollout", _response())

    async def validation_rollout() -> None:
        with use_rollout_kind("validation"):
            await capability.wrap_run(cast(Any, None), handler=expensive_rollout)

    # The unobserved validation kind admits one rollout; at $0.50 it is the
    # blended mean's blind spot but still fits under the remaining $0.60.
    meter.price_fn = lambda response: 0.5
    first = await asyncio.gather(validation_rollout(), return_exceptions=True)
    assert first == [None]
    assert meter.report().total_dollars == pytest.approx(0.70)

    # Now the kind is observed: the batch guard projects 4 * $0.50 and refuses,
    # while the blended projection would still allow it.
    assert meter.can_start("rollout", 4)
    assert not meter.can_start("rollout", 4, rollout_kind="validation")
    results = await asyncio.gather(
        *(validation_rollout() for _ in range(3)), return_exceptions=True
    )
    assert all(isinstance(result, CostBudgetExceeded) for result in results)
    assert started == 1
    assert meter.report().total_dollars == pytest.approx(0.70)
    assert meter.report().stop_reason == COST_STOP_REASON


@pytest.mark.asyncio
async def test_admission_serializes_an_unobserved_kind_and_stops_at_cap() -> None:
    meter = SpendMeter(
        max_token_cost=0.60, price_fn=lambda response: 0.5, max_concurrent=4
    )
    capability = SpendCapability(meter, "rollout")
    in_flight = 0
    max_in_flight = 0
    completed = 0

    async def rollout() -> None:
        nonlocal in_flight, max_in_flight, completed
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            await asyncio.sleep(0.01)
            meter.record("rollout", _response())
            completed += 1
        finally:
            in_flight -= 1

    async def one() -> None:
        await capability.wrap_run(cast(Any, None), handler=rollout)

    results = await asyncio.gather(*(one() for _ in range(4)), return_exceptions=True)
    # One $0.50 rollout observed the kind; the next cannot fit in the remaining
    # $0.10, so the run stops with nothing in flight instead of overshooting.
    assert completed == 1
    assert max_in_flight == 1
    assert sum(isinstance(result, CostBudgetExceeded) for result in results) == 3
    assert meter.report().total_dollars == pytest.approx(0.50)
    assert meter.report().stop_reason == COST_STOP_REASON


@pytest.mark.asyncio
async def test_admission_drops_to_one_rollout_near_the_cap() -> None:
    meter = SpendMeter(
        max_token_cost=0.35, price_fn=lambda response: 0.1, max_concurrent=8
    )
    capability = SpendCapability(meter, "rollout")

    async def seed() -> None:
        meter.record("rollout", _response())

    await capability.wrap_run(cast(Any, None), handler=seed)
    assert meter.report().total_dollars == pytest.approx(0.1)

    in_flight = 0
    max_in_flight = 0
    completed = 0

    async def rollout() -> None:
        nonlocal in_flight, max_in_flight, completed
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            await asyncio.sleep(0.005)
            meter.record("rollout", _response())
            completed += 1
        finally:
            in_flight -= 1

    async def one() -> None:
        await capability.wrap_run(cast(Any, None), handler=rollout)

    results = await asyncio.gather(*(one() for _ in range(8)), return_exceptions=True)
    # After the $0.10 seed, headroom $0.25 is below the ramped limit times the
    # observed high at every step (1 slot at one observation, then $0.15 < 2 *
    # $0.10), so rollouts run one at a time: two fit ($0.20, $0.30), then no
    # further rollout fits with nothing in flight.
    assert max_in_flight == 1
    assert completed == 2
    assert sum(isinstance(result, CostBudgetExceeded) for result in results) == 6
    assert meter.report().total_dollars == pytest.approx(0.30)
    assert meter.report().stop_reason == COST_STOP_REASON


@pytest.mark.asyncio
async def test_admission_releases_waiting_rollouts_as_in_flight_ones_finish() -> None:
    meter = SpendMeter(
        max_token_cost=1.00, price_fn=lambda response: 0.1, max_concurrent=3
    )
    capability = SpendCapability(meter, "rollout")

    async def seed() -> None:
        meter.record("rollout", _response())

    # Three observations ramp the kind's in-flight limit to the full
    # concurrency of 3 before the gated batch starts.
    for _ in range(3):
        await capability.wrap_run(cast(Any, None), handler=seed)

    gate = asyncio.Event()
    started = 0
    completed = 0

    async def rollout() -> None:
        nonlocal started, completed
        started += 1
        await gate.wait()
        meter.record("rollout", _response())
        completed += 1

    async def one() -> None:
        await capability.wrap_run(cast(Any, None), handler=rollout)

    tasks = [asyncio.create_task(one()) for _ in range(5)]
    await asyncio.sleep(0.05)
    # Headroom $0.70 covers 3 * $0.10, so the concurrency limit binds, not the
    # cap: three rollouts run, two wait for in-flight slots.
    assert started == 3
    gate.set()
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
    assert completed == 5
    assert meter.report().total_dollars == pytest.approx(0.80)
    assert not meter.report().stopped_by_cost


@pytest.mark.asyncio
async def test_in_flight_rollouts_settle_when_a_new_price_high_stops_the_run() -> None:
    prices = iter([0.1, 0.1, 2.0, 0.1])
    meter = SpendMeter(
        max_token_cost=1.00,
        price_fn=lambda response: next(prices),
        max_concurrent=2,
    )
    capability = SpendCapability(meter, "rollout")

    async def seed() -> None:
        meter.record("rollout", _response())

    # Two observations ramp the kind's in-flight limit to the full
    # concurrency of 2, so the expensive and cheap rollouts below overlap.
    for _ in range(2):
        await capability.wrap_run(cast(Any, None), handler=seed)

    expensive_started = asyncio.Event()
    cheap_release = asyncio.Event()

    async def expensive() -> None:
        expensive_started.set()
        await asyncio.sleep(0)
        # A new price high for the kind crosses the cap while admitted.
        meter.record("rollout", _response())

    async def cheap() -> None:
        await expensive_started.wait()
        await cheap_release.wait()
        meter.record("rollout", _response())

    async def queued() -> None:
        meter.record("rollout", _response())

    async def run(handler) -> Exception | None:
        try:
            await capability.wrap_run(cast(Any, None), handler=handler)
        except Exception as error:  # noqa: BLE001 - collected for assertions
            return error
        return None

    expensive_task = asyncio.create_task(run(expensive))
    cheap_task = asyncio.create_task(run(cheap))
    queued_task = asyncio.create_task(run(queued))
    await asyncio.wait_for(expensive_task, timeout=5)
    # The queued rollout was waiting for a slot and exits on the stop; the
    # in-flight cheap rollout still settles and is paid for.
    assert isinstance(
        await asyncio.wait_for(queued_task, timeout=5), CostBudgetExceeded
    )
    cheap_release.set()
    cheap_error = await asyncio.wait_for(cheap_task, timeout=5)

    assert isinstance(await expensive_task, CostBudgetExceeded)
    assert isinstance(cheap_error, CostBudgetExceeded)
    report = meter.report()
    assert report.by_model["custom-test"].requests == 4
    assert report.rollout_dollars == pytest.approx(0.1 + 0.1 + 2.0 + 0.1)
    assert report.stopped_by_cost
    assert report.stop_reason == COST_STOP_REASON


@pytest.mark.asyncio
@pytest.mark.parametrize("max_token_cost", [None, 100.0])
async def test_admission_never_serializes_without_a_binding_cap(
    max_token_cost: float | None,
) -> None:
    # Uncapped meters admit immediately; so do capped meters without a
    # configured concurrency (the CLI-managed path gates rollouts itself).
    meter = SpendMeter(max_token_cost=max_token_cost, price_fn=lambda response: 0.1)
    capability = SpendCapability(meter, "rollout")
    all_started = asyncio.Event()
    started = 0
    completed = 0

    async def rollout() -> None:
        nonlocal started, completed
        started += 1
        if started == 4:
            all_started.set()
        await all_started.wait()
        meter.record("rollout", _response())
        completed += 1

    async def one() -> None:
        await capability.wrap_run(cast(Any, None), handler=rollout)

    await asyncio.wait_for(asyncio.gather(*(one() for _ in range(4))), timeout=5)
    assert started == completed == 4
    assert meter.report().total_dollars == pytest.approx(0.4)
    assert not meter.report().stopped_by_cost


@pytest.mark.asyncio
async def test_admission_keeps_full_concurrency_with_ample_headroom() -> None:
    meter = SpendMeter(
        max_token_cost=10.0, price_fn=lambda response: 0.1, max_concurrent=4
    )
    capability = SpendCapability(meter, "rollout")

    async def seed() -> None:
        meter.record("rollout", _response())

    # Four observations ramp the kind's in-flight limit to the full
    # concurrency of 4 with ample headroom before the gated batch starts.
    for _ in range(4):
        await capability.wrap_run(cast(Any, None), handler=seed)

    all_started = asyncio.Event()
    started = 0

    async def rollout() -> None:
        nonlocal started
        started += 1
        if started == 4:
            all_started.set()
        await all_started.wait()
        meter.record("rollout", _response())

    async def one() -> None:
        await capability.wrap_run(cast(Any, None), handler=rollout)

    await asyncio.wait_for(asyncio.gather(*(one() for _ in range(4))), timeout=5)
    assert started == 4
    assert meter.report().total_dollars == pytest.approx(0.8)
    assert not meter.report().stopped_by_cost


@pytest.mark.asyncio
async def test_admission_release_survives_a_second_cancellation() -> None:
    """A rollout cancelled again while releasing must not leak its slot."""
    meter = SpendMeter(
        max_token_cost=10.0, price_fn=lambda response: 0.1, max_concurrent=2
    )
    capability = SpendCapability(meter, "rollout")

    async def seed() -> None:
        meter.record("rollout", _response())

    await capability.wrap_run(cast(Any, None), handler=seed)

    entered = asyncio.Event()
    never = asyncio.Event()

    async def rollout() -> None:
        entered.set()
        await never.wait()

    task = asyncio.create_task(capability.wrap_run(cast(Any, None), handler=rollout))
    await entered.wait()
    assert meter._active["training"] == 1
    assert meter._reserved == pytest.approx(0.1)

    # Hold the condition lock so the release's notify acquire has to queue,
    # then cancel the rollout a second time while it is queued there.
    await meter._condition.acquire()
    task.cancel()
    for _ in range(5):
        await asyncio.sleep(0)
    assert meter._active["training"] == 0
    assert meter._reserved == pytest.approx(0.0)
    task.cancel()
    meter._condition.release()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The condition is not corrupted: a later rollout is admitted normally.
    async def followup() -> None:
        meter.record("rollout", _response())

    await asyncio.wait_for(
        capability.wrap_run(cast(Any, None), handler=followup), timeout=5
    )
    assert meter._active["training"] == 0
    assert meter._reserved == pytest.approx(0.0)
    assert meter.report().total_dollars == pytest.approx(0.2)
    assert not meter.report().stopped_by_cost

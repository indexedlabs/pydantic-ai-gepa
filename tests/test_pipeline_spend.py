"""Offline regression coverage for one dollar cap across composed engines."""

import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa.compose import (
    OmniPlan,
    PipelineResult,
    optimize_adaptive_sequential,
    optimize_best_of,
    optimize_omni,
    optimize_parallel,
    optimize_sequential,
    optimize_vote,
)
from pydantic_ai_gepa.engines import EngineConfig, OptimizationTask
from pydantic_ai_gepa.gepa_graph.proposal.instruction import (
    ComponentUpdate,
    InstructionProposalOutput,
    TrajectoryAnalysis,
)
from pydantic_ai_gepa.spend import CostBudgetExceeded, SpendMeter
from pydantic_ai_gepa.types import MetricResult, ReflectionConfig


def _task(*, cases=1, model=None, concurrency=2):
    return OptimizationTask(
        agent=Agent(
            model or TestModel(custom_output_text="ok", model_name="student"),
            instructions="seed",
        ),
        trainset=[Case(name="private-training", inputs="train")],
        valset=[
            Case(name=f"secret-validation-{i}", inputs="val") for i in range(cases)
        ],
        test_set=[Case(name="secret-test", inputs="test")],
        metric=lambda case, output: MetricResult(score=0.5, feedback="Improve"),
        concurrency=concurrency,
    )


def _config(*, cap=100, calls=20, iterations=None):
    output = InstructionProposalOutput(
        reasoning=TrajectoryAnalysis(
            pattern_discovery="Incomplete",
            creative_hypothesis="Be precise",
            experimental_approach="Specify answers",
        ),
        updated_components=[
            ComponentUpdate(component_name="instructions", optimized_value="better")
        ],
    )
    return EngineConfig(
        engine="gepa",
        max_metric_calls=calls,
        max_token_cost=cap,
        max_iterations=iterations,
        engine_config={
            "reflection_minibatch_size": 1,
            "reflection_config": ReflectionConfig(
                model=TestModel(
                    model_name="reflector",
                    call_tools=[],
                    custom_output_args=output.model_dump(mode="python"),
                )
            ),
        },
    )


def _price(response):
    return 0.125 if response.model_name == "student" else 0.03125


def _engine_spend(result):
    return result.history[-1].data["spend_report"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "helper", [optimize_sequential, optimize_parallel, optimize_adaptive_sequential]
)
async def test_pipeline_cap_combines_engines_and_helper_rollouts(helper):
    prices = []

    def price(response):
        dollars = _price(response)
        prices.append(dollars)
        return dollars

    meter = SpendMeter(0.8, price, max_concurrent=2)
    result = await helper(
        _task(),
        [_config(), _config()],
        max_metric_calls=40,
        spend_meter=meter,
    )
    report = meter.report()
    if helper is optimize_parallel:
        assert isinstance(result, list)
        assert len(result) == 2
    else:
        assert isinstance(result, PipelineResult)
        assert result.spend_report == report
        assert result.decision["stopped_by_cost"]
    assert report.stopped_by_cost
    assert report.max_token_cost == 0.8
    assert report.total_dollars == sum(prices)
    assert report.total_dollars <= 0.8 + 2 * 0.125
    assert report.total_dollars == report.rollout_dollars + report.reflection_dollars
    assert report.reflection_dollars > 0
    assert "secret-validation" not in report.model_dump_json()
    assert "private-training" not in report.model_dump_json()
    from pydantic_ai_gepa.spend import _pipeline_meter

    assert _pipeline_meter.get() is None
    if helper is not optimize_parallel:
        # The seed was completely evaluated before optimization exhausted dollars.
        assert result.best.engine == "seed"
        assert result.best.best_score == 0.5
        assert report.total_dollars > sum(
            _engine_spend(r)["total_dollars"] for r in result.results
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", [optimize_best_of, optimize_vote])
async def test_interrupted_vote_discards_even_completed_candidate_samples(helper):
    task = _task(cases=2)
    result = await helper(
        task,
        [_config(calls=2), _config(calls=2)],
        max_metric_calls=4,
        fair_vote_repetitions=1,
        max_token_cost=0.9,
        price_fn=_price,
    )
    assert all(_engine_spend(r)["total_dollars"] == 0.25 for r in result.results)
    assert result.spend_report.total_dollars == 0.875
    assert result.spend_report.stopped_by_cost
    assert result.spend_report.rollout_dollars > 0.5  # Comparison really ran.
    assert result.decision["comparison_discarded"]
    assert result.fair_votes == []
    assert result.best_index == -1
    assert result.best.best_candidate == await task.seed_candidate()
    assert result.best.best_score is None


@pytest.mark.asyncio
async def test_smaller_engine_cap_does_not_stop_the_pipeline():
    result = await optimize_sequential(
        _task(),
        [_config(cap=0.2), _config(cap=0.2)],
        max_metric_calls=40,
        max_token_cost=5,
        price_fn=_price,
    )
    assert len(result.results) == 2
    assert all(_engine_spend(r)["stopped_by_cost"] for r in result.results)
    assert all(_engine_spend(r)["total_dollars"] == 0.25 for r in result.results)
    assert result.spend_report.total_dollars == 0.875
    assert not result.spend_report.stopped_by_cost


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "helper", [optimize_sequential, optimize_adaptive_sequential, optimize_parallel]
)
async def test_interrupted_seed_is_unscored(helper):
    result = await helper(
        _task(cases=3),
        [_config(), _config()],
        max_metric_calls=40,
        max_token_cost=0.0625,
        price_fn=_price,
    )
    if helper is optimize_parallel:
        assert isinstance(result, list)
        assert all(r.best_score is None for r in result)
        assert sum(_engine_spend(r)["total_dollars"] for r in result) == 0.125
        return
    assert result.best.engine == "seed"
    assert result.best.best_score is None
    assert result.spend_report.total_dollars == 0.125
    assert result.spend_report.stopped_by_cost


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "helper",
    [
        optimize_parallel,
        optimize_sequential,
        optimize_best_of,
        optimize_vote,
        optimize_adaptive_sequential,
        optimize_omni,
    ],
)
async def test_unsupported_engine_rejected_before_any_paid_work(helper):
    prices = []
    configs = [
        _config(),
        EngineConfig(
            engine="best_of_n",
            max_metric_calls=20,
            engine_config={"propose": lambda seed: seed},
        ),
    ]
    args = (
        OmniPlan(
            phase_one=configs,
            phase_two=_config(),
            phase_one_metric_calls=40,
            phase_two_metric_calls=20,
        )
        if helper is optimize_omni
        else configs
    )
    kwargs = {} if helper is optimize_omni else {"max_metric_calls": 40}
    with pytest.raises(ValueError, match="best_of_n.*cannot meter"):
        await helper(
            _task(),
            args,
            max_token_cost=1,
            price_fn=lambda response: prices.append(response) or 0.125,
            **kwargs,
        )
    assert prices == []


@pytest.mark.asyncio
async def test_parallel_children_cannot_reserve_the_same_last_dollars():
    parent = SpendMeter(0.3, lambda response: 0.125, max_concurrent=2)
    children = [SpendMeter(10, parent=parent, max_concurrent=2) for _ in range(2)]
    response = ModelResponse(parts=[TextPart("ok")], model_name="student")
    with parent.step("rollout", kind="validation"):
        parent.record("rollout", response)
    entered = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def rollout(index):
        async with children[index].admit_rollout("validation"):
            entered.append(index)
            started.set()
            await release.wait()
            with children[index].step("rollout", kind="validation"):
                children[index].record("rollout", response)

    first = asyncio.create_task(rollout(0))
    await started.wait()
    second = asyncio.create_task(rollout(1))
    await asyncio.sleep(0)
    assert entered == [0]
    release.set()
    outcomes = await asyncio.gather(first, second, return_exceptions=True)
    assert outcomes[0] is None
    assert isinstance(outcomes[1], CostBudgetExceeded)
    assert entered == [0]
    assert parent.report().total_dollars == 0.25
    assert parent.report().stopped_by_cost


@pytest.mark.asyncio
async def test_uncapped_pipeline_preserves_concurrency_and_list_return():
    started = 0
    both_started = asyncio.Event()

    async def model(messages, info):
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=5)
        return ModelResponse(parts=[TextPart("ok")])

    results = await optimize_parallel(
        _task(model=FunctionModel(model, model_name="student")),
        [_config(cap=None, calls=1), _config(cap=None, calls=1)],
        max_metric_calls=2,
        price_fn=_price,
    )
    assert isinstance(results, list)
    assert len(results) == 2
    assert started == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [0.3, 0.6, 0.8, 5])
async def test_single_family_omni_meters_continuation_and_reporting(cap):
    config = _config(calls=1)
    plan = OmniPlan(
        phase_one=[config],
        phase_two=config,
        phase_one_metric_calls=1,
        phase_two_metric_calls=1,
        fair_vote_repetitions=1,
        fair_vote_max_repetitions=1,
    )
    prices = []
    result = await optimize_omni(
        _task(),
        plan,
        max_token_cost=cap,
        price_fn=lambda response: prices.append(_price(response)) or _price(response),
    )
    assert result.spend_report.total_dollars == sum(prices)
    assert result.spend_report.total_dollars <= cap + 0.125
    if cap == 5:
        assert result.spend_report.total_dollars == 0.875
        assert result.reporting_metric_calls == 1
        assert result.test_score == 0.5
        assert not result.spend_report.stopped_by_cost
    else:
        assert result.spend_report.stopped_by_cost
        assert result.test_score is None


@pytest.mark.asyncio
async def test_complete_vote_at_exact_cap_is_retained():
    result = await optimize_best_of(
        _task(),
        [_config(calls=1), _config(calls=1)],
        max_metric_calls=2,
        fair_vote_repetitions=1,
        max_token_cost=0.5,
        price_fn=_price,
    )
    assert result.spend_report.total_dollars == 0.5
    assert result.spend_report.stopped_by_cost
    assert len(result.fair_votes) == 2
    assert result.best_index == 0
    assert result.best.best_score == 0.5
    assert not result.decision.get("comparison_discarded")


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", [optimize_best_of, optimize_vote])
async def test_supplied_meter_stops_vote_and_is_reported(helper):
    meter = SpendMeter(0.3, _price, max_concurrent=2)
    result = await helper(
        _task(),
        [_config(calls=1)],
        max_metric_calls=1,
        fair_vote_repetitions=2,
        spend_meter=meter,
    )
    assert result.spend_report == meter.report()
    assert result.spend_report.total_dollars == 0.25
    assert result.spend_report.stopped_by_cost
    assert result.best.best_score is None


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [{"max_token_cost": 1}, {"price_fn": _price}])
async def test_supplied_meter_rejects_ambiguous_configuration(options):
    with pytest.raises(ValueError, match="mutually exclusive"):
        await optimize_sequential(
            _task(),
            [_config()],
            max_metric_calls=20,
            spend_meter=SpendMeter(1, _price, max_concurrent=2),
            **options,
        )


@pytest.mark.asyncio
async def test_supplied_capped_meter_requires_an_admission_limit():
    with pytest.raises(ValueError, match="max_concurrent"):
        await optimize_parallel(
            _task(),
            [_config()],
            max_metric_calls=20,
            spend_meter=SpendMeter(1, _price),
        )


@pytest.mark.asyncio
async def test_already_stopped_supplied_meter_returns_seed_without_work():
    meter = SpendMeter(0.125, _price, max_concurrent=2)
    meter.record("rollout", ModelResponse(parts=[TextPart("ok")], model_name="student"))
    result = await optimize_best_of(
        _task(),
        [_config(), _config()],
        max_metric_calls=20,
        spend_meter=meter,
    )
    assert result.spend_report.total_dollars == 0.125
    assert result.spend_report.stopped_by_cost
    assert result.results == []
    assert result.best.best_score is None


@pytest.mark.asyncio
async def test_unknown_price_fails_closed_with_aggregate_report():
    result = await optimize_sequential(
        _task(),
        [_config()],
        max_metric_calls=20,
        max_token_cost=1,
        price_fn=lambda response: None,
    )
    assert result.spend_report.stopped_by_cost
    assert result.spend_report.unpriced_usage["student"].requests == 1
    assert result.best.best_score is None
    assert result.results == []


@pytest.mark.asyncio
async def test_two_gepa_engines_race_under_one_admission_gate():
    entered = 0
    second_started = asyncio.Event()
    release = asyncio.Event()

    async def model(messages, info):
        nonlocal entered
        entered += 1
        if entered == 2:
            second_started.set()
            await release.wait()
        return ModelResponse(parts=[TextPart("ok")])

    meter = SpendMeter(0.3, _price, max_concurrent=2)
    pipeline = asyncio.create_task(
        optimize_parallel(
            _task(cases=2, model=FunctionModel(model, model_name="student")),
            [_config(calls=2), _config(calls=2)],
            max_metric_calls=4,
            spend_meter=meter,
        )
    )
    try:
        await asyncio.wait_for(second_started.wait(), timeout=5)
        await asyncio.sleep(0)
        assert entered == 2
    finally:
        release.set()
    results = await asyncio.wait_for(pipeline, timeout=5)
    assert entered == 2
    assert meter.report().total_dollars == 0.25
    assert meter.report().stopped_by_cost
    assert sum(_engine_spend(r)["total_dollars"] for r in results) == 0.25


@pytest.mark.asyncio
async def test_reflection_waits_for_reserved_rollout_and_blocks_other_starts():
    parent = SpendMeter(0.5, _price, max_concurrent=2)
    rollout_meter = SpendMeter(parent=parent)
    reflection_meter = SpendMeter(parent=parent)
    events = []
    release_rollout = asyncio.Event()
    release_reflection = asyncio.Event()
    rollout_started = asyncio.Event()
    reflection_started = asyncio.Event()

    async def reflection():
        async with reflection_meter.admit_reflection():
            events.append("reflection")
            reflection_started.set()
            await release_reflection.wait()

    async def rollout(first=False):
        async with rollout_meter.admit_rollout("validation"):
            events.append("rollout")
            if first:
                rollout_started.set()
                await release_rollout.wait()

    first = asyncio.create_task(rollout(True))
    await asyncio.wait_for(rollout_started.wait(), timeout=5)
    pending_reflection = asyncio.create_task(reflection())
    await asyncio.sleep(0)
    assert events == ["rollout"]
    release_rollout.set()
    await asyncio.wait_for(reflection_started.wait(), timeout=5)
    pending_rollout = asyncio.create_task(rollout())
    await asyncio.sleep(0)
    assert events == ["rollout", "reflection"]
    release_reflection.set()
    await asyncio.wait_for(
        asyncio.gather(first, pending_reflection, pending_rollout), timeout=5
    )
    assert events == ["rollout", "reflection", "rollout"]


@pytest.mark.asyncio
async def test_supplied_child_meter_cannot_hide_its_ancestors_cap():
    meter = SpendMeter(parent=SpendMeter(1, _price, max_concurrent=2))
    with pytest.raises(ValueError, match="best_of_n.*cannot meter"):
        await optimize_sequential(
            _task(),
            [
                EngineConfig(
                    engine="best_of_n", engine_config={"propose": lambda seed: seed}
                )
            ],
            max_metric_calls=20,
            spend_meter=meter,
        )
    assert meter.report().total_dollars == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", [optimize_best_of, optimize_omni])
async def test_unselectable_comparison_at_exact_cap_returns_unscored_seed(helper):
    task = _task()
    task.metric = lambda case, output: MetricResult(
        score=0.5, side_info={"selectable": False}
    )
    config = _config(calls=1)
    if helper is optimize_omni:
        result = await helper(
            task,
            OmniPlan(
                phase_one=[config],
                phase_two=config,
                phase_one_metric_calls=1,
                phase_two_metric_calls=1,
                fair_vote_repetitions=1,
                fair_vote_max_repetitions=1,
            ),
            max_token_cost=0.375,
            price_fn=_price,
        )
    else:
        result = await helper(
            task,
            [config],
            max_metric_calls=1,
            fair_vote_repetitions=1,
            max_token_cost=0.25,
            price_fn=_price,
        )
    assert result.spend_report.stopped_by_cost
    assert result.spend_report.total_dollars == result.spend_report.max_token_cost
    assert result.best.engine == "seed"
    assert result.best.best_score is None
    assert result.fair_votes == []

"""Offline regression coverage for one dollar cap across composed engines."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

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
from pydantic_ai_gepa.engines import (
    BudgetTracker,
    EngineConfig,
    EngineResult,
    OptimizationTask,
    register_engine,
    unregister_engine,
)
from pydantic_ai_gepa.engines.gepa_engine import GepaEngine
from pydantic_ai_gepa.gepa_graph.models import ComponentValue
from pydantic_ai_gepa.gepa_graph.proposal.instruction import (
    ComponentUpdate,
    InstructionProposalOutput,
    TrajectoryAnalysis,
)
from pydantic_ai_gepa.spend import (
    COST_STOP_REASON,
    CostBudgetExceeded,
    SpendMeter,
    _pipeline_meter,
)
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
    assert _pipeline_meter.get() is None
    if helper is not optimize_parallel:
        # A refused engine projection may leave room for a fair comparison.
        # Keep the last fairly accepted incumbent when the pipeline later stops.
        accepted = [
            phase["stage"]
            for phase in result.phases
            if phase.get("adopted", phase.get("improved", False))
        ]
        assert result.best_index == (accepted[-1] if accepted else -1)
        if not accepted:
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


@contextmanager
def _registered(engine: Any) -> Iterator[str]:
    register_engine(engine.name, lambda config: engine)
    try:
        yield engine.name
    finally:
        unregister_engine(engine.name)


class _StaticEngine:
    name = "review-static"
    supports_token_cost = True  # Proposes fixed text without making model calls.

    async def run(
        self, task: OptimizationTask, config: EngineConfig, budget: BudgetTracker
    ) -> EngineResult:
        budget.spend(1)
        candidate = await task.seed_candidate()
        if config.engine_config.get("regress"):
            candidate = {
                "instructions": ComponentValue(name="instructions", text="worse")
            }
        return EngineResult(
            engine=self.name,
            best_candidate=candidate,
            best_score=None,
            num_metric_calls=1,
        )


def _scored_task(*, seed_selectable: bool = True) -> OptimizationTask:
    task = _task()

    def metric(case, output):
        override = task.agent._override_instructions.get()
        active = override.value if override is not None else task.agent._instructions
        worse = "worse" in str(active)
        return MetricResult(
            score=0.1 if worse else 0.9,
            side_info={"selectable": worse or seed_selectable},
        )

    task.metric = metric
    return task


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [None, 5.0])
async def test_sequential_keeps_fair_incumbent_when_engine_score_is_none(cap):
    with _registered(_StaticEngine()) as name:
        result = await optimize_sequential(
            _scored_task(),
            [
                EngineConfig(engine=name, max_metric_calls=1),
                EngineConfig(
                    engine=name, max_metric_calls=1, engine_config={"regress": True}
                ),
            ],
            max_metric_calls=2,
            max_token_cost=cap,
            price_fn=_price,
        )
    assert result.results[0].best_score is None
    assert result.fair_scores == [0.9, 0.1]
    assert [phase["adopted"] for phase in result.phases] == [True, False]
    assert result.best_index == 0
    assert result.best.best_candidate["instructions"].text == "seed"


@pytest.mark.asyncio
async def test_uncapped_adaptive_accepts_selectable_slice_over_unselectable_seed():
    with _registered(_StaticEngine()) as name:
        result = await optimize_adaptive_sequential(
            _scored_task(seed_selectable=False),
            [
                EngineConfig(
                    engine=name, max_metric_calls=1, engine_config={"regress": True}
                )
            ],
            max_metric_calls=1,
            price_fn=_price,
        )
    assert result.best_index == 0
    assert result.best.best_candidate["instructions"].text == "worse"
    assert result.phases[0]["improved"]
    assert result.fair_scores == [0.1]
    assert result.spend_report.max_token_cost is None


@pytest.mark.asyncio
async def test_nested_helper_inherits_outer_spend_cap_and_pricing():
    raw_task = _task()
    inner_results = []
    prices = []

    class NestedEngine:
        name = "review-nested"
        supports_token_cost = True

        async def run(self, task, config, budget):
            inner = await optimize_sequential(
                raw_task, [_config()], max_metric_calls=20
            )
            inner_results.append(inner)
            budget.spend(inner.total_metric_calls)
            return inner.best.model_copy(
                update={
                    "engine": self.name,
                    "num_metric_calls": inner.total_metric_calls,
                }
            )

    meter = SpendMeter(
        0.5,
        lambda response: prices.append(_price(response)) or _price(response),
        max_concurrent=2,
    )
    with _registered(NestedEngine()) as name:
        await optimize_parallel(
            raw_task,
            [EngineConfig(engine=name, max_metric_calls=20)],
            max_metric_calls=20,
            spend_meter=meter,
        )
    assert 0 < meter.report().total_dollars == sum(prices)
    assert meter.report().total_dollars <= 0.5 + 2 * 0.125
    assert meter.report().stopped_by_cost
    assert inner_results[0].spend_report.total_dollars == meter.report().total_dollars
    assert inner_results[0].spend_report.stopped_by_cost
    assert not inner_results[0].spend_report.unpriced_usage
    assert _pipeline_meter.get() is None


@pytest.mark.asyncio
async def test_nested_implicit_meter_checks_ancestor_engine_support():
    outer = SpendMeter(1, _price, max_concurrent=2)
    token = _pipeline_meter.set(outer)
    try:
        with pytest.raises(ValueError, match="best_of_n.*cannot meter"):
            await optimize_sequential(
                _task(),
                [
                    EngineConfig(
                        engine="best_of_n", engine_config={"propose": lambda seed: seed}
                    )
                ],
                max_metric_calls=20,
            )
        assert _pipeline_meter.get() is outer
    finally:
        _pipeline_meter.reset(token)
    assert outer.report().total_dollars == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("reuse_outer", [False, True])
async def test_nested_supplied_meter_must_descend_from_outer(reuse_outer):
    outer = SpendMeter(1, _price, max_concurrent=2)
    supplied = outer if reuse_outer else SpendMeter(price_fn=_price)
    token = _pipeline_meter.set(outer)
    try:
        with pytest.raises(ValueError, match="must descend"):
            await optimize_sequential(
                _task(), [_config(calls=1)], max_metric_calls=1, spend_meter=supplied
            )
        assert _pipeline_meter.get() is outer
    finally:
        _pipeline_meter.reset(token)
    assert supplied.report().total_dollars == 0
    assert outer.report().total_dollars == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [1, 2])
async def test_nested_supplied_descendant_accounts_into_outer(depth):
    outer = SpendMeter(5, _price, max_concurrent=2)
    supplied = outer
    for _ in range(depth):
        supplied = SpendMeter(parent=supplied)
    token = _pipeline_meter.set(outer)
    try:
        result = await optimize_sequential(
            _task(), [_config(calls=1)], max_metric_calls=1, spend_meter=supplied
        )
        assert _pipeline_meter.get() is outer
    finally:
        _pipeline_meter.reset(token)
    assert result.spend_report.total_dollars == 0.375
    assert outer.report().total_dollars == supplied.report().total_dollars == 0.375


@pytest.mark.asyncio
async def test_large_gepa_projection_stops_only_its_engine():
    small = _task()
    large = _task(cases=4)
    meter = SpendMeter(0.5, _price, max_concurrent=2)
    token = _pipeline_meter.set(meter)
    try:
        await small.evaluate(await small.seed_candidate())  # Observe validation cost.
    finally:
        _pipeline_meter.reset(token)
    refused = asyncio.Event()

    class ScheduledGepa:
        name = "review-projection"
        supports_token_cost = True

        async def run(self, task, config, budget):
            is_large = config.max_metric_calls == 4
            if not is_large:
                await asyncio.wait_for(refused.wait(), timeout=5)
            gepa_config = config.model_copy(update={"engine": "gepa"})
            result = await GepaEngine(gepa_config).run(
                large if is_large else small, gepa_config, budget
            )
            if is_large:
                refused.set()
            return result

    with _registered(ScheduledGepa()) as name:
        results = await optimize_parallel(
            small,
            [
                EngineConfig(engine=name, max_metric_calls=4, max_token_cost=100),
                EngineConfig(engine=name, max_metric_calls=1, max_token_cost=100),
            ],
            max_metric_calls=5,
            spend_meter=meter,
        )
    assert results[0].num_metric_calls == 0
    assert results[0].history[-1].data["stop_reason"] == COST_STOP_REASON
    assert results[0].history[-1].data["spend_report"]["stopped_by_cost"]
    assert results[1].num_metric_calls == 1
    assert results[1].best_score == 0.5
    assert meter.report().total_dollars == 0.25
    assert not meter.report().stopped_by_cost
    token = _pipeline_meter.set(meter)
    try:
        comparison = await small.evaluate(results[1].best_candidate)
        assert comparison.selectable
        assert meter.report().total_dollars == 0.375
        assert not meter.report().stopped_by_cost
        await small.evaluate(results[1].best_candidate)
    finally:
        _pipeline_meter.reset(token)
    with pytest.raises(CostBudgetExceeded):
        meter.check()
    assert meter.report().total_dollars == 0.5
    assert meter.report().stopped_by_cost


@pytest.mark.asyncio
async def test_projection_probe_does_not_stop_intermediate_ancestors():
    outer = SpendMeter(5, _price, max_concurrent=2)
    middle = SpendMeter(0.375, parent=outer, max_concurrent=2)
    token = _pipeline_meter.set(middle)
    try:
        task = _task()
        await task.evaluate(await task.seed_candidate())
    finally:
        _pipeline_meter.reset(token)
    child = SpendMeter(parent=middle)
    assert not child.can_start("rollout", 3, rollout_kind="validation")
    assert child.stop_reason == COST_STOP_REASON
    assert middle.stop_reason is None
    assert outer.stop_reason is None
    sibling = SpendMeter(parent=middle)
    assert sibling.can_start("rollout", 1, rollout_kind="validation")

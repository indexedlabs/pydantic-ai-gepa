"""A capped pipeline reserves enough money to finish its fair comparison.

Offline coverage (fake priced models only) for OTTO-4903: engine phases run
under an exploration meter that holds back the projected cost of the helper
work that must follow, and every helper comparison is preflighted so a
capped helper never starts a comparison it cannot afford to finish.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from pydantic_ai_gepa.compose import (
    OmniPlan,
    optimize_adaptive_sequential,
    optimize_best_of,
    optimize_omni,
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
from pydantic_ai_gepa.gepa_graph.models import ComponentValue
from pydantic_ai_gepa.spend import (
    CostBudgetExceeded,
    SpendMeter,
    _pipeline_meter,
)
from pydantic_ai_gepa.types import MetricResult

from tests.test_pipeline_spend import (
    _scored_task,
    _StaticEngine,
    _task,
    _price,
    _registered,
)


def _text_scored_task(**kwargs: Any) -> OptimizationTask:
    """A task whose metric scores 0.9 only for a "better" instructions text."""
    task = _task(**kwargs)

    def metric(case, output):
        override = task.agent._override_instructions.get()
        active = override.value if override is not None else task.agent._instructions
        return MetricResult(score=0.9 if "better" in str(active) else 0.1)

    task.metric = metric
    return task


class _RolloutEngine:
    """Priced fake engine: re-evaluates its fixed candidate until stopped.

    With a large ``max_metric_calls`` slice it would spend far more than any
    test cap, so only a meter stop can end its exploration. Each instance
    records the meter it saw and how many evaluations it completed.
    """

    supports_token_cost = True

    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.text = text
        self.meters: list[SpendMeter | None] = []
        self.evaluations = 0

    async def run(
        self, task: OptimizationTask, config: EngineConfig, budget: BudgetTracker
    ) -> EngineResult:
        self.meters.append(_pipeline_meter.get())
        candidate = {
            "instructions": ComponentValue(name="instructions", text=self.text)
        }
        calls = 0
        best_score = None
        while budget.remaining > 0:
            budget.spend(1)
            calls += 1
            try:
                score = await task.evaluate(candidate)
            except CostBudgetExceeded:
                break
            best_score = score.score
            self.evaluations += 1
        return EngineResult(
            engine=self.name,
            best_candidate=candidate,
            best_score=best_score,
            num_metric_calls=calls,
        )


@contextmanager
def _registered_engines(*engines: Any) -> Iterator[None]:
    for engine in engines:
        register_engine(engine.name, lambda config, engine=engine: engine)
    try:
        yield
    finally:
        for engine in engines:
            unregister_engine(engine.name)


def _counting_price(requests: list):
    def price(response):
        requests.append(response)
        return _price(response)

    return price


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", [optimize_best_of, optimize_vote])
async def test_capped_vote_is_funded_and_returns_the_winning_candidate(helper):
    strong = _RolloutEngine("reserve-strong", "better")
    weak = _RolloutEngine("reserve-weak", "worse")
    requests = []
    with _registered_engines(strong, weak):
        result = await helper(
            _text_scored_task(cases=1),
            [
                EngineConfig(engine=strong.name, max_metric_calls=50),
                EngineConfig(engine=weak.name, max_metric_calls=50),
            ],
            max_metric_calls=100,
            fair_vote_repetitions=1,
            max_token_cost=0.8,
            price_fn=_counting_price(requests),
        )
    # The comparison completes and the strictly better engine candidate wins.
    assert result.decision["kind"] == "instance"
    assert len(result.fair_votes) == 2
    assert result.fair_scores == [0.9, 0.1]
    assert result.best_index == 0
    assert result.best.best_candidate["instructions"].text == "better"
    # The exploration phase stopped at its reserve-adjusted local cap
    # (4 rollouts), not at the pipeline cap; the pipeline meter never stops.
    assert strong.evaluations + weak.evaluations == 4
    assert strong.evaluations < 50
    exploration = strong.meters[0]
    assert exploration is not None and exploration.stop_reason is not None
    assert not result.spend_report.stopped_by_cost
    # 4 exploration rollouts + 2 comparison rollouts, all inside the cap.
    assert len(requests) == 6
    assert result.spend_report.total_dollars == 0.75
    assert result.spend_report.total_dollars <= 0.8 + 2 * 0.125


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", [optimize_best_of, optimize_vote])
async def test_capped_vote_that_cannot_fit_starts_none_of_its_rollouts(helper):
    strong = _RolloutEngine("reserve-strong", "better")
    weak = _RolloutEngine("reserve-weak", "worse")
    requests = []
    with _registered_engines(strong, weak):
        result = await helper(
            _text_scored_task(cases=1),
            [
                EngineConfig(engine=strong.name, max_metric_calls=50),
                EngineConfig(engine=weak.name, max_metric_calls=50),
            ],
            max_metric_calls=100,
            fair_vote_repetitions=1,
            max_token_cost=0.35,
            price_fn=_counting_price(requests),
        )
    # The reserve admits only one first-observation engine rollout; the
    # comparison (2 rollouts) is then refused before it starts.
    assert len(requests) == 1
    assert result.spend_report.total_dollars == 0.125
    assert not result.spend_report.stopped_by_cost
    assert result.decision["kind"] == "cost_refused_comparison"
    assert result.decision["comparison_discarded"]
    assert result.fair_votes == []
    assert result.best_index == -1
    assert result.best.engine == "seed"
    assert result.best.best_score is None


@pytest.mark.asyncio
async def test_tiebreak_round_that_does_not_fit_stops_tiebreaking():
    requests = []
    with _registered(_StaticEngine()) as name:
        result = await optimize_best_of(
            _scored_task(),
            [
                EngineConfig(engine=name, max_metric_calls=1),
                EngineConfig(engine=name, max_metric_calls=1),
            ],
            max_metric_calls=2,
            fair_vote_repetitions=1,
            fair_vote_max_repetitions=3,
            max_token_cost=0.375,
            price_fn=_counting_price(requests),
        )
    # The base round ties; the tiebreak round is refused, and the completed
    # matched round still decides fairly (stable low-index winner).
    assert [len(vote.samples) for vote in result.fair_votes] == [1, 1]
    assert result.decision["tiebreak_stopped_for_cost"]
    assert result.decision["repetitions"] == 1
    assert result.decision["kind"] == "instance"
    assert result.best_index == 0
    assert len(requests) == 2
    assert result.spend_report.total_dollars == 0.25
    assert not result.spend_report.stopped_by_cost


@pytest.mark.asyncio
async def test_sequential_never_starts_a_stage_whose_comparison_cannot_fit():
    requests = []
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
            max_token_cost=0.35,
            price_fn=_counting_price(requests),
        )
    # Stage 0 is compared and adopted; stage 1's engine never starts because
    # its comparison could not be funded to completion.
    assert len(result.results) == 1
    assert result.phases[-1]["skipped"] == "comparison_refused_cost"
    assert result.decision["cost_refused_comparison"]
    assert result.best_index == 0
    assert result.fair_scores == [0.9]
    assert len(requests) == 2  # Seed evaluation + the one funded comparison.
    assert result.spend_report.total_dollars == 0.25
    assert not result.spend_report.stopped_by_cost


@pytest.mark.asyncio
async def test_sequential_seed_evaluation_is_preflighted():
    requests = []
    meter = SpendMeter(0.2, _counting_price(requests), max_concurrent=2)
    task = _scored_task()
    token = _pipeline_meter.set(meter)
    try:
        # Observe the validation rollout cost before the helper starts.
        await task.evaluate(await task.seed_candidate())
    finally:
        _pipeline_meter.reset(token)
    assert len(requests) == 1
    with _registered(_StaticEngine()) as name:
        result = await optimize_sequential(
            task,
            [EngineConfig(engine=name, max_metric_calls=1)],
            max_metric_calls=1,
            spend_meter=meter,
        )
    assert len(requests) == 1  # No further rollout was ever started.
    assert result.results == []
    assert result.best.engine == "seed"
    assert result.best.best_score is None
    assert result.decision["cost_refused_comparison"]
    assert meter.report().total_dollars == 0.125
    assert not meter.report().stopped_by_cost


@pytest.mark.asyncio
async def test_adaptive_cycle_does_not_reevaluate_an_unfundable_incumbent():
    requests = []
    with _registered(_StaticEngine()) as name:
        result = await optimize_adaptive_sequential(
            _scored_task(),
            [EngineConfig(engine=name, max_metric_calls=1)],
            max_metric_calls=5,
            cycle=True,
            max_slices=5,
            max_token_cost=0.35,
            price_fn=_counting_price(requests),
        )
    # One funded slice comparison, then the next slice is refused instead of
    # re-evaluating the same incumbent until slice_limit or the cap.
    assert len(result.results) == 1
    assert result.decision["cost_refused_comparison"]
    assert len(requests) == 2  # Seed evaluation + one funded comparison.
    assert result.best.engine == "seed"
    assert result.best.best_score == 0.9
    assert result.spend_report.total_dollars == 0.25
    assert not result.spend_report.stopped_by_cost


@pytest.mark.asyncio
async def test_capped_omni_funds_its_vote_and_refuses_the_rest():
    strong = _RolloutEngine("reserve-strong", "better")
    weak = _RolloutEngine("reserve-weak", "worse")
    continuation = _RolloutEngine("reserve-continuation", "continuation")
    requests = []
    plan = OmniPlan(
        phase_one=[
            EngineConfig(engine=strong.name, max_metric_calls=50),
            EngineConfig(engine=weak.name, max_metric_calls=50),
        ],
        phase_two=EngineConfig(engine=continuation.name, max_metric_calls=50),
        phase_one_metric_calls=100,
        phase_two_metric_calls=50,
        fair_vote_repetitions=2,
        fair_vote_max_repetitions=2,
    )
    with _registered_engines(strong, weak, continuation):
        result = await optimize_omni(
            _text_scored_task(cases=1),
            plan,
            max_token_cost=1.55,
            price_fn=_counting_price(requests),
        )
    # The phase-one vote completes and the strictly better candidate wins.
    assert result.decision["kind"] == "instance"
    assert len(result.fair_votes) == 3  # Seed baseline + two engines.
    assert result.best_index == 1
    assert result.best.best_candidate["instructions"].text == "better"
    assert result.decision["phase_two_seeded_from"] == 1
    # The remaining headroom funds neither the continuation comparison nor
    # the test report; both are refused before starting, keeping the fairly
    # compared phase-one winner.
    assert continuation.evaluations == 0
    assert result.decision["continuation_vote"]["kind"] == "cost_refused_comparison"
    assert not result.decision["phase_two_adopted"]
    assert result.test_score is None
    # 6 exploration rollouts + 6 vote rollouts; nothing after the refusal.
    assert len(requests) == 12
    assert result.spend_report.total_dollars == 1.5
    assert result.spend_report.total_dollars <= 1.55 + 2 * 0.125
    exploration = strong.meters[0]
    assert exploration is not None and exploration.stop_reason is not None
    assert not result.spend_report.stopped_by_cost


def _high_recurring_price(requests: list):
    """First rollout $0.125, every later rollout $0.25 (the high recurs)."""

    def price(response):
        requests.append(response)
        return 0.125 if len(requests) == 1 else 0.25

    return price


@pytest.mark.asyncio
async def test_comparison_is_refused_when_only_the_mean_projection_would_fit():
    # The seed evaluation observes validation rollouts of $0.125 and $0.25
    # (mean $0.1875, highest $0.25). A two-rollout stage comparison projects
    # $0.4375 at (N - 1) x mean + highest but $0.50 at the admission bound
    # N x highest; headroom is $0.45. Only the admission bound is safe: the
    # first comparison rollout recurs at the observed $0.25 high, and the
    # second would be refused with nothing in flight, stopping the pipeline
    # meter and discarding the whole comparison.
    requests = []
    with _registered(_StaticEngine()) as name:
        result = await optimize_sequential(
            _task(cases=2),
            [EngineConfig(engine=name, max_metric_calls=1)],
            max_metric_calls=1,
            max_token_cost=0.825,
            price_fn=_high_recurring_price(requests),
        )
    # Refused before the stage engine or any comparison rollout starts.
    assert len(requests) == 2  # The seed evaluation only.
    assert result.results == []
    assert result.phases == [
        {"stage": 0, "engine": name, "skipped": "comparison_refused_cost"}
    ]
    assert result.decision["cost_refused_comparison"]
    assert result.best.engine == "seed"
    assert result.best.best_score == 0.5
    assert result.spend_report.total_dollars == 0.375
    assert not result.spend_report.stopped_by_cost


@pytest.mark.asyncio
async def test_comparison_runs_when_headroom_covers_the_admission_bound():
    # Same observed costs, but headroom $0.55 covers 2 x $0.25: the
    # comparison completes even though the high recurs on both rollouts.
    requests = []
    with _registered(_StaticEngine()) as name:
        result = await optimize_sequential(
            _task(cases=2),
            [EngineConfig(engine=name, max_metric_calls=1)],
            max_metric_calls=1,
            max_token_cost=0.925,
            price_fn=_high_recurring_price(requests),
        )
    assert len(requests) == 4  # Seed evaluation + both comparison rollouts.
    assert len(result.results) == 1
    assert result.fair_scores == [0.5]
    assert result.best_index == 0
    assert "cost_refused_comparison" not in result.decision
    assert result.spend_report.total_dollars == 0.875
    assert not result.spend_report.stopped_by_cost


@pytest.mark.asyncio
async def test_uncapped_best_of_has_no_exploration_meter():
    strong = _RolloutEngine("reserve-strong", "better")
    weak = _RolloutEngine("reserve-weak", "worse")
    with _registered_engines(strong, weak):
        result = await optimize_best_of(
            _text_scored_task(cases=1),
            [
                EngineConfig(engine=strong.name, max_metric_calls=3),
                EngineConfig(engine=weak.name, max_metric_calls=3),
            ],
            max_metric_calls=6,
            fair_vote_repetitions=1,
            price_fn=_price,
        )
    # Uncapped pipelines keep one shared meter: no exploration child, no
    # change in concurrency or results.
    assert strong.meters[0] is not None
    assert strong.meters[0] is weak.meters[0]
    assert strong.meters[0].max_token_cost is None
    assert strong.evaluations == 3
    assert weak.evaluations == 3
    assert result.decision["kind"] == "instance"
    assert result.fair_scores == [0.9, 0.1]
    assert result.best_index == 0
    assert not result.spend_report.stopped_by_cost

"""Deterministic coverage for the first-class Omni composition contract."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa.compose import OmniPlan, optimize_omni
from pydantic_ai_gepa.engines import (
    BudgetTracker,
    EngineConfig,
    EngineResult,
    OptimizationTask,
    register_engine,
    unregister_engine,
)
from pydantic_ai_gepa.gepa_graph.models import CandidateMap, ComponentValue
from pydantic_ai_gepa.types import MetricResult, RolloutOutput


def _candidate(text: str) -> CandidateMap:
    return {"instructions": ComponentValue(name="instructions", text=text)}


class _OmniFake:
    name = "omni_fake"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    async def run(
        self, task: OptimizationTask, config: EngineConfig, budget: BudgetTracker
    ) -> EngineResult:
        seen = config.engine_config.get("seen")
        if isinstance(seen, list):
            seen.append((await task.seed_candidate())["instructions"].text)
        spend = int(config.engine_config.get("spend", 0))
        budget.spend(spend)
        return EngineResult(
            engine=self.name,
            best_candidate=config.engine_config["candidate"],
            best_score=0,
            num_metric_calls=spend,
        )


class _TestSetProbe:
    name = "test_set_probe"

    def __init__(self, config: EngineConfig) -> None:
        pass

    async def run(
        self, task: OptimizationTask, config: EngineConfig, budget: BudgetTracker
    ) -> EngineResult:
        assert task.test_set is None
        with pytest.raises(PermissionError):
            await task.evaluate(await task.seed_candidate(), dataset="test")
        return EngineResult(
            engine=self.name,
            best_candidate=await task.seed_candidate(),
            best_score=0,
            num_metric_calls=0,
        )


@pytest.fixture(autouse=True)
def _registry() -> Generator[None, None, None]:
    register_engine("omni_fake", _OmniFake, replace=True)
    register_engine("omni_fake_2", _OmniFake, replace=True)
    register_engine("test_set_probe", _TestSetProbe, replace=True)
    yield
    unregister_engine("omni_fake")
    unregister_engine("omni_fake_2")
    unregister_engine("test_set_probe")


def _task(*, test_set: bool = False) -> OptimizationTask:
    agent = Agent(TestModel(custom_output_text="ok"), instructions="seed")
    cases = [Case(name="a", inputs="x", expected_output="ok")]

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        active = agent._override_instructions.get()
        text = str(active.value if active else agent._instructions)
        return MetricResult(
            score=float("good" in text),
            side_info={"scores": {"reliability": float("good" in text)}},
        )

    return OptimizationTask(
        agent=agent,
        trainset=cases,
        valset=cases,
        test_set=cases if test_set else None,
        metric=metric,
    )


def _objective_task() -> OptimizationTask:
    agent = Agent(TestModel(custom_output_text="ok"), instructions="seed")
    cases = [Case(name="a", inputs="x", expected_output="ok")]
    values = {
        "seed": (0.1, 0.0),
        "good": (0.5, 1.0),
        "continuation": (0.6, 0.0),
        "better": (0.55, 2.0),
    }

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        override = agent._override_instructions.get()
        text = str(override.value if override else agent._instructions)
        key = next(name for name in values if name in text)
        score, objective = values[key]
        return MetricResult(score=score, side_info={"scores": {"quality": objective}})

    return OptimizationTask(agent=agent, trainset=cases, valset=cases, metric=metric)


def _noisy_task(values: dict[str, list[float]]) -> OptimizationTask:
    """Return deterministic stochastic samples keyed by the active candidate."""
    agent = Agent(TestModel(custom_output_text="ok"), instructions="seed")
    case = Case(name="a", inputs="x", expected_output="ok")
    samples = {name: iter(scores) for name, scores in values.items()}

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        del case, output
        override = agent._override_instructions.get()
        text = str(override.value if override else agent._instructions)
        key = next(name for name in samples if name in text)
        return MetricResult(score=next(samples[key]))

    return OptimizationTask(agent=agent, trainset=[case], valset=[case], metric=metric)


@pytest.mark.asyncio
async def test_omni_uses_matched_slices_repeated_vote_and_fresh_phase_two_seed() -> (
    None
):
    seen: list[str] = []
    plan = OmniPlan(
        phase_one=[
            EngineConfig(
                engine="omni_fake_2",
                max_metric_calls=2,
                engine_config={"candidate": _candidate("bad"), "spend": 1},
            ),
            EngineConfig(
                engine="omni_fake",
                max_metric_calls=2,
                engine_config={"candidate": _candidate("good"), "spend": 1},
            ),
        ],
        phase_two=EngineConfig(
            engine="omni_fake",
            max_metric_calls=2,
            engine_config={
                "candidate": _candidate("good final"),
                "spend": 1,
                "seen": seen,
            },
        ),
        phase_one_metric_calls=4,
        phase_two_metric_calls=2,
        fair_vote_repetitions=3,
        fair_vote_max_repetitions=3,
    )

    result = await optimize_omni(_task(), plan)

    assert [len(vote.samples) for vote in result.fair_votes] == [3, 3, 3]
    assert seen == ["good"]
    # Exact ties retain the incumbent, consistent with the outer controller's
    # deterministic scalar/identifier tie-break.
    assert result.decision["phase_two_adopted"] is False
    assert result.total_metric_calls == 3


@pytest.mark.asyncio
async def test_omni_keeps_seed_when_phase_two_regresses_and_reports_held_out_test() -> (
    None
):
    plan = OmniPlan(
        phase_one=[
            EngineConfig(
                engine="omni_fake",
                max_metric_calls=1,
                engine_config={"candidate": _candidate("good"), "spend": 0},
            )
        ],
        phase_two=EngineConfig(
            engine="omni_fake",
            max_metric_calls=1,
            engine_config={"candidate": _candidate("bad"), "spend": 0},
        ),
        phase_one_metric_calls=1,
        phase_two_metric_calls=1,
        fair_vote_repetitions=1,
        fair_vote_max_repetitions=1,
    )
    result = await optimize_omni(_task(test_set=True), plan)
    assert result.best.best_candidate["instructions"].text == "good"
    assert result.test_score == 1.0


@pytest.mark.asyncio
async def test_omni_hides_held_out_test_set_from_engine() -> None:
    plan = OmniPlan(
        phase_one=[EngineConfig(engine="test_set_probe", max_metric_calls=1)],
        phase_two=EngineConfig(
            engine="omni_fake",
            max_metric_calls=1,
            engine_config={"candidate": _candidate("good")},
        ),
        phase_one_metric_calls=1,
        phase_two_metric_calls=1,
        fair_vote_repetitions=1,
        fair_vote_max_repetitions=1,
    )
    await optimize_omni(_task(test_set=True), plan)


@pytest.mark.asyncio
async def test_omni_preflights_comparison_before_phase_one_engine_work() -> None:
    seen: list[str] = []
    plan = OmniPlan(
        phase_one=[
            EngineConfig(
                engine="omni_fake",
                max_metric_calls=1,
                engine_config={"candidate": _candidate("good"), "seen": seen},
            )
        ],
        phase_two=EngineConfig(
            engine="omni_fake",
            max_metric_calls=1,
            engine_config={"candidate": _candidate("good")},
        ),
        phase_one_metric_calls=1,
        phase_two_metric_calls=1,
        comparison_metric_calls=1,
        continuation_comparison_metric_calls=1,
        fair_vote_repetitions=1,
        fair_vote_max_repetitions=1,
    )
    with pytest.raises(ValueError, match="cannot fund"):
        await optimize_omni(_task(), plan)
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("continuation", "expected", "adopted"),
    [("continuation", "good", False), ("better", "better", True)],
)
async def test_omni_phase_two_obeys_objective_frontier_not_scalar_only(
    continuation: str, expected: str, adopted: bool
) -> None:
    plan = OmniPlan(
        phase_one=[
            EngineConfig(
                engine="omni_fake",
                max_metric_calls=1,
                engine_config={"candidate": _candidate("good")},
            )
        ],
        phase_two=EngineConfig(
            engine="omni_fake",
            max_metric_calls=1,
            engine_config={"candidate": _candidate(continuation)},
        ),
        phase_one_metric_calls=1,
        phase_two_metric_calls=1,
        fair_vote_repetitions=1,
        fair_vote_max_repetitions=1,
        selection_mode="objective",
    )
    result = await optimize_omni(_objective_task(), plan)
    assert result.best.best_candidate["instructions"].text == expected
    assert result.decision["phase_two_adopted"] is adopted


@pytest.mark.asyncio
async def test_omni_keeps_seed_after_inconclusive_noisy_phase_one_vote() -> None:
    plan = OmniPlan(
        phase_one=[
            EngineConfig(
                engine="omni_fake",
                max_metric_calls=1,
                engine_config={"candidate": _candidate("challenger")},
            )
        ],
        phase_two=EngineConfig(
            engine="omni_fake",
            max_metric_calls=1,
            engine_config={"candidate": _candidate("phase-two")},
        ),
        phase_one_metric_calls=1,
        phase_two_metric_calls=1,
        fair_vote_repetitions=2,
        fair_vote_max_repetitions=3,
    )
    task = _noisy_task(
        {
            "seed": [0.4, 0.6, 0.5, 0.5, 0.5],
            "challenger": [0.45, 0.65, 0.55],
            "phase-two": [0.5, 0.5],
        }
    )

    result = await optimize_omni(task, plan)

    assert result.decision["acceptance"]["verdict"] == "inconclusive"
    assert result.decision["repetitions"] == 3
    assert result.decision["phase_two_seeded_from"] == 0


@pytest.mark.asyncio
async def test_omni_keeps_incumbent_after_inconclusive_noisy_phase_two_vote() -> None:
    plan = OmniPlan(
        phase_one=[
            EngineConfig(
                engine="omni_fake",
                max_metric_calls=1,
                engine_config={"candidate": _candidate("good")},
            )
        ],
        phase_two=EngineConfig(
            engine="omni_fake",
            max_metric_calls=1,
            engine_config={"candidate": _candidate("continuation")},
        ),
        phase_one_metric_calls=1,
        phase_two_metric_calls=1,
        fair_vote_repetitions=2,
        fair_vote_max_repetitions=2,
    )
    task = _noisy_task(
        {
            "seed": [0.0, 0.0],
            "good": [0.8, 1.0, 0.8, 1.0],
            "continuation": [0.85, 1.05],
        }
    )

    result = await optimize_omni(task, plan)

    assert result.best.best_candidate["instructions"].text == "good"
    assert (
        result.decision["continuation_vote"]["acceptance"]["verdict"] == "inconclusive"
    )


@pytest.mark.asyncio
async def test_omni_fails_closed_when_the_frozen_baseline_is_not_selectable() -> None:
    agent = Agent(TestModel(custom_output_text="ok"), instructions="seed")
    case = Case(name="a", inputs="x", expected_output="ok")

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        del case, output
        active = agent._override_instructions.get()
        text = str(active.value if active else agent._instructions)
        return MetricResult(
            score=float("good" in text),
            side_info={"selectable": "seed" not in text},
        )

    task = OptimizationTask(agent=agent, trainset=[case], valset=[case], metric=metric)
    plan = OmniPlan(
        phase_one=[
            EngineConfig(
                engine="omni_fake",
                max_metric_calls=1,
                engine_config={"candidate": _candidate("good")},
            )
        ],
        phase_two=EngineConfig(
            engine="omni_fake",
            max_metric_calls=1,
            engine_config={"candidate": _candidate("good")},
        ),
        phase_one_metric_calls=1,
        phase_two_metric_calls=1,
        fair_vote_repetitions=1,
        fair_vote_max_repetitions=1,
    )

    with pytest.raises(ValueError, match="baseline must be selectable"):
        await optimize_omni(task, plan)


def test_omni_rejects_unmatched_slices_before_work() -> None:
    with pytest.raises(ValueError, match="matched"):
        OmniPlan(
            phase_one=[
                EngineConfig(engine="omni_fake", max_metric_calls=1),
                EngineConfig(engine="omni_fake_2", max_metric_calls=2),
            ],
            phase_two=EngineConfig(engine="omni_fake", max_metric_calls=1),
            phase_one_metric_calls=3,
            phase_two_metric_calls=1,
        )

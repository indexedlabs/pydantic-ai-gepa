"""Graph stops preserve full-validation winners at every budget boundary."""

from pathlib import Path
from typing import Any, cast

import pytest

from pydantic_ai_gepa.gepa_graph import create_deps, create_gepa_graph
from pydantic_ai_gepa.gepa_graph.datasets import ListDataLoader
from pydantic_ai_gepa.gepa_graph.models import GepaConfig, GepaState
from pydantic_ai_gepa.types import ReflectionConfig
from tests.gepa_graph.utils import (
    AdapterStub,
    ProposalGeneratorStub,
    make_adapter_stub,
    make_dataset,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("budget", "spent", "best_score", "candidate_count"),
    [
        (2, 0, None, 1),
        (3, 3, 0.4, 1),
        (4, 3, 0.4, 1),
        (6, 5, 0.4, 1),
        (8, 7, 0.4, 2),
        (10, 10, 0.85, 2),
    ],
)
async def test_budget_stops_before_each_unaffordable_batch(
    budget: int,
    spent: int,
    best_score: float | None,
    candidate_count: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    observed_calls = 0
    original_evaluate = AdapterStub.evaluate

    async def counted_evaluate(self, batch, *args, **kwargs):
        nonlocal observed_calls
        observed_calls += len(batch)
        assert observed_calls <= budget
        return await original_evaluate(self, batch, *args, **kwargs)

    monkeypatch.setattr(AdapterStub, "evaluate", counted_evaluate)
    config = GepaConfig(
        max_evaluations=budget,
        minibatch_size=2,
        reflection_config=ReflectionConfig(model="stub"),
    )
    deps = create_deps(make_adapter_stub(), config)
    deps.proposal_generator = cast(Any, ProposalGeneratorStub())
    state = GepaState(
        config=config,
        training_set=ListDataLoader(make_dataset(2)),
        validation_set=ListDataLoader(make_dataset(3)),
    )

    result = await create_gepa_graph(config=config).run(state=state, deps=deps)

    assert result.stopped
    assert result.stop_reason.startswith("Max evaluations reached")
    assert result.total_evaluations == observed_calls == spent
    if best_score is None:
        assert result.best_score is None
    else:
        assert result.best_score == pytest.approx(best_score)
    assert len(result.candidates) == candidate_count
    assert result.best_candidate is not None
    if spent == 0:
        assert "cannot cover the validation set" in result.stop_reason
        assert result.best_candidate == result.original_candidate
        assert result.best_candidate.validation_scores == {}
        assert result.full_validations == 0
    else:
        assert len(result.best_candidate.validation_scores) == 3
        assert result.best_candidate_idx == (1 if best_score == 0.85 else 0)
    if budget == 8:
        assert result.candidates[-1].validation_scores == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(("target", "spent", "iterations"), [(0.4, 3, 0), (0.8, 10, 1)])
async def test_target_stops_on_seed_or_improved_validation(
    target: float,
    spent: int,
    iterations: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = GepaConfig(
        max_evaluations=100,
        max_iterations=10,
        minibatch_size=2,
        stop_at_score=target,
        reflection_config=ReflectionConfig(model="stub"),
    )
    deps = create_deps(make_adapter_stub(), config)
    deps.proposal_generator = cast(Any, ProposalGeneratorStub())
    state = GepaState(
        config=config,
        training_set=ListDataLoader(make_dataset(2)),
        validation_set=ListDataLoader(make_dataset(3)),
    )

    result = await create_gepa_graph(config=config).run(state=state, deps=deps)

    assert result.stop_reason == "Target score reached"
    assert result.best_score >= target
    assert result.total_evaluations == spent
    assert result.iterations == iterations < config.max_iterations


@pytest.mark.asyncio
async def test_perfect_training_score_does_not_satisfy_validation_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_evaluate = AdapterStub.evaluate

    async def perfect_training(self, batch, candidate, capture_traces, **kwargs):
        result = await original_evaluate(
            self, batch, candidate, capture_traces, **kwargs
        )
        if capture_traces:
            result.scores = [1.0] * len(batch)
        return result

    monkeypatch.setattr(AdapterStub, "evaluate", perfect_training)
    config = GepaConfig(
        max_evaluations=100, max_iterations=2, minibatch_size=2, stop_at_score=0.8
    )
    deps = create_deps(make_adapter_stub(), config)
    state = GepaState(
        config=config,
        training_set=ListDataLoader(make_dataset(2)),
        validation_set=ListDataLoader(make_dataset(3)),
    )

    result = await create_gepa_graph(config=config).run(state=state, deps=deps)

    assert result.stop_reason == "Max iterations reached"
    assert result.best_score == pytest.approx(0.4)
    assert result.total_evaluations == 7
    assert result.iterations == 2

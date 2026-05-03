"""Tests for helper utilities in the GEPA graph package."""

from __future__ import annotations

from typing import Sequence, cast
import random

import pytest

from pydantic_evals import Case

from pydantic_ai_gepa.adapter import Adapter, SharedReflectiveDataset
from pydantic_ai_gepa.gepa_graph import create_deps
from pydantic_ai_gepa.gepa_graph.deps import GepaDeps
from pydantic_ai_gepa.gepa_graph.models import (
    CandidateMap,
    CandidateSelectorStrategy,
    ComponentValue,
    GepaConfig,
    GepaState,
)
from pydantic_ai_gepa.gepa_graph.datasets import ListDataLoader
from pydantic_ai_gepa.gepa_graph.selectors import (
    AllComponentSelector,
    BatchSampler,
    CurrentBestCandidateSelector,
    ParetoCandidateSelector,
    ReflectionComponentSelector,
    RoundRobinComponentSelector,
)


class _AgentStub:
    def __init__(self) -> None:
        self._instructions = "seed"


class _AdapterStub:
    def __init__(self) -> None:
        self.agent = _AgentStub()
        self.input_spec = None

    async def evaluate(self, batch, candidate, capture_traces):  # pragma: no cover
        raise RuntimeError("evaluate should not be called in helper tests")

    def make_reflective_dataset(
        self,
        *,
        candidate,
        eval_batch,
        components_to_update: Sequence[str],
        include_case_metadata: bool = False,
        include_expected_output: bool = False,
    ) -> SharedReflectiveDataset:  # pragma: no cover
        return SharedReflectiveDataset(records=[])

    def get_components(self) -> CandidateMap:  # pragma: no cover
        return {"instructions": ComponentValue(name="instructions", text="seed")}


def _make_adapter() -> Adapter[str, str, dict[str, str]]:
    return cast(Adapter[str, str, dict[str, str]], _AdapterStub())


def _make_state(config: GepaConfig) -> GepaState:
    dataset = [
        Case(name=f"case-{idx}", inputs=f"prompt-{idx}", metadata={"label": "stub"})
        for idx in range(3)
    ]
    return GepaState(
        config=config,
        training_set=ListDataLoader(dataset),
        validation_set=ListDataLoader(dataset),
    )


def test_create_deps_defaults() -> None:
    adapter = _make_adapter()
    from pydantic_ai_gepa.types import ReflectionConfig

    config = GepaConfig(
        seed=7,
        reflection_config=ReflectionConfig(
            model="reflection-model",
            max_spawned_agents=2,
        ),
    )

    deps = create_deps(adapter, config)

    assert isinstance(deps, GepaDeps)
    assert deps.adapter is adapter
    assert isinstance(deps.candidate_selector, ParetoCandidateSelector)
    assert isinstance(deps.component_selector, RoundRobinComponentSelector)
    assert isinstance(deps.batch_sampler, BatchSampler)
    assert isinstance(deps.merge_builder.seed, int)
    assert deps.merge_builder.seed == config.seed
    assert config.reflection_config is not None
    assert deps.model == config.reflection_config.model
    assert getattr(deps.proposal_generator, "_max_spawned_agents") == 2

    # Batch sampler should respect the config seed for determinism.
    sampler_rng = getattr(deps.batch_sampler, "_rng")
    assert abs(sampler_rng.random() - random.Random(config.seed).random()) < 1e-9


@pytest.mark.asyncio
async def test_create_deps_supports_alternate_selectors() -> None:
    adapter = _make_adapter()
    config = GepaConfig(
        candidate_selector=CandidateSelectorStrategy.CURRENT_BEST,
        component_selector="all",
    )

    deps = create_deps(adapter, config)

    assert isinstance(deps.candidate_selector, CurrentBestCandidateSelector)
    assert isinstance(deps.component_selector, AllComponentSelector)

    # Ensure helper returns usable dependencies for sampling.
    state = _make_state(config)
    batch = await deps.batch_sampler.sample(state.training_set, state, size=2)
    assert len(batch) == 2


def test_create_deps_supports_agent_component_selector() -> None:
    from pydantic_ai_gepa.types import ReflectionConfig

    adapter = _make_adapter()
    config = GepaConfig(
        component_selector="reflection",
        reflection_config=ReflectionConfig(model="reflection-model"),
    )

    deps = create_deps(adapter, config)
    assert isinstance(deps.component_selector, ReflectionComponentSelector)

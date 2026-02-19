"""Tests for the ReflectStep implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pytest
from pydantic_graph.beta import StepContext
from pydantic_evals import Case

from pydantic_ai_gepa.adapter import (
    Adapter,
    ComponentReflectiveDataset,
    ReflectiveDataset,
    SharedReflectiveDataset,
)

from pydantic_ai_gepa.gepa_graph.datasets import ListDataLoader
from pydantic_ai_gepa.gepa_graph.deps import GepaDeps
from pydantic_ai_gepa.gepa_graph.evaluation import (
    EvaluationResults,
    ParallelEvaluator,
    ParetoFrontManager,
)
from pydantic_ai_gepa.gepa_graph.models import (
    CandidateMap,
    CandidateProgram,
    ComponentValue,
    GepaConfig,
    GepaState,
)
from pydantic_ai_gepa.gepa_graph.steps import reflect_step
from pydantic_ai_gepa.gepa_graph.proposal import (
    InstructionProposalGenerator,
    MergeProposalBuilder,
    ProposalResult,
)
from pydantic_ai_gepa.gepa_graph.selectors import (
    BatchSampler,
    CurrentBestCandidateSelector,
    AllComponentSelector,
    RoundRobinComponentSelector,
)
from pydantic_ai_gepa.types import RolloutOutput


def _make_data(case_id: str) -> Case[str, str, dict[str, str]]:
    return Case(name=case_id, inputs=f"prompt-{case_id}", metadata={})


def _make_state(
    *,
    minibatch_size: int = 2,
    config: GepaConfig | None = None,
) -> GepaState:
    if config is None:
        config = GepaConfig(
            max_evaluations=100,
            minibatch_size=minibatch_size,
            merges_per_accept=2,
            perfect_score=1.0,
            skip_perfect_score=True,
        )
    training = [_make_data("a"), _make_data("b")]
    loader = ListDataLoader(training)
    state = GepaState(
        config=config,
        training_set=loader,
        validation_set=ListDataLoader(training),
    )
    seed = CandidateProgram(
        idx=0,
        components={
            "instructions": ComponentValue(
                name="instructions", text="Seed instructions"
            ),
        },
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )
    state.add_candidate(seed, auto_assign_idx=False)
    state.iteration = 1
    return state


async def _training_examples(state: GepaState) -> list[Case[str, str, dict[str, str]]]:
    loader = state.training_set
    ids = await loader.all_ids()
    return cast(list[Case[str, str, dict[str, str]]], await loader.fetch(ids))


@dataclass
class _StubTrajectory:
    instructions: str | None = "seed"
    metric_feedback: str | None = "feedback"
    final_output: Any | None = "result"

    def to_reflective_record(self) -> dict[str, Any]:
        return {"messages": [], "user_prompt": "stub"}


def _eval_results(scores: list[float]) -> EvaluationResults[str]:
    outputs = [
        RolloutOutput.from_success(f"result-{idx}") for idx, _ in enumerate(scores)
    ]
    case_ids = [f"case-{idx}" for idx in range(len(scores))]
    return EvaluationResults(
        data_ids=case_ids,
        scores=list(scores),
        outputs=outputs,
        trajectories=[_StubTrajectory() for _ in scores],
    )


class _StubAdapter:
    def __init__(
        self,
        *,
        dataset: dict[str, list[dict[str, str]]] | None = None,
    ) -> None:
        self.agent = type("Agent", (), {"_instructions": "seed"})()
        self.input_spec = None
        self._dataset = dataset or {"instructions": [{"feedback": "needs work"}]}
        self.dataset_calls = 0

    async def evaluate(self, batch, candidate, capture_traces):  # pragma: no cover
        raise RuntimeError("evaluate should not be called in ReflectStep tests")

    def make_reflective_dataset(
        self,
        *,
        candidate: CandidateMap,
        eval_batch: Any,
        components_to_update: Sequence[str],
        include_case_metadata: bool = False,
        include_expected_output: bool = False,
    ) -> ComponentReflectiveDataset:
        self.dataset_calls += 1
        data = {
            component: list(self._dataset.get(component, self._dataset["instructions"]))
            for component in components_to_update
        }
        return ComponentReflectiveDataset(records_by_component=data)

    def get_components(self) -> CandidateMap:
        return {"instructions": ComponentValue(name="instructions", text="seed")}


class _StubBatchSampler(BatchSampler):
    def __init__(self, batch):
        super().__init__(seed=0)
        self._batch = list(batch)
        self.calls = 0

    async def sample(self, training_set, state, size):
        self.calls += 1
        return list(self._batch)


class _StubProposalGenerator(InstructionProposalGenerator):
    def __init__(
        self,
        updates: dict[str, str],
        *,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self._updates = updates
        self._metadata = metadata or {}
        self.calls = 0
        self.last_reflective_data: ReflectiveDataset | None = None

    async def propose_texts(
        self,
        *,
        candidate: CandidateProgram,
        reflective_data: ReflectiveDataset,
        components: Sequence[str],
        model: Any,
        iteration: int | None = None,
        current_best_score: float | None = None,
        parent_score: float | None = None,
        model_settings: Any = None,
        example_bank: Any = None,
    ) -> ProposalResult:
        self.calls += 1
        self.last_reflective_data = reflective_data
        updates: dict[str, str] = {
            component: self._updates.get(
                component, candidate.components[component].text
            )
            for component in components
        }
        component_metadata = {
            component: self._metadata[component]
            for component in components
            if component in self._metadata
        }
        return ProposalResult(
            texts=updates,
            component_metadata=component_metadata,
            reasoning=None,
        )


class _StubEvaluator(ParallelEvaluator):
    def __init__(self, results: list[EvaluationResults[str]]) -> None:
        self._results = list(results)
        self.calls = 0

    async def evaluate_batch(self, **kwargs):
        if not self._results:
            raise AssertionError("No more evaluation results scheduled.")
        self.calls += 1
        return self._results.pop(0)


def _make_deps(
    *,
    adapter: Adapter[str, str, dict[str, str]],
    evaluator: ParallelEvaluator,
    batch_sampler: BatchSampler,
    proposal_generator: InstructionProposalGenerator,
    model: str | None = "reflection-model",
) -> GepaDeps:
    return GepaDeps(
        adapter=adapter,
        evaluator=evaluator,
        pareto_manager=ParetoFrontManager(),
        candidate_selector=CurrentBestCandidateSelector(),
        component_selector=RoundRobinComponentSelector(),
        batch_sampler=batch_sampler,
        proposal_generator=proposal_generator,
        merge_builder=MergeProposalBuilder(),
        model=model,
    )


def _ctx(
    state: GepaState,
    deps: GepaDeps,
) -> StepContext[GepaState, GepaDeps, None]:
    return StepContext(state=state, deps=deps, inputs=None)


@pytest.mark.asyncio
async def test_reflect_step_accepts_strict_improvement() -> None:
    state = _make_state()
    minibatch = await _training_examples(state)
    evaluator = _StubEvaluator([_eval_results([0.4, 0.5]), _eval_results([0.6, 0.7])])
    batch_sampler = _StubBatchSampler(minibatch)
    stub_adapter = _StubAdapter()
    adapter = cast(Adapter[str, str, dict[str, str]], stub_adapter)
    generator = _StubProposalGenerator({"instructions": "Improved text"})
    deps = _make_deps(
        adapter=adapter,
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
    )
    ctx = _ctx(state, deps)

    result = await reflect_step(ctx)

    assert result == "evaluate"
    assert state.last_accepted is True
    assert len(state.candidates) == 2
    new_candidate = state.candidates[-1]
    assert new_candidate.components["instructions"].text == "Improved text"
    assert new_candidate.components["instructions"].version == 1
    assert new_candidate.parent_indices == [0]
    assert new_candidate.minibatch_scores == [0.6, 0.7]
    assert state.merge_scheduled == state.config.merges_per_accept
    assert state.total_evaluations == 4
    assert stub_adapter.dataset_calls == 1
    assert generator.calls == 1
    assert evaluator.calls == 2
    assert batch_sampler.calls == 1


@pytest.mark.asyncio
async def test_reflect_step_tracks_hypothesis_metadata_when_enabled() -> None:
    config = GepaConfig(
        max_evaluations=100,
        minibatch_size=2,
        merges_per_accept=1,
        perfect_score=1.0,
        skip_perfect_score=True,
        track_component_hypotheses=True,
    )
    state = _make_state(config=config)
    minibatch = await _training_examples(state)
    evaluator = _StubEvaluator([_eval_results([0.4, 0.6]), _eval_results([0.7, 0.8])])
    batch_sampler = _StubBatchSampler(minibatch)
    stub_adapter = _StubAdapter()
    adapter = cast(Adapter[str, str, dict[str, str]], stub_adapter)
    generator = _StubProposalGenerator(
        {"instructions": "Improved text"},
        metadata={"instructions": {"hypothesis": "Fix boundary errors"}},
    )
    deps = _make_deps(
        adapter=adapter,
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
    )
    ctx = _ctx(state, deps)

    await reflect_step(ctx)

    assert len(state.candidates) == 2
    new_metadata = state.candidates[-1].components["instructions"].metadata
    assert new_metadata is not None
    assert new_metadata["hypothesis"] == "Fix boundary errors"
    assert new_metadata["iteration"] == state.iteration


@pytest.mark.asyncio
async def test_reflect_step_applies_config_sampler() -> None:
    sampler_calls: list[tuple[int, int]] = []

    def sampler(
        records: list[dict[str, object]], max_records: int
    ) -> list[dict[str, object]]:
        sampler_calls.append((len(records), max_records))
        return records[:max_records]

    config = GepaConfig(
        max_evaluations=50,
        minibatch_size=2,
        merges_per_accept=1,
        perfect_score=1.0,
        skip_perfect_score=True,
        reflection_sampler=sampler,
        reflection_sampler_max_records=1,
    )
    state = _make_state(config=config)
    minibatch = await _training_examples(state)
    evaluator = _StubEvaluator([_eval_results([0.4, 0.5]), _eval_results([0.6, 0.7])])
    batch_sampler = _StubBatchSampler(minibatch)
    dataset = {"instructions": [{"feedback": "a"}, {"feedback": "b"}]}
    stub_adapter = _StubAdapter(dataset=dataset)
    adapter = cast(Adapter[str, str, dict[str, str]], stub_adapter)
    generator = _StubProposalGenerator({"instructions": "Improved text"})
    deps = _make_deps(
        adapter=adapter,
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
    )
    ctx = _ctx(state, deps)

    await reflect_step(ctx)

    assert sampler_calls == [(2, 1)]
    assert isinstance(generator.last_reflective_data, ComponentReflectiveDataset)
    assert len(generator.last_reflective_data.records_by_component["instructions"]) == 1


@pytest.mark.asyncio
async def test_reflect_step_rejects_when_not_improved() -> None:
    state = _make_state()
    minibatch = await _training_examples(state)
    evaluator = _StubEvaluator([_eval_results([0.6, 0.6]), _eval_results([0.6, 0.6])])
    batch_sampler = _StubBatchSampler(minibatch)
    stub_adapter = _StubAdapter()
    adapter = cast(Adapter[str, str, dict[str, str]], stub_adapter)
    generator = _StubProposalGenerator({"instructions": "Same text"})
    deps = _make_deps(
        adapter=adapter,
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
    )
    ctx = _ctx(state, deps)

    result = await reflect_step(ctx)

    assert result == "continue"
    assert state.last_accepted is False
    assert len(state.candidates) == 1
    assert state.merge_scheduled == 0
    assert state.total_evaluations == 4


@pytest.mark.asyncio
async def test_reflect_step_skips_candidate_eval_on_noop_proposal() -> None:
    class _NoopProposalGenerator(_StubProposalGenerator):
        async def propose_texts(
            self,
            *,
            candidate: CandidateProgram,
            reflective_data: ReflectiveDataset,
            components: Sequence[str],
            model: Any,
            iteration: int | None = None,
            current_best_score: float | None = None,
            parent_score: float | None = None,
            model_settings: Any = None,
            example_bank: Any = None,
        ) -> ProposalResult:
            self.calls += 1
            self.last_reflective_data = reflective_data
            return ProposalResult(texts={}, component_metadata={}, reasoning=None)

    state = _make_state()
    minibatch = await _training_examples(state)
    evaluator = _StubEvaluator([_eval_results([0.4, 0.5])])
    batch_sampler = _StubBatchSampler(minibatch)
    stub_adapter = _StubAdapter()
    adapter = cast(Adapter[str, str, dict[str, str]], stub_adapter)
    generator = _NoopProposalGenerator({})
    deps = _make_deps(
        adapter=adapter,
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
    )
    ctx = _ctx(state, deps)

    result = await reflect_step(ctx)

    assert result == "continue"
    assert state.last_accepted is False
    assert len(state.candidates) == 1
    assert state.merge_scheduled == 0
    assert state.total_evaluations == 2
    assert stub_adapter.dataset_calls == 1
    assert generator.calls == 1
    assert evaluator.calls == 1


@pytest.mark.asyncio
async def test_reflect_step_skips_when_batch_is_perfect() -> None:
    state = _make_state()
    minibatch = await _training_examples(state)
    evaluator = _StubEvaluator([_eval_results([1.0, 1.0])])
    batch_sampler = _StubBatchSampler(minibatch)
    stub_adapter = _StubAdapter()
    adapter = cast(Adapter[str, str, dict[str, str]], stub_adapter)
    generator = _StubProposalGenerator({"instructions": "Unused"})
    deps = _make_deps(
        adapter=adapter,
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
    )
    ctx = _ctx(state, deps)

    result = await reflect_step(ctx)

    assert result == "continue"
    assert state.last_accepted is False
    assert len(state.candidates) == 1
    assert stub_adapter.dataset_calls == 0
    assert generator.calls == 0
    assert evaluator.calls == 1
    assert state.total_evaluations == 2


@pytest.mark.asyncio
async def test_reflect_step_requires_model() -> None:
    state = _make_state()
    minibatch = await _training_examples(state)
    evaluator = _StubEvaluator([_eval_results([0.3, 0.4])])
    batch_sampler = _StubBatchSampler(minibatch)
    adapter = cast(Adapter[str, str, dict[str, str]], _StubAdapter())
    generator = _StubProposalGenerator({"instructions": "Improved"})
    deps = _make_deps(
        adapter=adapter,
        evaluator=evaluator,
        batch_sampler=batch_sampler,
        proposal_generator=generator,
        model=None,
    )
    ctx = _ctx(state, deps)

    with pytest.raises(ValueError, match="model must be configured"):
        await reflect_step(ctx)


@pytest.mark.asyncio
async def test_reflect_step_preserves_shared_dataset_with_all_selector() -> None:
    class _SharedAdapter:
        def __init__(self) -> None:
            self.dataset_calls = 0
            self.records = [{"messages": [], "feedback": "shared"}]

        async def evaluate(self, batch, candidate, capture_traces):  # pragma: no cover
            raise RuntimeError("evaluate should not be called in ReflectStep tests")

        def make_reflective_dataset(
            self,
            *,
            candidate,
            eval_batch,
            components_to_update,
            include_case_metadata: bool = False,
            include_expected_output: bool = False,
        ) -> SharedReflectiveDataset:
            self.dataset_calls += 1
            return SharedReflectiveDataset(records=list(self.records))

        def get_components(self) -> CandidateMap:
            return {"instructions": ComponentValue(name="instructions", text="seed")}

    config = GepaConfig(
        max_evaluations=100,
        minibatch_size=2,
        merges_per_accept=2,
        perfect_score=1.0,
        skip_perfect_score=True,
        component_selector="all",
    )
    state = _make_state(config=config)
    minibatch = await _training_examples(state)
    evaluator = _StubEvaluator([_eval_results([0.4, 0.6]), _eval_results([0.7, 0.9])])
    batch_sampler = _StubBatchSampler(minibatch)
    shared_adapter = _SharedAdapter()
    adapter = cast(Adapter[str, str, dict[str, str]], shared_adapter)
    generator = _StubProposalGenerator({"instructions": "Improved text"})

    deps = GepaDeps(
        adapter=adapter,
        evaluator=evaluator,
        pareto_manager=ParetoFrontManager(),
        candidate_selector=CurrentBestCandidateSelector(),
        component_selector=AllComponentSelector(),
        batch_sampler=batch_sampler,
        proposal_generator=generator,
        merge_builder=MergeProposalBuilder(),
        model="reflection-model",
    )
    ctx = _ctx(state, deps)

    await reflect_step(ctx)

    assert isinstance(generator.last_reflective_data, SharedReflectiveDataset)
    assert generator.calls == 1
    assert shared_adapter.dataset_calls == 1

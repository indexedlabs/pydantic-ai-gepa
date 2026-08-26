"""Tests for the MergeStep implementation."""

from __future__ import annotations

import pytest
from pydantic_graph import StepContext

from typing import Literal, Sequence, cast

from pydantic_evals import Case

from pydantic_ai_gepa.gepa_graph.datasets import ListDataLoader
from pydantic_ai_gepa.gepa_graph.deps import GepaDeps
from pydantic_ai_gepa.gepa_graph.evaluation import (
    EvaluationResults,
    ParetoFrontManager,
    ParallelEvaluator,
)
from pydantic_ai_gepa.gepa_graph.models import (
    CandidateMap,
    CandidateProgram,
    ComponentValue,
    GepaConfig,
    GepaState,
)
from pydantic_ai_gepa.gepa_graph.steps import merge_step
from pydantic_ai_gepa.gepa_graph.proposal import (
    InstructionProposalGenerator,
    MergeProposalBuilder,
)
from pydantic_ai_gepa.gepa_graph.selectors import (
    BatchSampler,
    CurrentBestCandidateSelector,
    RoundRobinComponentSelector,
)
from pydantic_ai_gepa.types import RolloutOutput
from pydantic_ai_gepa.adapter import Adapter, SharedReflectiveDataset


def _make_data_inst(case_id: str) -> Case[str, str, dict[str, str]]:
    return Case(name=case_id, inputs=f"prompt-{case_id}", metadata={})


def _make_state() -> GepaState:
    validation = [_make_data_inst(f"case-{idx}") for idx in range(5)]
    config = GepaConfig(
        max_evaluations=100,
        use_merge=True,
        min_shared_validation=5,
    )
    return GepaState(
        config=config,
        training_set=ListDataLoader(validation),
        validation_set=ListDataLoader(validation),
    )


def _ctx(
    state: GepaState,
    deps: GepaDeps,
) -> StepContext[GepaState, GepaDeps, None]:
    return StepContext(state=state, deps=deps, inputs=None)


def _add_candidate(
    state: GepaState,
    *,
    instructions: str,
    tools: str,
    creation_type: Literal["seed", "reflection", "merge"],
    parent_indices: list[int],
    iteration: int,
) -> CandidateProgram:
    idx = len(state.candidates)
    candidate = CandidateProgram(
        idx=idx,
        components={
            "instructions": ComponentValue(
                name="instructions", text=instructions, version=iteration
            ),
            "tools": ComponentValue(name="tools", text=tools, version=iteration),
        },
        creation_type=creation_type,
        parent_indices=parent_indices,
        discovered_at_iteration=iteration,
        discovered_at_evaluation=iteration,
    )
    state.add_candidate(candidate, auto_assign_idx=False)
    return candidate


def _populate_scores(candidate: CandidateProgram, scores: list[float]) -> None:
    for idx, score in enumerate(scores):
        candidate.record_validation(
            data_id=f"case-{idx}",
            score=score,
            output=RolloutOutput.from_success(f"{candidate.idx}-{idx}"),
        )


def _build_lineage(state: GepaState) -> tuple[int, int, int]:
    ancestor = _add_candidate(
        state,
        instructions="Seed instructions",
        tools="Seed tools",
        creation_type="seed",
        parent_indices=[],
        iteration=0,
    )
    _populate_scores(ancestor, [0.5, 0.5, 0.4, 0.3, 0.3])

    parent1 = _add_candidate(
        state,
        instructions="Parent1 instructions",
        tools="Seed tools",
        creation_type="reflection",
        parent_indices=[ancestor.idx],
        iteration=1,
    )
    _populate_scores(parent1, [0.7, 0.6, 0.5, 0.5, 0.4])

    parent2 = _add_candidate(
        state,
        instructions="Seed instructions",
        tools="Parent2 tools",
        creation_type="reflection",
        parent_indices=[ancestor.idx],
        iteration=2,
    )
    _populate_scores(parent2, [0.6, 0.7, 0.6, 0.4, 0.5])

    return ancestor.idx, parent1.idx, parent2.idx


async def _validation_instances(
    state: GepaState,
) -> list[Case[str, str, dict[str, str]]]:
    loader = state.validation_set
    assert loader is not None
    ids = await loader.all_ids()
    return cast(list[Case[str, str, dict[str, str]]], await loader.fetch(ids))


class _StubAdapter:
    reflection_sampler = None
    agent = type("Agent", (), {"_instructions": "seed"})()
    input_spec = None

    async def evaluate(self, batch, candidate, capture_traces):  # pragma: no cover
        raise RuntimeError("evaluate should not be called in MergeStep tests")

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


class _StubEvaluator(ParallelEvaluator):
    def __init__(self, result: EvaluationResults[str]) -> None:
        self._result = result
        self.calls = 0

    async def evaluate_batch(self, **kwargs):
        self.calls += 1
        return self._result


class _StubMergeBuilder(MergeProposalBuilder):
    def __init__(
        self,
        *,
        pair: tuple[int, int],
        ancestor_idx: int,
        merged_candidate: CandidateProgram,
        subsample: list[Case[str, str, dict[str, str]]],
        register_returns: bool = True,
    ) -> None:
        super().__init__(seed=0)
        self._pair = pair
        self._ancestor_idx = ancestor_idx
        self._merged_candidate = merged_candidate
        self._subsample = list(subsample)
        self._register_returns = register_returns

    def find_merge_pair(
        self, state: GepaState, dominators: Sequence[int]
    ) -> tuple[int, int] | None:
        return self._pair

    def find_common_ancestor(
        self, state: GepaState, idx1: int, idx2: int
    ) -> int | None:
        return self._ancestor_idx

    def build_merged_candidate(
        self,
        state: GepaState,
        parent1_idx: int,
        parent2_idx: int,
        ancestor_idx: int,
    ) -> CandidateProgram:
        return self._merged_candidate

    async def select_merge_subsample(
        self,
        state: GepaState,
        parent1_idx: int,
        parent2_idx: int,
    ) -> list[tuple[str, Case[str, str, dict[str, str]]]]:
        subsample: list[tuple[str, Case[str, str, dict[str, str]]]] = []
        for idx, inst in enumerate(self._subsample):
            case_name = inst.name or f"case-{idx}"
            subsample.append((case_name, inst))
        return subsample

    def register_candidate(
        self,
        *,
        candidate: CandidateProgram,
        parent1_idx: int,
        parent2_idx: int,
    ) -> bool:
        return self._register_returns


def _make_deps(
    *,
    merge_builder: MergeProposalBuilder,
    evaluator: ParallelEvaluator,
) -> GepaDeps:
    return GepaDeps(
        adapter=cast(Adapter[str, str, dict[str, str]], _StubAdapter()),
        evaluator=evaluator,
        pareto_manager=ParetoFrontManager(),
        candidate_selector=CurrentBestCandidateSelector(),
        component_selector=RoundRobinComponentSelector(),
        batch_sampler=BatchSampler(),
        proposal_generator=InstructionProposalGenerator(),
        merge_builder=merge_builder,
        model="test-model",
    )


def _evaluation_results(
    subsample: Sequence[Case[str, str, dict[str, str]]], scores: list[float]
) -> EvaluationResults[str]:
    return EvaluationResults(
        data_ids=[inst.name or f"case-{idx}" for idx, inst in enumerate(subsample)],
        scores=list(scores),
        outputs=[RolloutOutput.from_success("merged")] * len(subsample),
        trajectories=None,
    )


@pytest.mark.asyncio
async def test_merge_step_accepts_when_scores_non_strictly_better() -> None:
    state = _make_state()
    ancestor_idx, parent1_idx, parent2_idx = _build_lineage(state)
    validation_items = await _validation_instances(state)
    subsample: list[Case[str, str, dict[str, str]]] = validation_items[:3]

    merged_candidate = CandidateProgram(
        idx=len(state.candidates),
        components={
            "instructions": ComponentValue(
                name="instructions", text="Parent1 instructions", version=3
            ),
            "tools": ComponentValue(name="tools", text="Parent2 tools", version=3),
        },
        creation_type="merge",
        parent_indices=[parent1_idx, parent2_idx],
        discovered_at_iteration=3,
        discovered_at_evaluation=10,
    )

    results = _evaluation_results(subsample, [0.7, 0.65, 0.6])
    evaluator = _StubEvaluator(results)
    builder = _StubMergeBuilder(
        pair=(parent1_idx, parent2_idx),
        ancestor_idx=ancestor_idx,
        merged_candidate=merged_candidate,
        subsample=subsample,
    )
    deps = _make_deps(merge_builder=builder, evaluator=evaluator)
    ctx = _ctx(state, deps)

    next_node = await merge_step(ctx)

    assert next_node == "evaluate"
    assert evaluator.calls == 1
    assert state.last_accepted is True
    assert len(state.candidates) == 4
    assert state.merge_attempts == 1
    new_candidate = state.candidates[-1]
    assert new_candidate.creation_type == "merge"
    assert new_candidate.parent_indices == [parent1_idx, parent2_idx]
    assert new_candidate.minibatch_scores == results.scores
    assert set(new_candidate.validation_scores) == {inst.name for inst in subsample}


@pytest.mark.asyncio
async def test_merge_step_rejects_when_merged_scores_lower() -> None:
    state = _make_state()
    ancestor_idx, parent1_idx, parent2_idx = _build_lineage(state)
    validation_items = await _validation_instances(state)
    subsample = validation_items[:3]

    merged_candidate = CandidateProgram(
        idx=len(state.candidates),
        components={
            "instructions": ComponentValue(
                name="instructions", text="Parent1 instructions", version=3
            ),
            "tools": ComponentValue(name="tools", text="Parent2 tools", version=3),
        },
        creation_type="merge",
        parent_indices=[parent1_idx, parent2_idx],
        discovered_at_iteration=3,
        discovered_at_evaluation=10,
    )

    results = _evaluation_results(subsample, [0.5, 0.5, 0.45])
    evaluator = _StubEvaluator(results)
    builder = _StubMergeBuilder(
        pair=(parent1_idx, parent2_idx),
        ancestor_idx=ancestor_idx,
        merged_candidate=merged_candidate,
        subsample=subsample,
    )
    deps = _make_deps(merge_builder=builder, evaluator=evaluator)
    ctx = _ctx(state, deps)

    next_node = await merge_step(ctx)

    assert next_node == "continue"
    assert evaluator.calls == 1
    assert state.last_accepted is False
    assert len(state.candidates) == 3
    assert state.merge_attempts == 1


@pytest.mark.asyncio
async def test_merge_step_skips_when_duplicate_detected() -> None:
    state = _make_state()
    ancestor_idx, parent1_idx, parent2_idx = _build_lineage(state)
    validation_items = await _validation_instances(state)
    subsample = validation_items[:3]

    merged_candidate = CandidateProgram(
        idx=len(state.candidates),
        components={
            "instructions": ComponentValue(
                name="instructions", text="Parent1 instructions", version=3
            ),
            "tools": ComponentValue(name="tools", text="Parent2 tools", version=3),
        },
        creation_type="merge",
        parent_indices=[parent1_idx, parent2_idx],
        discovered_at_iteration=3,
        discovered_at_evaluation=10,
    )

    results = _evaluation_results(subsample, [0.7, 0.65, 0.6])
    evaluator = _StubEvaluator(results)
    builder = _StubMergeBuilder(
        pair=(parent1_idx, parent2_idx),
        ancestor_idx=ancestor_idx,
        merged_candidate=merged_candidate,
        subsample=subsample,
        register_returns=False,
    )
    deps = _make_deps(merge_builder=builder, evaluator=evaluator)
    ctx = _ctx(state, deps)

    next_node = await merge_step(ctx)

    assert next_node == "continue"
    assert evaluator.calls == 0
    assert state.last_accepted is False
    assert len(state.candidates) == 3
    assert state.merge_attempts == 0

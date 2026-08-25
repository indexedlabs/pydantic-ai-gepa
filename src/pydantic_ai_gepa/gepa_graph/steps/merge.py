"""Merge step - combines two Pareto candidates via genetic crossover."""

from __future__ import annotations

from typing import Any, Iterable

import logfire
from pydantic_graph import StepContext

from pydantic_evals import Case
from ..deps import GepaDeps
from ..models import CandidateProgram, GepaState
from .continue_step import IterationAction


async def merge_step(ctx: StepContext[GepaState, GepaDeps, None]) -> IterationAction:
    """Attempt to merge two Pareto-front candidates."""

    state = ctx.state
    deps = ctx.deps

    def reject() -> IterationAction:
        state.last_accepted = False
        return "continue"

    if not _can_attempt_merge(state):
        return reject()

    dominators = deps.pareto_manager.find_dominators(state)
    pair = deps.merge_builder.find_merge_pair(state, dominators)
    if pair is None:
        return reject()

    parent1_idx, parent2_idx = pair
    ancestor_idx = deps.merge_builder.find_common_ancestor(
        state=state,
        idx1=parent1_idx,
        idx2=parent2_idx,
    )
    if ancestor_idx is None:
        return reject()

    merged_candidate = deps.merge_builder.build_merged_candidate(
        state=state,
        parent1_idx=parent1_idx,
        parent2_idx=parent2_idx,
        ancestor_idx=ancestor_idx,
    )

    already_seen = not deps.merge_builder.register_candidate(
        candidate=merged_candidate,
        parent1_idx=parent1_idx,
        parent2_idx=parent2_idx,
    )
    if already_seen or _matches_existing_candidate(state, merged_candidate):
        return reject()

    subsample = await deps.merge_builder.select_merge_subsample(
        state=state,
        parent1_idx=parent1_idx,
        parent2_idx=parent2_idx,
    )
    if not subsample:
        return reject()
    subsample_batch = [instance for _, instance in subsample]

    state.record_merge_attempt()

    with logfire.span(
        "evaluate merged candidate",
        candidate_idx=merged_candidate.idx,
        subsample_size=len(subsample),
    ):
        merged_results = await deps.evaluator.evaluate_batch(
            candidate=merged_candidate,
            batch=subsample_batch,
            adapter=deps.adapter,
            max_concurrent=state.config.max_concurrent_evaluations,
        )

    state.record_evaluation_errors(
        candidate_idx=merged_candidate.idx,
        stage="merge",
        data_ids=merged_results.data_ids,
        outputs=merged_results.outputs,
    )

    parent1_scores = _get_subsample_scores(state, parent1_idx, subsample)
    parent2_scores = _get_subsample_scores(state, parent2_idx, subsample)

    merged_candidate.minibatch_scores = list(merged_results.scores)
    _record_partial_validation(merged_candidate, merged_results)
    state.total_evaluations += len(merged_results.data_ids)

    merged_total = sum(merged_results.scores)
    baseline_total = max(sum(parent1_scores), sum(parent2_scores))
    if merged_total >= baseline_total:
        state.add_candidate(merged_candidate)
        state.last_accepted = True
        if state.merge_scheduled > 0:
            state.merge_scheduled -= 1
        return "evaluate"

    return reject()


def _can_attempt_merge(state: GepaState) -> bool:
    if not state.config.use_merge:
        return False
    if len(state.candidates) < 2:
        return False
    max_merges = state.config.max_total_merges
    if max_merges > 0 and state.merge_attempts >= max_merges:
        return False
    return True


def _matches_existing_candidate(
    state: GepaState,
    candidate: CandidateProgram,
) -> bool:
    signature = _component_signature(candidate)
    return any(
        _component_signature(existing) == signature for existing in state.candidates
    )


def _component_signature(
    candidate: CandidateProgram,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (name, component.text) for name, component in candidate.components.items()
        )
    )


def _record_partial_validation(
    candidate: CandidateProgram,
    results,
) -> None:
    for data_id, score, output in results:
        candidate.record_validation(
            data_id=data_id,
            score=score,
            output=output,
        )


def _get_subsample_scores(
    state: GepaState,
    parent_idx: int,
    subsample: Iterable[tuple[str, Case[Any, Any, Any]]],
) -> list[float]:
    if parent_idx < 0 or parent_idx >= len(state.candidates):
        raise IndexError(f"Parent index {parent_idx} is out of range.")
    candidate = state.candidates[parent_idx]
    scores: list[float] = []
    for data_id, _ in subsample:
        score = candidate.validation_scores.get(data_id)
        if score is None:
            raise ValueError(
                f"Candidate {parent_idx} missing validation score for instance {data_id!r}."
            )
        scores.append(score)
    return scores


__all__ = ["merge_step"]

"""Pareto-front management utilities."""

from __future__ import annotations

from typing import Mapping, TypeVar

from ..models import CandidateProgram, GepaState, ParetoFrontEntry
from .evaluator import EvaluationResults

SCORE_EPSILON = 1e-6

Program = TypeVar("Program", int, str)


def remove_dominated_programs(
    fronts: Mapping[str, set[Program]], scores: Mapping[Program, float]
) -> dict[str, set[Program]]:
    """Prune redundant case winners, lowest aggregate first, as upstream GEPA does.

    A program can be removed when the remaining programs still cover every
    case it wins. Unique case winners always survive; ties retain coverage.
    """
    remaining = set().union(*fronts.values()) if fronts else set()
    for program in sorted(remaining, key=lambda p: (scores[p], p)):
        if all(
            winners - {program} & remaining
            for winners in fronts.values()
            if program in winners
        ):
            remaining.remove(program)
    return {case: winners & remaining for case, winners in fronts.items()}


class ParetoFrontManager:
    """Maintain per-instance Pareto fronts and dominator calculations."""

    def update_fronts(
        self,
        state: GepaState,
        candidate_idx: int,
        eval_results: EvaluationResults,
    ) -> None:
        """Merge validation scores into the fronts without retaining outputs."""
        for data_id, score, _output in eval_results:
            entry = state.pareto_front.get(data_id)
            if entry is None:
                entry = ParetoFrontEntry(data_id=data_id)
                state.pareto_front[data_id] = entry
            entry.best_outputs.clear()
            entry.update(
                candidate_idx=candidate_idx,
                score=score,
            )

    def find_dominators(self, state: GepaState) -> list[int]:
        """Return indices of non-dominated candidates."""
        candidates = state.candidates
        if not candidates:
            return []

        dominated: set[int] = set()
        for idx, candidate in enumerate(candidates):
            if idx in dominated:
                continue
            if not candidate.validation_scores:
                # Until evaluated, treat as potentially useful.
                continue
            for other_idx, other in enumerate(candidates):
                if idx == other_idx or other_idx in dominated:
                    continue
                if self._dominates(other, candidate):
                    dominated.add(idx)
                    break

        return [idx for idx in range(len(candidates)) if idx not in dominated]

    def _dominates(
        self,
        candidate_a: CandidateProgram,
        candidate_b: CandidateProgram,
    ) -> bool:
        """Return True if ``candidate_a`` dominates ``candidate_b``."""
        scores_a = candidate_a.validation_scores
        scores_b = candidate_b.validation_scores
        if not scores_a or not scores_b:
            return False

        coverage_b = set(scores_b.keys())
        shared = coverage_b & set(scores_a.keys())
        if not shared or shared != coverage_b:
            return False

        strictly_better = False
        for data_id in shared:
            score_a = scores_a[data_id]
            score_b = scores_b[data_id]
            if score_a + SCORE_EPSILON < score_b:
                return False
            if score_a - SCORE_EPSILON > score_b:
                strictly_better = True

        return strictly_better

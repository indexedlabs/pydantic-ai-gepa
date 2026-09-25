"""Candidate selection strategies."""

from __future__ import annotations

from collections import Counter
from typing import Protocol
import random

from ..models import GepaState
from ..evaluation.pareto import remove_dominated_programs


class CandidateSelector(Protocol):
    """Protocol for deciding which candidate to improve next."""

    def select(self, state: GepaState) -> int:
        """Return the index of the candidate to improve."""
        ...


class ParetoCandidateSelector:
    """Sample candidates proportionally to their Pareto-front frequency."""

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def select(self, state: GepaState) -> int:
        """Select a candidate weighted by Pareto-front frequency."""
        fronts = remove_dominated_programs(
            {
                key: {
                    idx
                    for idx in entry.candidate_indices
                    if 0 <= idx < len(state.candidates)
                }
                for key, entry in state.pareto_front.items()
            },
            {
                idx: sum(candidate.validation_scores.values())
                / len(candidate.validation_scores)
                if candidate.validation_scores
                else 0.0
                for idx, candidate in enumerate(state.candidates)
            },
        )
        counts = Counter(idx for winners in fronts.values() for idx in sorted(winners))

        if counts:
            candidates = list(counts.keys())
            weights = list(counts.values())
            choice = self._rng.choices(candidates, weights=weights, k=1)[0]
            return choice

        if state.best_candidate_idx is not None:
            return state.best_candidate_idx

        if state.candidates:
            return 0

        raise ValueError("Cannot select candidate; state has no candidates.")


class CurrentBestCandidateSelector:
    """Always return the current best candidate index (fallback to seed)."""

    def select(self, state: GepaState) -> int:
        if state.best_candidate_idx is not None:
            return state.best_candidate_idx
        if state.candidates:
            return 0
        raise ValueError("Cannot select candidate; state has no candidates.")

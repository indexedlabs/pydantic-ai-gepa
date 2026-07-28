"""Tests for variance-aware candidate comparison."""

from __future__ import annotations

import pytest

from pydantic_ai_gepa.acceptance import compare_candidate_samples


def test_accepts_candidate_whose_lower_bound_clears_min_delta() -> None:
    comparison = compare_candidate_samples(
        [0.40, 0.41, 0.39],
        [0.70, 0.71, 0.69],
        min_delta=0.05,
    )

    assert comparison.verdict == "accepted"
    assert comparison.improved is True
    assert comparison.lower_bound > 0.05


def test_rejects_candidate_whose_upper_bound_clears_negative_delta() -> None:
    comparison = compare_candidate_samples(
        [0.70, 0.71, 0.69],
        [0.40, 0.41, 0.39],
        min_delta=0.05,
    )

    assert comparison.verdict == "rejected"
    assert comparison.improved is False
    assert comparison.upper_bound < -0.05


def test_classifies_deterministically_equal_candidate_as_equivalent() -> None:
    comparison = compare_candidate_samples([0.5, 0.5], [0.5, 0.5])

    assert comparison.verdict == "equivalent"
    assert comparison.lower_bound == pytest.approx(0.0)
    assert comparison.upper_bound == pytest.approx(0.0)


def test_classifies_overlapping_noisy_candidate_as_inconclusive() -> None:
    comparison = compare_candidate_samples(
        [0.4, 0.6, 0.5],
        [0.45, 0.65, 0.55],
    )

    assert comparison.verdict == "inconclusive"
    assert comparison.lower_bound < 0.0 < comparison.upper_bound


def test_single_repetition_preserves_explicit_compatibility_mode() -> None:
    accepted = compare_candidate_samples([0.5], [0.6])
    rejected = compare_candidate_samples([0.5], [0.4])
    equivalent = compare_candidate_samples([0.5], [0.5])

    assert accepted.verdict == "accepted"
    assert rejected.verdict == "rejected"
    assert equivalent.verdict == "equivalent"


@pytest.mark.parametrize(
    ("baseline", "candidate", "confidence", "min_delta", "message"),
    [
        ([], [0.5], 0.9, 0.0, "must not be empty"),
        ([0.5], [], 0.9, 0.0, "must not be empty"),
        ([0.5], [0.5], 1.0, 0.0, "between 0 and 1"),
        ([0.5], [0.5], 0.9, -0.1, "greater than or equal to zero"),
    ],
)
def test_rejects_invalid_comparison_configuration(
    baseline: list[float],
    candidate: list[float],
    confidence: float,
    min_delta: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_candidate_samples(
            baseline,
            candidate,
            confidence=confidence,
            min_delta=min_delta,
        )

"""Tests for variance-aware candidate comparison."""

from __future__ import annotations

from math import sqrt
from random import Random
from statistics import NormalDist, mean, variance

import pytest

from pydantic_ai_gepa.acceptance import _student_t_quantile, compare_candidate_samples


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


def test_single_repetition_never_accepts_without_paired_evidence() -> None:
    accepted = compare_candidate_samples([0.5], [0.6])
    rejected = compare_candidate_samples([0.5], [0.4])
    equivalent = compare_candidate_samples([0.5], [0.5])

    assert accepted.verdict == "inconclusive"
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


@pytest.mark.parametrize(
    ("probability", "df", "expected"),
    [
        (0.95, 2, 2.919986),
        (0.95, 4, 2.131847),
        (0.975, 10, 2.228139),
        (0.95, 1, 6.313752),
        (0.05, 2, -2.919986),
        (0.5, 4, 0.0),
    ],
)
def test_student_t_quantile_matches_tables(
    probability: float, df: float, expected: float
) -> None:
    assert _student_t_quantile(probability, df) == pytest.approx(expected, abs=1e-6)


def test_student_t_converges_to_normal() -> None:
    assert _student_t_quantile(0.95, 100_000) == pytest.approx(
        NormalDist().inv_cdf(0.95), abs=2e-5
    )


def test_welch_degrees_of_freedom_and_alpha_split() -> None:
    baseline, candidate = [0.3, 0.5, 0.7], [0.55, 0.6, 0.65, 0.7]
    comparison = compare_candidate_samples(baseline, candidate, max_looks=3)
    first, second = variance(baseline) / 3, variance(candidate) / 4
    expected_df = (first + second) ** 2 / (first**2 / 2 + second**2 / 3)
    assert comparison.degrees_of_freedom == pytest.approx(expected_df)
    assert comparison.per_look_confidence == pytest.approx(1 - 0.1 / 3)
    assert comparison.to_dict()["max_looks"] == 3
    assert comparison.to_dict()["per_look_confidence"] == comparison.per_look_confidence
    unsplit = compare_candidate_samples(baseline, candidate)
    assert comparison.lower_bound < unsplit.lower_bound
    assert comparison.upper_bound > unsplit.upper_bound


@pytest.mark.parametrize(
    "baseline,candidate", [([0.4], [0.8, 0.8]), ([0.4, 0.4], [0.8])]
)
def test_one_under_sampled_side_cannot_accept(
    baseline: list[float], candidate: list[float]
) -> None:
    assert not compare_candidate_samples(baseline, candidate).improved


def test_deterministic_repetitions_can_accept() -> None:
    assert compare_candidate_samples([0.4, 0.4], [0.8, 0.8], max_looks=3).improved


def test_paired_comparison_uses_matching_case_differences() -> None:
    baseline = {str(i): i / 20 for i in range(10)}
    candidate = {key: value + 0.1 for key, value in reversed(list(baseline.items()))}
    comparison = compare_candidate_samples(
        [mean(baseline.values())],
        [mean(candidate.values())],
        paired_baseline_scores=baseline,
        paired_candidate_scores=candidate,
    )
    assert comparison.improved
    assert comparison.delta == pytest.approx(0.1)
    assert comparison.standard_error == pytest.approx(0.0, abs=1e-15)
    assert comparison.method == "paired_t"
    assert comparison.degrees_of_freedom == 9
    assert comparison.paired_case_count == 10


@pytest.mark.parametrize(
    "baseline,candidate",
    [
        ({"a": 0.4, "b": 0.5}, None),
        ({}, {}),
        ({}, {"a": 0.5}),
        ({"a": 0.4}, {"a": 0.5}),
        ({"a": 0.4, "b": 0.5}, {"a": 0.4, "c": 0.5}),
    ],
)
def test_paired_comparison_rejects_missing_or_unmatched_cases(
    baseline: dict[str, float],
    candidate: dict[str, float] | None,
) -> None:
    comparison = compare_candidate_samples(
        [0.4], [0.5], paired_baseline_scores=baseline, paired_candidate_scores=candidate
    )
    assert comparison.verdict == "inconclusive"
    assert not comparison.improved
    assert comparison.reason_code in {
        "paired_evidence_missing",
        "paired_cases_mismatched",
        "paired_insufficient_cases",
    }


@pytest.mark.parametrize("max_looks", [0, -1, 1.5])
def test_rejects_invalid_maximum_looks(max_looks: int) -> None:
    with pytest.raises(ValueError, match="max_looks"):
        compare_candidate_samples([0.4], [0.5], max_looks=max_looks)


def _synthetic_scores(rng: Random, cases: int, effect: float) -> list[float]:
    """Independent clipped Gaussian scores, baseline 0.4 and per-case sd 0.1."""
    return [min(1.0, max(0.0, rng.gauss(0.4 + effect, 0.1))) for _ in range(cases)]


@pytest.mark.parametrize("cases", [3, 4, 5])
def test_seeded_small_set_noise_and_effect_simulations(cases: int) -> None:
    trials = 1000
    rates = {}
    for effect in (0.0, 0.2):
        rng = Random(4689 + cases)
        accepted = 0
        for _ in range(trials):
            baseline = [mean(_synthetic_scores(rng, cases, 0.0)) for _ in range(5)]
            candidate = [mean(_synthetic_scores(rng, cases, effect)) for _ in range(5)]
            for count in range(3, 6):
                comparison = compare_candidate_samples(
                    baseline[:count],
                    candidate[:count],
                    confidence=0.9,
                    max_looks=3,
                )
                if comparison.verdict != "inconclusive":
                    accepted += comparison.improved
                    break
        rates[effect] = accepted / trials
    # Three binomial standard errors allow deterministic sampling variation.
    tolerance = 3 * sqrt(0.05 * 0.95 / trials)
    assert rates[0.0] <= 0.05 + tolerance
    assert rates[0.2] >= 0.8
    print(
        f"small cases={cases}: false promotion={rates[0.0]:.3f}, power(+0.2)={rates[0.2]:.3f}; n={trials}"
    )


def test_seeded_paired_single_repetition_simulations() -> None:
    trials, cases = 1000, 100
    rates = {}
    for effect in (0.0, 0.2):
        rng = Random(4689)
        accepted = 0
        for _ in range(trials):
            baseline = dict(
                zip(map(str, range(cases)), _synthetic_scores(rng, cases, 0.0))
            )
            candidate = dict(
                zip(map(str, range(cases)), _synthetic_scores(rng, cases, effect))
            )
            comparison = compare_candidate_samples(
                [mean(baseline.values())],
                [mean(candidate.values())],
                confidence=0.9,
                paired_baseline_scores=baseline,
                paired_candidate_scores=candidate,
            )
            accepted += comparison.improved
        rates[effect] = accepted / trials
    assert rates[0.0] <= 0.05 + 3 * sqrt(0.05 * 0.95 / trials)
    assert rates[0.2] >= 0.8
    print(
        f"paired cases={cases}: false promotion={rates[0.0]:.3f}, power(+0.2)={rates[0.2]:.3f}; n={trials}"
    )


@pytest.mark.parametrize("count", [2, 9, 10, 20])
def test_zero_spread_requires_ten_paired_cases(count):
    comparison = compare_candidate_samples(
        [0.4],
        [0.6],
        paired_baseline_scores={str(i): 0.4 for i in range(count)},
        paired_candidate_scores={str(i): 0.6 for i in range(count)},
    )
    assert comparison.improved is (count >= 10)
    if count < 10:
        assert comparison.verdict == "inconclusive"
        assert comparison.reason_code == "paired_zero_spread_insufficient_cases"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("aggregate", [False, True])
def test_paired_nonfinite_scores_are_inconclusive(value, aggregate):
    import json

    comparison = compare_candidate_samples(
        [0.4],
        [value if aggregate else 0.6],
        paired_baseline_scores={"a": 0.4, "b": 0.4},
        paired_candidate_scores={"a": value, "b": 0.6},
    )
    assert comparison.verdict == "inconclusive"
    assert comparison.reason_code == "paired_scores_non_finite"
    assert not comparison.improved
    json.dumps(comparison.to_dict(), allow_nan=False)

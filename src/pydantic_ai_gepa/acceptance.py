"""Variance-aware comparison for stochastic optimization candidates."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist, mean, variance
from typing import Literal, Sequence


AcceptanceVerdict = Literal["accepted", "rejected", "equivalent", "inconclusive"]


@dataclass(frozen=True, slots=True)
class AcceptanceComparison:
    """Statistical comparison of repeated baseline and candidate evaluations."""

    verdict: AcceptanceVerdict
    baseline_samples: tuple[float, ...]
    candidate_samples: tuple[float, ...]
    baseline_mean: float
    candidate_mean: float
    delta: float
    baseline_variance: float
    candidate_variance: float
    standard_error: float
    confidence: float
    lower_bound: float
    upper_bound: float
    min_delta: float

    @property
    def improved(self) -> bool:
        """Return whether the evidence supports adopting the candidate."""

        return self.verdict == "accepted"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "verdict": self.verdict,
            "baseline_samples": list(self.baseline_samples),
            "candidate_samples": list(self.candidate_samples),
            "baseline_sample_count": len(self.baseline_samples),
            "candidate_sample_count": len(self.candidate_samples),
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "delta": self.delta,
            "baseline_variance": self.baseline_variance,
            "candidate_variance": self.candidate_variance,
            "standard_error": self.standard_error,
            "confidence": self.confidence,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "min_delta": self.min_delta,
            "improved": self.improved,
        }


def compare_candidate_samples(
    baseline_samples: Sequence[float],
    candidate_samples: Sequence[float],
    *,
    confidence: float = 0.9,
    min_delta: float = 0.0,
) -> AcceptanceComparison:
    """Compare repeated rollout means with an uncertainty interval.

    The samples must come from evaluations of the same persisted case set.
    They need not share model randomness: the interval therefore uses the
    independent-sample standard error rather than claiming common random
    numbers.
    """

    baseline = tuple(float(value) for value in baseline_samples)
    candidate = tuple(float(value) for value in candidate_samples)
    if not baseline or not candidate:
        raise ValueError("baseline_samples and candidate_samples must not be empty.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    if min_delta < 0.0:
        raise ValueError("min_delta must be greater than or equal to zero.")

    baseline_mean = mean(baseline)
    candidate_mean = mean(candidate)
    delta = candidate_mean - baseline_mean
    baseline_variance = variance(baseline) if len(baseline) > 1 else 0.0
    candidate_variance = variance(candidate) if len(candidate) > 1 else 0.0

    if len(baseline) == len(candidate) == 1:
        standard_error = 0.0
    else:
        standard_error = sqrt(
            baseline_variance / len(baseline) + candidate_variance / len(candidate)
        )

    critical_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    margin = critical_value * standard_error
    lower_bound = delta - margin
    upper_bound = delta + margin

    if lower_bound > min_delta:
        verdict: AcceptanceVerdict = "accepted"
    elif upper_bound < -min_delta:
        verdict = "rejected"
    elif lower_bound >= -min_delta and upper_bound <= min_delta:
        verdict = "equivalent"
    else:
        verdict = "inconclusive"

    return AcceptanceComparison(
        verdict=verdict,
        baseline_samples=baseline,
        candidate_samples=candidate,
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        delta=delta,
        baseline_variance=baseline_variance,
        candidate_variance=candidate_variance,
        standard_error=standard_error,
        confidence=confidence,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        min_delta=min_delta,
    )

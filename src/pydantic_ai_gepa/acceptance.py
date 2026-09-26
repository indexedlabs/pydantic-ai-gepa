"""Variance-aware comparison for stochastic optimization candidates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import exp, isfinite, lgamma, log, log1p, sqrt
from statistics import NormalDist, mean, variance
from typing import Literal, Mapping, Sequence


AcceptanceVerdict = Literal["accepted", "rejected", "equivalent", "inconclusive"]


def _beta_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued fraction using modified Lentz."""
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) >= tiny else tiny)
    result = d
    for m in range(1, 201):
        even = 2 * m
        for coefficient in (
            m * (b - m) * x / ((a + even - 1) * (a + even)),
            -(a + m) * (a + b + m) * x / ((a + even) * (a + even + 1)),
        ):
            d = 1.0 + coefficient * d
            d = d if abs(d) >= tiny else tiny
            c = 1.0 + coefficient / c
            c = c if abs(c) >= tiny else tiny
            d = 1.0 / d
            adjustment = d * c
            result *= adjustment
        if abs(adjustment - 1.0) < 3e-14:
            return result
    raise ArithmeticError("Incomplete beta continued fraction did not converge.")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    factor = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_fraction(a, b, x) / a
    return 1.0 - factor * _beta_fraction(b, a, 1.0 - x) / b


@lru_cache(maxsize=256)
def _student_t_quantile(probability: float, degrees_of_freedom: float) -> float:
    """Invert Student's t CDF using incomplete beta and bisection."""
    if not 0.0 < probability < 1.0 or degrees_of_freedom <= 0.0:
        raise ValueError("Student-t requires 0 < probability < 1 and positive df.")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -_student_t_quantile(1.0 - probability, degrees_of_freedom)
    if degrees_of_freedom >= 1e7:
        return NormalDist().inv_cdf(probability)

    def tail(value: float) -> float:
        x = degrees_of_freedom / (degrees_of_freedom + value * value)
        return 0.5 * _regularized_beta(x, degrees_of_freedom / 2.0, 0.5)

    target = 1.0 - probability
    low, high = 0.0, 1.0
    while tail(high) > target:
        high *= 2.0
    for _ in range(50):
        midpoint = (low + high) / 2.0
        if tail(midpoint) > target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


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
    per_look_confidence: float
    max_looks: int
    degrees_of_freedom: float | None
    method: Literal["welch_t", "paired_t"]
    paired_case_count: int
    reason_code: str | None = None

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
            "per_look_confidence": self.per_look_confidence,
            "max_looks": self.max_looks,
            "degrees_of_freedom": self.degrees_of_freedom,
            "method": self.method,
            "paired_case_count": self.paired_case_count,
            **({"reason_code": self.reason_code} if self.reason_code else {}),
        }


def compare_candidate_samples(
    baseline_samples: Sequence[float],
    candidate_samples: Sequence[float],
    *,
    confidence: float = 0.9,
    min_delta: float = 0.0,
    max_looks: int = 1,
    paired_baseline_scores: Mapping[str, float] | None = None,
    paired_candidate_scores: Mapping[str, float] | None = None,
) -> AcceptanceComparison:
    """Compare evaluation means with a Welch or paired Student-t interval.

    Repeated evaluations must use the same persisted case set. ``max_looks``
    is fixed before sampling: every sequential look spends an equal share of
    the configured alpha. Paired mode instead tests differences across the
    same cases, allowing one evaluation per side when case scores are present.
    """

    baseline = tuple(float(value) for value in baseline_samples)
    candidate = tuple(float(value) for value in candidate_samples)
    paired = paired_baseline_scores is not None or paired_candidate_scores is not None
    reason_code = None
    if paired:
        if not paired_baseline_scores or not paired_candidate_scores:
            reason_code = "paired_evidence_missing"
        elif paired_baseline_scores.keys() != paired_candidate_scores.keys():
            reason_code = "paired_cases_mismatched"
        elif len(paired_baseline_scores) < 2:
            reason_code = "paired_insufficient_cases"
        elif not all(
            isfinite(value)
            for scores in (paired_baseline_scores, paired_candidate_scores)
            for value in scores.values()
        ):
            reason_code = "paired_scores_non_finite"
        if not all(isfinite(value) for value in (*baseline, *candidate)):
            reason_code = "paired_scores_non_finite"
            # Do not serialize invalid scores into public comparison records.
            baseline = tuple(value for value in baseline if isfinite(value))
            candidate = tuple(value for value in candidate if isfinite(value))
        if not baseline or not candidate:
            reason_code = reason_code or "paired_evidence_missing"
    if not paired and (not baseline or not candidate):
        raise ValueError("baseline_samples and candidate_samples must not be empty.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    if not all(isfinite(value) for value in (*baseline, *candidate)):
        raise ValueError("Samples must be finite.")
    if not isinstance(max_looks, int) or max_looks < 1:
        raise ValueError("max_looks must be a positive integer.")
    if not isfinite(min_delta) or min_delta < 0.0:
        raise ValueError("min_delta must be greater than or equal to zero.")

    baseline_mean = mean(baseline) if baseline else 0.0
    candidate_mean = mean(candidate) if candidate else 0.0
    delta = candidate_mean - baseline_mean
    baseline_variance = variance(baseline) if len(baseline) > 1 else 0.0
    candidate_variance = variance(candidate) if len(candidate) > 1 else 0.0

    method: Literal["welch_t", "paired_t"] = "paired_t" if paired else "welch_t"
    paired_case_count = 0
    degrees_of_freedom = None
    enough_samples = len(baseline) >= 2 and len(candidate) >= 2
    baseline_term = baseline_variance / len(baseline) if baseline else 0.0
    candidate_term = candidate_variance / len(candidate) if candidate else 0.0
    standard_error = sqrt(baseline_term + candidate_term)
    if enough_samples and standard_error > 0.0:
        degrees_of_freedom = (baseline_term + candidate_term) ** 2 / (
            baseline_term**2 / (len(baseline) - 1)
            + candidate_term**2 / (len(candidate) - 1)
        )

    if paired and reason_code is None:
        assert (
            paired_baseline_scores is not None and paired_candidate_scores is not None
        )
        differences = [
            paired_candidate_scores[key] - score
            for key, score in paired_baseline_scores.items()
        ]
        method = "paired_t"
        paired_case_count = len(differences)
        delta = mean(differences)
        standard_error = sqrt(variance(differences) / paired_case_count)
        degrees_of_freedom = float(paired_case_count - 1)
        enough_samples = True
        # A tiny constant gain on a handful of cases is not reliable evidence.
        # Treat rounding-scale spread as zero too; require at least ten cases.
        if standard_error <= 1e-15 and paired_case_count < 10:
            reason_code = "paired_zero_spread_insufficient_cases"

    per_look_confidence = 1.0 - (1.0 - confidence) / max_looks
    critical_value = (
        _student_t_quantile(0.5 + per_look_confidence / 2.0, degrees_of_freedom)
        if degrees_of_freedom is not None and standard_error > 0.0
        else 0.0
    )
    margin = critical_value * standard_error
    lower_bound = delta - margin
    upper_bound = delta + margin

    if reason_code is not None:
        verdict: AcceptanceVerdict = "inconclusive"
    elif lower_bound > min_delta and enough_samples:
        verdict = "accepted"
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
        per_look_confidence=per_look_confidence,
        max_looks=max_looks,
        degrees_of_freedom=degrees_of_freedom,
        method=method,
        paired_case_count=paired_case_count,
        reason_code=reason_code,
    )

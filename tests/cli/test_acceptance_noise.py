"""Stubbed managed acceptance checks, including seeded stochastic evaluators."""

from __future__ import annotations

from dataclasses import replace
import random
from statistics import mean
from typing import Any

import pytest
import typer

from pydantic_ai_gepa.cli import run
from pydantic_ai_gepa.cli.eval import EvalOutcome
from pydantic_ai_gepa.evaluation import EvaluationRecord
from pydantic_ai_gepa.types import RolloutOutput


def _state(**changes: Any) -> run.RunState:
    return replace(
        run.RunState(
            run_id="noise-test",
            status="running",
            max_iterations=100,
            size=3,
            seed=0,
            next_epoch=1,
            concurrency=1,
            threshold=0.999,
            acceptance_repetitions=3,
            acceptance_max_repetitions=5,
            acceptance_confidence=0.9,
            acceptance_min_delta=0.0,
            candidate_source="components",
            iterations=0,
            created_at="2026-09-25T00:00:00Z",
            updated_at="2026-09-25T00:00:00Z",
        ),
        **changes,
    )


def _outcome(
    scores: list[float],
    *,
    iteration: int = 1,
    candidate_id: str = "candidate",
    failed: bool = False,
) -> EvalOutcome:
    records = [
        EvaluationRecord(
            case_id=f"case-{i}",
            score=score,
            feedback=None,
            payload={
                "output": RolloutOutput.from_error(
                    RuntimeError("transport failed"), kind="system"
                )
            }
            if failed and i == 0
            else {},
        )
        for i, score in enumerate(scores)
    ]
    return EvalOutcome(
        records=records,
        summary={
            "iterations": iteration,
            "mean_score": mean(scores),
            "candidate_id": candidate_id,
            "minibatch_id": "batch",
            "eval_id": f"eval-{iteration}",
            "report_path": "report.md",
            "trace_path": None,
            "n_failures": len(scores),
        },
        report_path=None,
        trace_path=None,
    )


def _stub_validation(
    monkeypatch: pytest.MonkeyPatch,
    scores: list[list[float]],
    *,
    failed_call: int | None = None,
) -> list[int]:
    calls: list[int] = []

    def evaluate(state: run.RunState, **kwargs: Any):
        index = len(calls)
        calls.append(index)
        outcome = _outcome(
            scores[index], iteration=state.iterations + 1, failed=index == failed_call
        )
        return replace(
            state,
            iterations=state.iterations + 1,
            validation_evaluations=state.validation_evaluations + 1,
        ), outcome

    monkeypatch.setattr(run, "_evaluate_validation_candidate", evaluate)
    return calls


def test_old_state_defaults_to_no_paired_mode_or_incumbent_evidence() -> None:
    raw = _state().to_dict()
    for key in (
        "acceptance_paired_min_cases",
        "best_validation_samples",
        "best_validation_per_case_scores",
    ):
        raw.pop(key)
    restored = run.RunState.from_dict(raw)
    assert restored.acceptance_paired_min_cases is None
    assert restored.best_validation_samples == ()
    assert restored.best_validation_per_case_scores == {}


def test_paired_validation_evidence_roundtrips_with_case_ids() -> None:
    state = run._mark_best_validation_samples(
        _state(acceptance_paired_min_cases=2), [_outcome([0.3, 0.5])]
    )
    restored = run.RunState.from_dict(state.to_dict())
    assert restored.best_validation_samples == (0.4,)
    assert restored.best_validation_per_case_scores == {"case-0": 0.3, "case-1": 0.5}
    assert restored.acceptance_paired_min_cases == 2


def test_reflection_baseline_discards_failure_selected_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run, "_held_out_validation_enabled", lambda: False)
    calls: list[dict[str, Any]] = []

    def evaluate(**kwargs: Any) -> EvalOutcome:
        calls.append(kwargs)
        return _outcome([0.6], iteration=len(calls) + 1)

    monkeypatch.setattr(run, "run_eval_once", evaluate)
    selected = _outcome([0.0], iteration=1)
    # One selected sample + 3 fresh baselines + 3 candidate repetitions.
    state, outcomes = run._capture_reflection_baseline(
        _state(iterations=1, max_iterations=7), selected
    )
    assert len(calls) == len(outcomes) == 3
    assert selected not in outcomes
    assert state.reflection_baseline_report_path == selected.summary["report_path"]
    assert len(state.reflection_baseline_report_paths) == 4
    assert state.reflection_baseline_samples == (0.6, 0.6, 0.6)
    assert state.reflection_baseline_eval_ids == ("eval-2", "eval-3", "eval-4")
    assert state.iterations == 4
    assert all(call["minibatch_id"] == "batch" for call in calls)


@pytest.mark.parametrize("validation", [False, True])
def test_reflection_baseline_budget_reserves_fresh_samples_and_validation(
    monkeypatch: pytest.MonkeyPatch,
    validation: bool,
) -> None:
    monkeypatch.setattr(run, "_held_out_validation_enabled", lambda: validation)
    monkeypatch.setattr(run, "_validation_schedule", lambda *args: (3, 5))

    def no_evaluation(**kwargs: Any) -> EvalOutcome:
        pytest.fail("Cannot afford three fresh baselines and candidate samples")

    monkeypatch.setattr(run, "run_eval_once", no_evaluation)
    state, outcomes = run._capture_reflection_baseline(
        _state(iterations=1, max_iterations=9 if validation else 6),
        _outcome([0.0], iteration=1),
    )
    assert not outcomes
    assert state.status == "done"
    assert state.reflection_baseline_samples == ()
    assert state.last_comparison["reason_code"] == "baseline_budget_exhausted"


@pytest.mark.parametrize("selected_failure", [False, True])
def test_reflection_baseline_infrastructure_failure_is_not_evidence(
    monkeypatch: pytest.MonkeyPatch,
    selected_failure: bool,
) -> None:
    monkeypatch.setattr(run, "_held_out_validation_enabled", lambda: False)
    calls: list[int] = []

    def evaluate(**kwargs: Any) -> EvalOutcome:
        calls.append(1)
        return _outcome([0.0], iteration=2, failed=True)

    monkeypatch.setattr(run, "run_eval_once", evaluate)
    state, _ = run._capture_reflection_baseline(
        _state(iterations=1), _outcome([0.0], failed=selected_failure)
    )
    assert len(calls) == (0 if selected_failure else 1)
    assert state.status == "paused_after_infrastructure_error"
    assert state.reflection_baseline_samples == ()
    assert state.last_comparison["selectable"] is False


def test_acceptance_schedule_requires_repetitions_until_paired_threshold() -> None:
    state = _state(acceptance_repetitions=1, acceptance_max_repetitions=1)
    assert run._acceptance_schedule(state, 1000) == (3, 3)
    state = replace(state, acceptance_paired_min_cases=100)
    assert run._acceptance_schedule(state, 99) == (3, 3)
    assert run._acceptance_schedule(state, 100) == (1, 1)


@pytest.mark.parametrize("old_run", [False, True])
def test_seed_collects_repeated_incumbent_validation_evidence(
    monkeypatch: pytest.MonkeyPatch,
    old_run: bool,
) -> None:
    monkeypatch.setattr(run, "_held_out_validation_enabled", lambda: True)
    monkeypatch.setattr(run, "_validation_schedule", lambda *args: (3, 5))
    monkeypatch.setattr(
        run, "_validation_dataset_identity", lambda: ("validation.jsonl", "digest")
    )
    monkeypatch.setattr(
        run, "_current_baseline_candidate_id", lambda *args, **kwargs: "candidate"
    )
    calls = _stub_validation(monkeypatch, [[0.3], [0.5], [0.4]])
    state, outcomes = run._ensure_validation_seed(
        _state(
            best_candidate_id="candidate" if old_run else None,
            validation_seeded=old_run,
        )
    )
    assert len(calls) == len(outcomes) == 3
    assert state.best_validation_samples == (0.3, 0.5, 0.4)
    assert state.best_mean_score == pytest.approx(0.4)
    assert state.validation_evaluations == state.iterations == 3
    assert run._ensure_validation_seed(state) == (state, [])


def test_seed_budget_cannot_degrade_to_one_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run, "_held_out_validation_enabled", lambda: True)
    monkeypatch.setattr(run, "_validation_schedule", lambda *args: (3, 5))
    calls = _stub_validation(monkeypatch, [])
    state, outcomes = run._ensure_validation_seed(_state(max_iterations=2))
    assert not calls and not outcomes
    assert state.best_validation_samples == ()
    assert state.last_comparison["reason_code"] == "validation_budget_exhausted"


def test_validation_seed_failure_does_not_install_training_retry_minibatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run, "_held_out_validation_enabled", lambda: True)
    monkeypatch.setattr(run, "_validation_schedule", lambda *args: (3, 5))
    monkeypatch.setattr(
        run, "_validation_dataset_identity", lambda: ("validation.jsonl", "digest")
    )

    def failed_validation(state: run.RunState, **kwargs: Any):
        outcome = _outcome([0.0], failed=True)
        outcome.summary.update(
            dataset_role="validation", minibatch_id="validation-unsaved"
        )
        return replace(state, iterations=state.iterations + 1), outcome

    monkeypatch.setattr(run, "_evaluate_validation_candidate", failed_validation)
    paused, _ = run._ensure_validation_seed(_state())
    assert paused.status == "paused_after_infrastructure_error"
    # Validation minibatches are not persisted in the training minibatch store.
    assert paused.infrastructure_retry_minibatch_id is None


def test_missing_paired_case_evidence_is_recovered_from_incumbent_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run, "_held_out_validation_enabled", lambda: True)
    monkeypatch.setattr(run, "_validation_schedule", lambda *args: (1, 1))
    monkeypatch.setattr(
        run, "_validation_dataset_identity", lambda: ("validation.jsonl", "digest")
    )
    monkeypatch.setattr(
        run, "_current_baseline_candidate_id", lambda *args, **kwargs: "candidate"
    )
    calls = _stub_validation(monkeypatch, [[0.3, 0.5]])
    state, _ = run._ensure_validation_seed(
        _state(
            acceptance_paired_min_cases=2,
            best_candidate_id="candidate",
            best_validation_samples=(0.4,),
            best_validation_per_case_scores={},
        )
    )
    assert len(calls) == 1
    assert state.best_validation_per_case_scores == {"case-0": 0.3, "case-1": 0.5}


def test_missing_evidence_only_recovered_from_incumbent_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run, "_held_out_validation_enabled", lambda: True)
    monkeypatch.setattr(
        run, "_current_baseline_candidate_id", lambda *args, **kwargs: "edited"
    )
    calls = _stub_validation(monkeypatch, [])
    state, outcomes = run._ensure_validation_seed(_state(best_candidate_id="incumbent"))
    assert not calls and not outcomes
    assert state.last_comparison["reason_code"] == "incumbent_evidence_missing"
    assert state.best_candidate_id == "incumbent"


def test_incumbent_evidence_recovery_preserves_frozen_validation_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run, "_held_out_validation_enabled", lambda: True)
    monkeypatch.setattr(run, "_validation_schedule", lambda *args: (3, 5))
    monkeypatch.setattr(
        run, "_current_baseline_candidate_id", lambda *args, **kwargs: "incumbent"
    )
    monkeypatch.setattr(
        run,
        "_validation_dataset_identity",
        lambda *args: ("validation.jsonl", "changed"),
    )
    calls = _stub_validation(monkeypatch, [[0.4], [0.4], [0.4]])
    with pytest.raises(typer.BadParameter, match="validation dataset changed"):
        run._ensure_validation_seed(
            _state(
                best_candidate_id="incumbent",
                validation_seeded=True,
                validation_dataset_path="validation.jsonl",
                validation_dataset_digest="original",
            )
        )
    assert not calls


@pytest.mark.parametrize("missing", [True, False])
def test_confirmation_fails_closed_without_evidence_or_budget(
    monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    monkeypatch.setattr(run, "_validation_schedule", lambda *args: (3, 5))
    calls = _stub_validation(monkeypatch, [])
    state = _state(
        max_iterations=2,
        best_mean_score=-100.0,
        best_validation_samples=() if missing else (0.4, 0.4, 0.4),
    )
    updated, outcomes, comparison = run._confirm_validation_candidate(state)
    assert updated == state and not calls and not outcomes
    assert comparison["verdict"] == "inconclusive"
    assert comparison["reason_code"] == (
        "incumbent_evidence_missing" if missing else "validation_budget_exhausted"
    )


def test_validation_uses_incumbent_samples_and_replaces_them_on_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run, "_validation_schedule", lambda *args: (3, 5))
    _stub_validation(monkeypatch, [[0.6], [0.6], [0.6]])
    state = _state(best_mean_score=1.0, best_validation_samples=(0.4, 0.4, 0.4))
    updated, outcomes, comparison = run._confirm_validation_candidate(state)
    assert comparison["verdict"] == "accepted"
    assert updated.best_validation_samples == state.best_validation_samples
    promoted = run._mark_best_validation_samples(updated, outcomes)
    assert promoted.best_validation_samples == (0.6, 0.6, 0.6)
    assert promoted.best_mean_score == pytest.approx(0.6)
    assert promoted.best_candidate_id == "candidate"


@pytest.mark.parametrize("seed", [True, False])
def test_failed_validation_rollout_pauses_without_scoring(
    monkeypatch: pytest.MonkeyPatch, seed: bool
) -> None:
    monkeypatch.setattr(run, "_validation_schedule", lambda *args: (3, 5))
    monkeypatch.setattr(run, "_held_out_validation_enabled", lambda: True)
    monkeypatch.setattr(
        run, "_validation_dataset_identity", lambda: ("validation.jsonl", "digest")
    )
    calls = _stub_validation(monkeypatch, [[0.9], [0.9]], failed_call=1)

    def no_comparison(*args: Any, **kwargs: Any):
        pytest.fail("An infrastructure failure must not enter the quality test")

    monkeypatch.setattr(run, "compare_candidate_samples", no_comparison)
    if seed:
        state, _ = run._ensure_validation_seed(_state())
    else:
        state, _, _ = run._confirm_validation_candidate(
            _state(best_validation_samples=(0.4, 0.4, 0.4))
        )
    assert len(calls) == 2
    assert state.status == "paused_after_infrastructure_error"
    assert state.last_comparison["selectable"] is False


@pytest.mark.parametrize("case_count", [3, 5, 100])
@pytest.mark.parametrize("effect", [0.0, 0.2])
def test_seeded_managed_confirmation_noise_and_power(
    monkeypatch: pytest.MonkeyPatch, case_count: int, effect: float
) -> None:
    """200 trials per scenario, per-case Gaussian sd=.1, clipped to [0, 1]."""
    rng = random.Random(4689)
    paired = case_count == 100
    initial = 1 if paired else 3
    maximum = 1 if paired else 5
    monkeypatch.setattr(run, "_validation_schedule", lambda *args: (initial, maximum))

    def scores(center: float) -> list[float]:
        return [max(0.0, min(1.0, rng.gauss(center, 0.1))) for _ in range(case_count)]

    accepted = 0
    for _ in range(200):
        baseline = [
            _outcome(scores(0.4), candidate_id="incumbent") for _ in range(initial)
        ]
        state = run._mark_best_validation_samples(
            _state(acceptance_paired_min_cases=100 if paired else None), baseline
        )
        _stub_validation(monkeypatch, [scores(0.4 + effect) for _ in range(maximum)])
        updated, outcomes, comparison = run._confirm_validation_candidate(state)
        accepted += comparison["verdict"] == "accepted"
        assert initial <= len(outcomes) <= maximum
        assert updated.iterations == len(outcomes)
        assert comparison["method"] == ("paired_t" if paired else "welch_t")
    rate = accepted / 200
    print(f"managed cases={case_count} effect={effect}: {accepted}/200 ({rate:.3f})")
    if effect:
        assert rate >= 0.8
    else:
        # One-sided alpha .05 plus three binomial standard errors.
        assert rate <= 0.05 + 3 * (0.05 * 0.95 / 200) ** 0.5


def test_non_selectable_seed_remains_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run, "_held_out_validation_enabled", lambda: True)
    monkeypatch.setattr(run, "_validation_schedule", lambda *args: (3, 5))
    monkeypatch.setattr(
        run, "_validation_dataset_identity", lambda: ("validation.jsonl", "digest")
    )
    calls = []

    def evaluate(state, **kwargs):
        calls.append(state.iterations)
        outcome = _outcome([0.4], iteration=state.iterations + 1, candidate_id="seed")
        outcome.summary["selectable"] = False
        return replace(
            state,
            iterations=state.iterations + 1,
            validation_evaluations=state.validation_evaluations + 1,
        ), outcome

    monkeypatch.setattr(run, "_evaluate_validation_candidate", evaluate)
    state, outcomes = run._ensure_validation_seed(_state())
    assert len(calls) == len(outcomes) == 3
    assert state.status == "running"
    assert state.validation_seeded
    assert state.best_candidate_id == "seed"
    assert state.best_validation_samples == (0.4, 0.4, 0.4)
    assert state.best_mean_score == pytest.approx(0.4)
    assert state.last_comparison is None

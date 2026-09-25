"""Seeded run-level noise control through the managed training and validation gates."""

from __future__ import annotations

from dataclasses import replace
import random
from types import SimpleNamespace
from typing import Any

import pytest

from pydantic_ai_gepa.cli import run
from pydantic_ai_gepa.cli.eval import EvalOutcome

from .test_acceptance_noise import _outcome, _state


def test_no_effect_runs_rarely_promote_across_multiple_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """500 runs, 10 candidates each, Gaussian per-case sd=.1 and no real effect.

    Every seed and proposal has the same clipped Gaussian score distribution.
    Training uses three cases; held-out validation uses five independent cases.
    Both gates use three to five repetitions at confidence .9. Validation evidence
    is retained across proposals and replaced only after an accepted promotion.
    """
    rng = random.Random(4689)
    total_runs = 500
    candidates_per_run = 10
    rows = 0
    candidate_id = "seed"
    validation_calls = 0

    def evaluate(case_count: int) -> EvalOutcome:
        nonlocal rows
        rows += 1
        scores = [max(0.0, min(1.0, rng.gauss(0.5, 0.1))) for _ in range(case_count)]
        return _outcome(scores, iteration=rows, candidate_id=candidate_id)

    def evaluate_training(**kwargs: Any) -> EvalOutcome:
        return evaluate(3)

    def evaluate_validation(state: run.RunState, **kwargs: Any):
        nonlocal validation_calls
        validation_calls += 1
        outcome = evaluate(5)
        return replace(
            state,
            iterations=rows,
            validation_evaluations=state.validation_evaluations + 1,
        ), outcome

    monkeypatch.setattr(run, "run_eval_once", evaluate_training)
    monkeypatch.setattr(run, "_evaluate_validation_candidate", evaluate_validation)
    monkeypatch.setattr(run, "_held_out_validation_enabled", lambda: True)
    monkeypatch.setattr(run, "_validation_schedule", lambda *args: (3, 5))
    monkeypatch.setattr(
        run, "_validation_dataset_identity", lambda: ("validation.jsonl", "fixed")
    )
    monkeypatch.setattr(
        run.MinibatchStore,
        "load",
        lambda *args: SimpleNamespace(case_ids=["case-0", "case-1", "case-2"]),
    )

    promoted_runs = 0
    training_acceptances = 0
    for _ in range(total_runs):
        rows = 0
        candidate_id = "seed"
        state, seed_outcomes = run._ensure_validation_seed(_state(max_iterations=200))
        assert len(seed_outcomes) == 3
        any_promotion = False
        for candidate_index in range(candidates_per_run):
            candidate_id = str(state.best_candidate_id)
            selected = evaluate_training()
            state = run._with_last_outcome(state, selected)
            state, _ = run._capture_reflection_baseline(state, selected)
            candidate_id = f"proposal-{candidate_index}"
            state, _, training = run._evaluate_reflected_candidate(state)
            if training["verdict"] != "accepted":
                continue
            training_acceptances += 1
            state, validation_outcomes, validation = run._confirm_validation_candidate(
                state
            )
            if validation["verdict"] == "accepted":
                any_promotion = True
                state = run._mark_best_validation_samples(state, validation_outcomes)
        promoted_runs += any_promotion

    # Ensure this exercises validation after noisy training wins, rather than
    # passing because the training gate never lets a candidate through.
    assert training_acceptances > 0
    assert validation_calls > total_runs * 3
    rate = promoted_runs / total_runs
    print(
        f"run-level null: {promoted_runs}/{total_runs} runs promoted ({rate:.3%}); "
        f"{candidates_per_run} candidates/run, {training_acceptances} training wins"
    )
    assert rate <= 0.05

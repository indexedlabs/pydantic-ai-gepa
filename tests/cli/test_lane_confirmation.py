"""Independent confirmation prevents the noisiest lane from winning by luck."""

from dataclasses import replace
import random
from types import SimpleNamespace

import pytest
import typer

from pydantic_ai_gepa.cli import run, select
from pydantic_ai_gepa.cli.eval import EvalOutcome
from pydantic_ai_gepa.cli.lanes import LaneState
from pydantic_ai_gepa.evaluation import EvaluationRecord
from pydantic_ai_gepa.types import RolloutOutput


def _state(**overrides):
    return replace(
        run.RunState(
            run_id="confirmation",
            status="running",
            max_iterations=50,
            size=3,
            seed=0,
            next_epoch=1,
            concurrency=1,
            threshold=0.999,
            acceptance_repetitions=3,
            acceptance_max_repetitions=5,
            acceptance_confidence=0.9,
            acceptance_min_delta=0.0,
            candidate_source="git",
            iterations=3,
            created_at="",
            updated_at="",
            best_candidate_id="incumbent",
            best_commit_sha="incumbent",
            best_mean_score=0.1,
            best_validation_samples=(0.5, 0.5, 0.5),
            reflection_baseline_commit_sha="incumbent",
            lanes=2,
            heldout_required=True,
        ),
        **overrides,
    )


@pytest.fixture
def selection(monkeypatch, tmp_path):
    from pydantic_ai_gepa.cli import harness_record

    # These unit tests inject held-out scoring and its private authority; real
    # index/record lookup is covered by harness and lane CLI integration tests.
    monkeypatch.setattr(harness_record, "for_run", lambda *_: object())
    lanes = [
        LaneState(
            lane=f"lane-{index}",
            status="awaiting_selection",
            iteration=1,
            branch=f"branch-{index}",
            worktree_path=str(tmp_path),
            candidate_sha=f"candidate-{index}",
            verdict="accepted",
        )
        for index in (1, 2)
    ]
    config = SimpleNamespace(acceptance=SimpleNamespace(mode="scalar"))
    monkeypatch.setattr(select.GepaConfig, "load", lambda _: config)
    monkeypatch.setattr(select, "load_all_lane_states", lambda *_: lanes)
    monkeypatch.setattr(select, "_is_ancestor", lambda *_: True)
    monkeypatch.setattr(select, "candidate_project_root", lambda _, root: root)
    monkeypatch.setattr(
        select,
        "ParetoLog",
        lambda *_: SimpleNamespace(iter_rows=lambda: [], count_budget_rows=lambda: 0),
    )
    monkeypatch.setattr(select, "_checkpoint", lambda state, *args: state)
    monkeypatch.setattr(
        select, "_load_comparison", lambda lane: {"candidate_id": lane.candidate_sha}
    )
    monkeypatch.setattr(select, "_journal_lane_outcome", lambda *_: None)
    monkeypatch.setattr(select, "_diff_stat", lambda *_: "")
    monkeypatch.setattr(select, "_record_accepted_promotion", lambda *_, **__: 1)
    monkeypatch.setattr(
        select, "_primary_checkout_state", lambda _: ("incumbent", True)
    )
    monkeypatch.setattr(select, "_emit_merge_opportunities", lambda *_: None)
    monkeypatch.setattr(run, "_validation_schedule", lambda *_: (3, 5))
    monkeypatch.setattr(run, "_ensure_validation_seed", lambda state: (state, []))
    calls = []
    scores = {
        "lane-1:validation": 0.95,
        "lane-2:validation": 0.7,
        "lane-1:confirmation": 0.5,
    }

    def evaluate(state, *, lane=None, **kwargs):
        calls.append(lane)
        score = scores[lane]
        if callable(score):
            score = score()
        records = [EvaluationRecord(f"case-{i}", score, None, {}) for i in range(3)]
        if score < 0:
            records[0].payload = {
                "output": RolloutOutput(
                    result=None,
                    success=False,
                    error_message="private failure",
                    error_kind="transport",
                )
            }
        outcome = EvalOutcome(
            records=records,
            summary={
                "candidate_id": lane.split(":")[0].replace("lane", "candidate"),
                "commit_sha": lane.split(":")[0].replace("lane", "candidate"),
                "mean_score": score,
                "dataset_role": "validation",
                "minibatch_id": "hidden",
                "report_path": None,
                "trace_path": None,
            },
            report_path=None,
            trace_path=None,
        )
        return replace(
            state,
            iterations=state.iterations + 1,
            validation_evaluations=state.validation_evaluations + 1,
        ), outcome

    monkeypatch.setattr(run, "_evaluate_validation_candidate", evaluate)
    return tmp_path, calls, scores


def test_lucky_lane_is_not_promoted_without_independent_confirmation(selection):
    root, calls, _ = selection
    state, ctx, phase = select._phase_promote(root, _state(), {})
    assert phase == "journal"
    assert ctx["winner"] is None
    assert state.best_candidate_id == "incumbent"
    assert ctx["validation_confirmation"]["verdict"] == "equivalent"
    assert (
        calls
        == ["lane-1:validation", "lane-2:validation"] + ["lane-1:confirmation"] * 3
    )


def test_lane_promotion_stores_confirmation_evidence(selection):
    root, calls, scores = selection
    scores["lane-1:confirmation"] = 0.8
    state, ctx, _ = select._phase_promote(root, _state(), {})
    assert ctx["winner"] == "lane-1"
    assert state.best_validation_samples == (0.8, 0.8, 0.8)
    assert state.best_mean_score == pytest.approx(0.8)
    assert state.best_candidate_id == "candidate-1"
    assert len(calls) == 5
    # A crash after the promotion checkpoint must not re-test the winner
    # against its own newly saved incumbent samples.
    select._phase_promote(root, state, ctx)
    assert len(calls) == 5


def test_private_authority_enables_validation_even_if_state_flag_is_false(selection):
    root, calls, scores = selection
    scores["lane-1:confirmation"] = 0.8
    state, ctx, _ = select._phase_promote(root, _state(heldout_required=False), {})
    assert (
        calls
        == ["lane-1:validation", "lane-2:validation"] + ["lane-1:confirmation"] * 3
    )
    assert state.best_candidate_id == "candidate-1"
    assert ctx["validation_confirmation"]["verdict"] == "accepted"


def test_lane_confirmation_budget_fails_closed(selection):
    root, calls, _ = selection
    state, ctx, _ = select._phase_promote(root, _state(max_iterations=7), {})
    assert state.best_candidate_id == "incumbent"
    assert ctx["winner"] is None
    assert (
        ctx["validation_confirmation"]["reason_code"] == "validation_budget_exhausted"
    )
    assert len(calls) == 2


def test_lane_confirmation_missing_incumbent_evidence_fails_closed(selection):
    root, calls, _ = selection
    state, ctx, _ = select._phase_promote(root, _state(best_validation_samples=()), {})
    assert state.best_candidate_id == "incumbent"
    assert ctx["winner"] is None
    assert ctx["validation_confirmation"]["reason_code"] == "incumbent_evidence_missing"
    assert len(calls) == 2


def test_changed_lane_finalist_cannot_promote(selection, monkeypatch):
    root, _, scores = selection
    scores["lane-1:confirmation"] = 0.8
    evaluate = run._evaluate_validation_candidate

    def changed(state, *, lane=None, **kwargs):
        state, outcome = evaluate(state, lane=lane, **kwargs)
        if lane.endswith(":confirmation"):
            outcome.summary["candidate_id"] = "changed-finalist"
        return state, outcome

    monkeypatch.setattr(run, "_evaluate_validation_candidate", changed)
    with pytest.raises(typer.BadParameter, match="finalist changed"):
        select._phase_promote(root, _state(), {})


def test_lane_confirmation_infrastructure_failure_pauses(selection, monkeypatch):
    root, calls, scores = selection
    scores["lane-1:confirmation"] = -1
    checkpoints = []
    monkeypatch.setattr(
        select, "_checkpoint", lambda state, *args: checkpoints.append(state) or state
    )
    with pytest.raises(typer.Exit):
        select._phase_promote(root, _state(), {})
    assert checkpoints[-1].status == "paused_after_infrastructure_error"
    assert checkpoints[-1].best_candidate_id == "incumbent"
    assert len(calls) == 3


@pytest.mark.parametrize("missing_evidence", [False, True])
def test_lane_confirmation_paired_mode_uses_one_fresh_sample(
    selection, monkeypatch, missing_evidence
):
    root, calls, scores = selection
    monkeypatch.setattr(run, "_validation_schedule", lambda *_: (1, 1))
    incumbent_scores = {f"case-{i}": 0.5 for i in range(3)}
    if missing_evidence:
        recovered = []

        def recover(state):
            recovered.append(state)
            return replace(state, best_validation_per_case_scores=incumbent_scores), []

        monkeypatch.setattr(run, "_ensure_validation_seed", recover)
    scores["lane-1:confirmation"] = 0.8
    state, ctx, _ = select._phase_promote(
        root,
        _state(
            best_validation_samples=(0.5,),
            best_validation_per_case_scores={}
            if missing_evidence
            else incumbent_scores,
            acceptance_paired_min_cases=3,
        ),
        {},
    )
    assert ctx["winner"] == "lane-1"
    assert ctx["validation_confirmation"]["method"] == "paired_t"
    assert state.best_validation_samples == (0.8,)
    assert len(calls) == 3
    if missing_evidence:
        assert len(recovered) == 1


@pytest.mark.parametrize("effect", [0.0, 0.2])
def test_lane_confirmation_seeded_noise(selection, effect):
    root, _, scores = selection
    rng = random.Random(4689)
    trials = 200
    promoted = 0

    def row_mean(offset=0.0):
        return sum(rng.gauss(0.5 + offset, 0.1) for _ in range(3)) / 3

    for lane in ("lane-1", "lane-2"):
        scores[f"{lane}:validation"] = lambda: row_mean(effect)
        scores[f"{lane}:confirmation"] = lambda: row_mean(effect)
    for _ in range(trials):
        baseline = tuple(row_mean() for _ in range(3))
        _, ctx, _ = select._phase_promote(
            root, _state(best_validation_samples=baseline), {}
        )
        promoted += ctx["winner"] is not None
    rate = promoted / trials
    print(
        f"lane simulation: effect={effect}, promotions={promoted}/{trials}, rate={rate:.3f}"
    )
    assert (
        rate <= 0.05 + 3 * (0.05 * 0.95 / trials) ** 0.5 if effect == 0 else rate >= 0.8
    )


def test_changed_lane_identity_cannot_reuse_private_validation(selection):
    root, calls, scores = selection
    scores["lane-1:confirmation"] = 0.5
    forged_prior = {
        "validation_results": {
            "lane-1": {
                "candidate_id": "another-candidate",
                "commit_sha": "another-commit",
                "mean_score": 999.0,
                "selectable": True,
            }
        }
    }
    state, _, _ = select._phase_promote(root, _state(), forged_prior)
    assert "lane-1:validation" in calls
    assert state.best_candidate_id == "incumbent"


def test_lane_proposal_must_match_the_scored_commit(selection, monkeypatch):
    root, calls, _ = selection
    original = run._evaluate_validation_candidate

    def mismatched(state, **kwargs):
        state, outcome = original(state, **kwargs)
        outcome.summary["commit_sha"] = "different-commit"
        return state, outcome

    monkeypatch.setattr(run, "_evaluate_validation_candidate", mismatched)
    with pytest.raises(typer.BadParameter, match="differs from the commit scored"):
        select._phase_promote(root, _state(), {})
    assert calls == ["lane-1:validation"]

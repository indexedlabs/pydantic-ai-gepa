"""Tests for ParetoFrontManager."""

from __future__ import annotations

import pytest
from pydantic_evals import Case

from pydantic_ai_gepa.gepa_graph.datasets import ListDataLoader
from pydantic_ai_gepa.gepa_graph.evaluation import EvaluationResults, ParetoFrontManager
from pydantic_ai_gepa.gepa_graph.models import (
    CandidateProgram,
    ComponentValue,
    GepaConfig,
    GepaState,
)
from pydantic_ai_gepa.types import RolloutOutput


def _make_state() -> GepaState:
    config = GepaConfig()
    training = [
        Case(name=f"case-{idx}", inputs=f"prompt-{idx}", metadata={})
        for idx in range(3)
    ]
    return GepaState(config=config, training_set=ListDataLoader(training))


def _candidate(idx: int, text: str, iteration: int) -> CandidateProgram:
    return CandidateProgram(
        idx=idx,
        components={"system": ComponentValue(name="system", text=text)},
        creation_type="seed" if idx == 0 else "reflection",
        discovered_at_iteration=iteration,
        discovered_at_evaluation=iteration,
    )


def test_pareto_manager_updates_fronts() -> None:
    state = _make_state()
    candidate = _candidate(0, "A", 0)
    state.add_candidate(candidate, auto_assign_idx=False)
    manager = ParetoFrontManager()

    results = EvaluationResults(
        data_ids=["0", "1"],
        scores=[0.5, 0.7],
        outputs=[
            RolloutOutput.from_success("o0"),
            RolloutOutput.from_success("o1"),
        ],
    )

    manager.update_fronts(state, candidate_idx=0, eval_results=results)

    assert state.pareto_front["0"].best_score == pytest.approx(0.5)
    assert state.pareto_front["0"].candidate_indices == {0}
    assert state.pareto_front["1"].best_score == pytest.approx(0.7)
    assert all(not entry.best_outputs for entry in state.pareto_front.values())


def test_pareto_manager_handles_ties() -> None:
    state = _make_state()
    cand_a = _candidate(0, "A", 0)
    cand_b = _candidate(1, "B", 1)
    state.add_candidate(cand_a, auto_assign_idx=False)
    state.add_candidate(cand_b, auto_assign_idx=False)
    manager = ParetoFrontManager()

    first = EvaluationResults(
        data_ids=["0"],
        scores=[0.6],
        outputs=[RolloutOutput.from_success("a")],
    )
    second = EvaluationResults(
        data_ids=["0"],
        scores=[0.6],
        outputs=[RolloutOutput.from_success("b")],
    )

    manager.update_fronts(state, candidate_idx=0, eval_results=first)
    manager.update_fronts(state, candidate_idx=1, eval_results=second)

    entry = state.pareto_front["0"]
    assert entry.best_score == pytest.approx(0.6)
    assert entry.candidate_indices == {0, 1}
    assert entry.best_outputs == []


def test_find_dominators_filters_dominated_candidates() -> None:
    state = _make_state()
    cand_a = _candidate(0, "A", 0)
    cand_b = _candidate(1, "B", 1)
    cand_c = _candidate(2, "C", 2)

    cand_a.record_validation(
        data_id="0", score=0.9, output=RolloutOutput.from_success("a0")
    )
    cand_a.record_validation(
        data_id="1", score=0.4, output=RolloutOutput.from_success("a1")
    )

    cand_b.record_validation(
        data_id="0", score=0.7, output=RolloutOutput.from_success("b0")
    )
    cand_b.record_validation(
        data_id="1", score=0.8, output=RolloutOutput.from_success("b1")
    )

    cand_c.record_validation(
        data_id="0", score=0.6, output=RolloutOutput.from_success("c0")
    )
    cand_c.record_validation(
        data_id="1", score=0.3, output=RolloutOutput.from_success("c1")
    )

    state.add_candidate(cand_a, auto_assign_idx=False)
    state.add_candidate(cand_b, auto_assign_idx=False)
    state.add_candidate(cand_c, auto_assign_idx=False)

    manager = ParetoFrontManager()
    dominators = manager.find_dominators(state)

    assert set(dominators) == {0, 1}

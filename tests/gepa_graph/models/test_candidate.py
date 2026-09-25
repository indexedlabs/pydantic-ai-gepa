"""Tests for candidate and component models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pydantic_ai_gepa.gepa_graph.models import CandidateProgram, ComponentValue
from pydantic_ai_gepa.types import RolloutOutput


def test_candidate_program_creation_and_conversion() -> None:
    candidate = CandidateProgram(
        idx=0,
        components={"system": ComponentValue(name="system", text="Hello")},
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )

    assert candidate.to_dict_str() == {"system": "Hello"}
    assert candidate.coverage == 0
    assert candidate.avg_validation_score == 0.0


def test_candidate_validation_tracking() -> None:
    candidate = CandidateProgram(
        idx=1,
        components={"assistant": ComponentValue(name="assistant", text="Respond")},
        creation_type="reflection",
        parent_indices=[0],
        discovered_at_iteration=1,
        discovered_at_evaluation=3,
    )

    candidate.record_validation(
        data_id="case-1",
        score=0.7,
        output=RolloutOutput.from_success("A"),
    )
    candidate.record_validation(
        data_id="case-2",
        score=0.9,
        output=RolloutOutput.from_success("B"),
    )

    assert candidate.validation_outputs == {}
    assert candidate.coverage == 2
    assert candidate.avg_validation_score == pytest.approx(0.8)


def test_candidate_invalid_idx_raises() -> None:
    with pytest.raises(ValidationError):
        CandidateProgram(
            idx=-1,
            components={"system": ComponentValue(name="system", text="Hello")},
            creation_type="seed",
            discovered_at_iteration=0,
            discovered_at_evaluation=0,
        )


def test_clone_with_new_idx() -> None:
    candidate = CandidateProgram(
        idx=0,
        components={"system": ComponentValue(name="system", text="Hello")},
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )
    clone = candidate.clone_with_new_idx(5)
    assert clone.idx == 5
    assert clone.to_dict_str() == candidate.to_dict_str()

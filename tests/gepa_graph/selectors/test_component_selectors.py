from __future__ import annotations

import pytest

from pydantic_evals import Case

from pydantic_ai_gepa.gepa_graph.datasets import ListDataLoader
from pydantic_ai_gepa.gepa_graph.models import (
    CandidateProgram,
    ComponentValue,
    GepaConfig,
    GepaState,
)
from pydantic_ai_gepa.gepa_graph.selectors import (
    AllComponentSelector,
    RoundRobinComponentSelector,
)


def _make_state(component_names: list[str]) -> GepaState:
    config = GepaConfig()
    dataset = [Case(name="case-1", inputs="x", metadata=None)]
    state = GepaState(
        config=config,
        training_set=ListDataLoader(dataset),
        validation_set=ListDataLoader(dataset),
    )
    state.add_candidate(
        CandidateProgram(
            idx=0,
            components={
                name: ComponentValue(name=name, text=f"seed {name}")
                for name in component_names
            },
            creation_type="seed",
            discovered_at_iteration=0,
            discovered_at_evaluation=0,
        )
    )
    return state


def test_round_robin_component_selector_cycles_components() -> None:
    state = _make_state(["a", "b", "c"])
    selector = RoundRobinComponentSelector()
    assert selector.select(state, 0) == ["a"]
    assert selector.select(state, 0) == ["b"]
    assert selector.select(state, 0) == ["c"]
    assert selector.select(state, 0) == ["a"]


def test_round_robin_component_selector_raises_for_missing_candidate() -> None:
    state = _make_state(["a"])
    selector = RoundRobinComponentSelector()
    with pytest.raises(IndexError):
        selector.select(state, 1)


def test_round_robin_child_inherits_advanced_parent_pointer() -> None:
    state = _make_state(["a", "b", "c"])
    selector = RoundRobinComponentSelector()
    assert selector.select(state, 0) == ["a"]
    child = state.candidates[0].model_copy(
        update={"idx": 1, "parent_indices": [0], "creation_type": "reflection"}
    )
    state.add_candidate(child)
    assert selector.select(state, 1) == ["b"]
    # Parent and child advance independently after inheritance.
    assert selector.select(state, 0) == ["b"]
    assert selector.select(state, 1) == ["c"]
    grandchild = child.model_copy(update={"idx": 2, "parent_indices": [1]})
    state.add_candidate(grandchild)
    assert selector.select(state, 2) == ["a"]


def test_all_component_selector_returns_all_components() -> None:
    state = _make_state(["a", "b"])
    selector = AllComponentSelector()
    assert selector.select(state, 0) == ["a", "b"]

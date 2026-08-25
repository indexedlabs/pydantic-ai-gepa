"""End-to-end manual iteration tests for the GEPA graph."""

from __future__ import annotations

from typing import Any, Sequence, cast

import pytest

from pydantic_graph import EndMarker, GraphTask

from pydantic_ai_gepa.adapter import Adapter
from pydantic_ai_gepa.gepa_graph.datasets import ListDataLoader
from pydantic_ai_gepa.gepa_graph import create_deps, create_gepa_graph
from pydantic_ai_gepa.gepa_graph.models import ComponentValue, GepaConfig, GepaState
from pydantic_ai_gepa.types import ReflectionConfig
from tests.gepa_graph.utils import (
    AdapterStub,
    ProposalGeneratorStub,
    make_dataset,
)


@pytest.mark.asyncio
async def test_manual_iteration_flow() -> None:
    adapter = cast(Adapter[str, str, dict[str, str]], AdapterStub())
    config = GepaConfig(
        max_evaluations=30,
        minibatch_size=2,
        seed=17,
        reflection_config=ReflectionConfig(model="reflection-model"),
    )
    deps = create_deps(
        adapter,
        config,
        seed_candidate={
            "instructions": ComponentValue(
                name="instructions", text="seed instructions"
            )
        },
    )
    deps.proposal_generator = cast(Any, ProposalGeneratorStub())

    graph = create_gepa_graph(config=config)
    dataset = make_dataset()
    loader = ListDataLoader(dataset)
    state = GepaState(
        config=config,
        training_set=loader,
        validation_set=ListDataLoader(dataset),
    )

    executed_steps: list[str] = []
    async with graph.iter(state=state, deps=deps) as run:
        async for event in run:
            executed_steps.extend(_event_step_labels(graph, event))
            if state.best_score is not None and state.best_score >= 0.8:
                state.stopped = True
                state.stop_reason = "target score met"

    run_output = run.output
    assert run_output is not None
    result = run_output

    assert "StartStep" in executed_steps
    assert "ReflectStep" in executed_steps
    assert state.stop_reason == "target score met"
    assert result.stopped is True
    assert result.best_score is not None
    assert result.original_score is not None
    assert result.best_score > result.original_score
    assert len(result.candidates) >= 2


def _event_step_labels(graph, event: EndMarker | Sequence[GraphTask]) -> list[str]:
    if isinstance(event, EndMarker):
        return ["End"]

    node_ids = {task.node_id for task in event}
    if not node_ids:
        return []

    return [_step_label(graph, node_id) for node_id in node_ids]


def _step_label(graph, node_id) -> str:
    node = graph.nodes.get(node_id)
    if node is None:
        return str(node_id)
    label = getattr(node, "label", None)
    if label:
        return label
    node_identifier = getattr(node, "id", None)
    if node_identifier is not None:
        return str(node_identifier)
    if hasattr(node, "__class__"):
        return node.__class__.__name__
    return str(node_id)

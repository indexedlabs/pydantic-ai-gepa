"""Tests for GEPA graph construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast

from pydantic_graph import Graph

from pydantic_ai_gepa.adapter import Adapter, SharedReflectiveDataset
from pydantic_ai_gepa.gepa_graph.graph import create_gepa_graph
from pydantic_ai_gepa.gepa_graph.models import CandidateMap, ComponentValue, GepaConfig


@dataclass
class _AgentStub:
    name: str | None = None


@dataclass
class _AdapterStub:
    agent: _AgentStub
    input_spec: None = None

    async def evaluate(
        self, batch, candidate, capture_traces
    ):  # pragma: no cover - unused
        raise RuntimeError("evaluate should not be called in this test")

    def make_reflective_dataset(
        self,
        *,
        candidate,
        eval_batch,
        components_to_update: Sequence[str],
    ) -> SharedReflectiveDataset:  # pragma: no cover - unused
        return SharedReflectiveDataset(records=[])

    def get_components(self) -> CandidateMap:  # pragma: no cover - unused
        return {"instructions": ComponentValue(name="instructions", text="seed")}


def _make_adapter(name: str | None = None) -> Adapter[str, str, dict[str, str]]:
    return cast(
        Adapter[str, str, dict[str, str]], _AdapterStub(agent=_AgentStub(name=name))
    )


def test_create_gepa_graph_without_merge() -> None:
    config = GepaConfig()

    graph = create_gepa_graph(config=config)

    assert isinstance(graph, Graph)
    assert graph.name == "gepa_graph"
    step_ids = {str(node_id) for node_id in graph.nodes.keys()}
    assert {
        "StartStep",
        "EvaluateStep",
        "ContinueStep",
        "ReflectStep",
        "MergeStep",
    }.issubset(step_ids)


def test_create_gepa_graph_with_merge_enabled() -> None:
    config = GepaConfig(use_merge=True)

    graph = create_gepa_graph(config=config)

    assert isinstance(graph, Graph)
    assert "MergeStep" in {str(node_id) for node_id in graph.nodes.keys()}

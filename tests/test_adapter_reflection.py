"""Tests for AgentAdapter reflective dataset creation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from pydantic_evals import Case

from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapter, AgentAdapterTrajectory
from pydantic_ai_gepa.adapter import SharedReflectiveDataset
from pydantic_ai_gepa.gepa_graph.models import CandidateMap, ComponentValue
from pydantic_ai_gepa.evaluation_models import EvaluationBatch
from pydantic_ai_gepa.types import (
    MetricResult,
    RolloutOutput,
)


def _candidate_map(text: str) -> CandidateMap:
    return {"instructions": ComponentValue(name="instructions", text=text)}


def _make_adapter() -> AgentAdapter[str, dict[str, Any]]:
    agent = Agent(TestModel(), instructions="Base instructions")

    def metric(
        case: Case[str, str, dict[str, Any]],
        output: RolloutOutput[Any],
    ) -> MetricResult:
        return MetricResult(score=0.5, feedback="feedback")

    return AgentAdapter(agent=agent, metric=metric)


def _make_trajectory(
    *,
    prompt: str,
    response: str,
    feedback: str | None,
) -> AgentAdapterTrajectory:
    return AgentAdapterTrajectory(
        messages=[
            ModelRequest(
                parts=[UserPromptPart(content=prompt)],
                instructions="Base instructions",
            ),
            ModelResponse(parts=[TextPart(content=response)]),
        ],
        final_output=response,
        instructions="Base instructions",
        metric_feedback=feedback,
        usage={"requests": 1},
    )


def _build_batch() -> EvaluationBatch:
    trajectories = [
        _make_trajectory(prompt="Hello", response="Hi!", feedback="Detailed feedback"),
        _make_trajectory(prompt="Bad", response="Err", feedback=None),
    ]
    outputs = [
        RolloutOutput.from_success("Hi!"),
        RolloutOutput.from_error(ValueError("boom")),
    ]
    scores = [0.9, 0.2]
    return EvaluationBatch(
        outputs=outputs,
        scores=scores,
        trajectories=trajectories,
    )


def test_make_reflective_dataset_includes_feedback_and_errors() -> None:
    adapter = _make_adapter()
    dataset = adapter.make_reflective_dataset(
        candidate=_candidate_map("Base instructions"),
        eval_batch=_build_batch(),
        components_to_update=["instructions", "tools"],
    )
    assert isinstance(dataset, SharedReflectiveDataset)

    records = dataset.records
    assert len(records) == 2
    assert records[0]["feedback"] == "Detailed feedback"
    assert records[1]["feedback"].startswith("Poor response (score: 0.20)")
    assert records[1]["error_message"] == "boom"
    assert records[0]["instructions"] == "Base instructions"


def test_make_reflective_dataset_returns_full_records() -> None:
    adapter = _make_adapter()
    batch = _build_batch()
    dataset = adapter.make_reflective_dataset(
        candidate=_candidate_map("seed"),
        eval_batch=batch,
        components_to_update=["instructions"],
    )

    assert isinstance(dataset, SharedReflectiveDataset)
    assert len(dataset.records) == len(batch.outputs)


def test_make_reflective_dataset_handles_missing_trajectories() -> None:
    adapter = _make_adapter()
    batch = EvaluationBatch(outputs=[RolloutOutput.from_success("ok")], scores=[0.5])
    dataset = adapter.make_reflective_dataset(
        candidate=_candidate_map("seed"),
        eval_batch=batch,
        components_to_update=["instructions"],
    )
    assert isinstance(dataset, SharedReflectiveDataset)
    assert dataset.records == []


def test_reflective_record_includes_output_tool_metadata() -> None:
    """Ensure serialized records expose output tools for reflection prompts."""

    tool_def = ToolDefinition(
        name="final_result",
        description="Return the final structured answer",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
            },
            "required": ["answer"],
        },
        kind="output",
    )
    trajectory = AgentAdapterTrajectory(
        messages=[
            ModelRequest(
                parts=[UserPromptPart(content="Hi")], instructions="Base instructions"
            ),
            ModelResponse(parts=[TextPart(content="Hello")]),
        ],
        final_output="Hello",
        instructions="Base instructions",
        output_tools=[tool_def],
    )

    record = trajectory.to_reflective_record()
    tools = record.get("tools")
    assert tools is not None
    matching = [
        tool for tool in tools if tool.get("function", {}).get("name") == "final_result"
    ]
    assert matching, "Expected serialized output tool merged into tools list"
    assert matching[0]["kind"] == "output"


@pytest.mark.asyncio
async def test_run_with_trace_returns_trajectory_on_usage_limit() -> None:
    agent = Agent(TestModel(), instructions="Base instructions")

    adapter = AgentAdapter(
        agent=agent,
        metric=lambda case, output: MetricResult(score=0.0, feedback="unused"),
        agent_usage_limits=UsageLimits(request_limit=0),
    )

    case = Case(name="usage-limit-case", inputs="Hello", metadata={})

    trajectory, output = await adapter._run_with_trace(case, 0, candidate=None)
    assert trajectory is not None
    assert output.success is False
    assert trajectory.error is not None
    assert trajectory.messages, "usage-limit trajectories should capture prompts"
    assert any(
        isinstance(message, ModelRequest) for message in trajectory.messages
    ), "expected to capture the synthesized user request"
    record = trajectory.to_reflective_record()
    assert record["messages"], "reflective record should include serialized messages"
    assert "request_limit" in (record["error"] or "")

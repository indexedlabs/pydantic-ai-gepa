"""Integration tests for pydantic-ai-gepa."""

from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
import time_machine

import pydantic_ai_gepa.adapters.agent_adapter as agent_adapter_module
from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapter
from pydantic_ai_gepa.components import (
    extract_seed_candidate,
    get_component_names,
)
from pydantic_ai_gepa.signature_agent import SignatureAgent
from pydantic_ai_gepa.evaluation import evaluate_candidate_dataset
from pydantic_ai_gepa.adapter import SharedReflectiveDataset
from pydantic_ai_gepa.types import (
    MetricResult,
    RolloutOutput,
)
from pydantic_ai_gepa.gepa_graph.models import ComponentValue


def test_extract_seed_candidate():
    """Test extracting prompts from an agent."""
    agent = Agent(
        TestModel(),
        instructions="Be helpful",
    )

    candidate = extract_seed_candidate(agent)

    assert candidate["instructions"].text == "Be helpful"
    assert len(candidate) == 1


@pytest.mark.asyncio
async def test_evaluate_candidate_dataset_helper() -> None:
    base_agent = Agent(TestModel(custom_output_text="alpha"), instructions="Seed")

    dataset = Dataset(
        name="candidate-evaluation",
        cases=[
            Case(name="case-0", inputs={"text": "a"}, expected_output="alpha"),
            Case(name="case-1", inputs={"text": "b"}, expected_output="beta"),
        ],
    )

    class InputModel(BaseModel):
        text: str

    def metric(
        case: Case[InputModel, str, dict[str, str] | None],
        output: RolloutOutput[Any],
    ) -> MetricResult:
        predicted = (
            str(output.result).strip()
            if output.success and output.result is not None
            else ""
        )
        expected = str(case.expected_output or "")
        score = 1.0 if predicted == expected else 0.0
        return MetricResult(score=score, feedback="match" if score else "mismatch")

    agent = SignatureAgent(base_agent, input_type=InputModel, optimize_tools=False)

    records = await evaluate_candidate_dataset(
        agent=agent,
        metric=metric,
        input_type=InputModel,
        dataset=dataset,
        candidate={
            "instructions": ComponentValue(
                name="instructions", text="Always answer alpha"
            )
        },
        concurrency=2,
    )

    assert len(records) == 2
    scores = {record.case_id: record.score for record in records}
    assert scores["case-0"] == pytest.approx(1.0)
    assert scores["case-1"] == pytest.approx(0.0)


def test_get_component_names():
    """Test getting optimizable component names."""
    agent = Agent(
        TestModel(),
        instructions="Instructions",
    )

    components = get_component_names(agent)

    assert "instructions" in components
    assert len(components) == 1


@pytest.mark.asyncio
async def test_process_case():
    """Test processing a single case."""
    agent = Agent(
        TestModel(custom_output_text="Test response"), instructions="Be helpful"
    )

    def metric(
        case: Case[str, str, dict[str, Any]], output: RolloutOutput[Any]
    ) -> MetricResult:
        if output.success:
            return MetricResult(score=0.8, feedback="Good")
        return MetricResult(score=0.0, feedback="Failed")

    adapter = AgentAdapter(agent=agent, metric=metric)

    case = Case(name="test-4", inputs="Hello", metadata={})
    result = await adapter.process_case(case, 0, capture_traces=False)

    assert "output" in result
    assert "score" in result
    assert result["output"].success is True
    assert result["score"] == 0.8
    assert "trajectory" not in result

    # Test with traces
    result_with_trace = await adapter.process_case(
        case,
        0,
        capture_traces=True,
    )

    assert "output" in result_with_trace
    assert "score" in result_with_trace
    assert "trajectory" in result_with_trace
    assert result_with_trace["output"].success is True
    assert result_with_trace["score"] == 0.8
    assert result_with_trace["trajectory"].final_output == "Test response"


@pytest.mark.asyncio
async def test_process_case_captures_messages_on_tool_error():
    """Ensure traces include prompt/response history even when tools fail."""
    agent = Agent(TestModel(), instructions="Be helpful")

    @agent.tool
    async def broken_tool(ctx, code: str) -> str:
        raise ValueError("boom")

    def metric(
        case: Case[str, str, dict[str, Any]], output: RolloutOutput[Any]
    ) -> MetricResult:
        return MetricResult(score=0.0, feedback="failed")

    adapter = AgentAdapter(agent=agent, metric=metric)
    case = Case(name="tool-error", inputs="Hello", metadata={})

    result = await adapter.process_case(case, 0, capture_traces=True)

    assert result["output"].success is False
    assert result["output"].error_kind == "tool"
    trajectory = result["trajectory"]
    assert trajectory.messages, "expected captured messages for debugging"
    assert any(isinstance(message, ModelRequest) for message in trajectory.messages)


@pytest.mark.asyncio
async def test_process_case_skips_system_error_trajectory(monkeypatch):
    """System/library errors should not be surfaced to reflection trajectories."""
    agent = Agent(TestModel(), instructions="Be helpful")

    @agent.tool
    async def broken_tool(ctx, code: str) -> str:
        raise ValueError("boom")

    def metric(
        case: Case[str, str, dict[str, Any]], output: RolloutOutput[Any]
    ) -> MetricResult:
        return MetricResult(score=0.0, feedback="failed")

    adapter = AgentAdapter(agent=agent, metric=metric)
    case = Case(name="system-error", inputs="Hello", metadata={})

    monkeypatch.setattr(
        agent_adapter_module, "_classify_exception", lambda exc: "system"
    )

    result = await adapter.process_case(case, 0, capture_traces=True)

    assert result["output"].success is False
    assert result["output"].error_kind == "system"
    assert "trajectory" not in result


@pytest.mark.asyncio
@time_machine.travel("2023-01-01 08:00:00+00:00", tick=False)
async def test_make_reflective_dataset():
    """Test making a reflective dataset."""
    agent = Agent(
        TestModel(custom_output_text="Test response"), instructions="Be helpful"
    )

    def metric(
        case: Case[str, str, dict[str, Any]], output: RolloutOutput[Any]
    ) -> MetricResult:
        if output.success:
            return MetricResult(score=0.8, feedback="Good")
        return MetricResult(score=0.0, feedback="Failed")

    adapter = AgentAdapter(agent=agent, metric=metric)
    candidate = extract_seed_candidate(agent)

    case = Case(name="test-4", inputs="Hello", metadata={})
    result = await adapter.evaluate([case], candidate, capture_traces=True)

    reflective_dataset = adapter.make_reflective_dataset(
        candidate=candidate,
        eval_batch=result,
        components_to_update=["instructions"],
    )
    assert isinstance(reflective_dataset, SharedReflectiveDataset)
    trace_id = reflective_dataset.records[0].pop("trace_id")
    root_span_id = reflective_dataset.records[0].pop("root_span_id")
    assert isinstance(trace_id, str) and len(trace_id) == 32
    assert isinstance(root_span_id, str) and len(root_span_id) == 16
    assert reflective_dataset.records[0].pop("trace_completeness") == "root-only"
    assert reflective_dataset == snapshot(
        SharedReflectiveDataset(
            records=[
                {
                    "user_prompt": "Hello",
                    "assistant_response": "Test response",
                    "error": None,
                    "messages": [
                        {
                            "kind": "request",
                            "parts": [
                                {
                                    "type": "user_prompt",
                                    "role": "user",
                                    "content": "Hello",
                                    "timestamp": "2023-01-01T08:00:00+00:00",
                                }
                            ],
                            "instructions": "Be helpful",
                        },
                        {
                            "kind": "response",
                            "model_name": "test",
                            "provider_name": "test",
                            "timestamp": "2023-01-01T08:00:00+00:00",
                            "parts": [
                                {
                                    "type": "text",
                                    "role": "assistant",
                                    "content": "Test response",
                                }
                            ],
                            "usage": {
                                "input_tokens": 51,
                                "cache_write_tokens": 0,
                                "cache_read_tokens": 0,
                                "output_tokens": 2,
                                "input_audio_tokens": 0,
                                "cache_audio_read_tokens": 0,
                                "output_audio_tokens": 0,
                                "details": {},
                                "cost": None,
                            },
                        },
                    ],
                    "run_usage": {
                        "input_tokens": 51,
                        "cache_write_tokens": 0,
                        "cache_read_tokens": 0,
                        "output_tokens": 2,
                        "input_audio_tokens": 0,
                        "cache_audio_read_tokens": 0,
                        "output_audio_tokens": 0,
                        "details": {},
                        "cost": None,
                        "requests": 1,
                        "tool_calls": 0,
                    },
                    "score": 0.8,
                    "success": True,
                    "instructions": "Be helpful",
                    "feedback": "Good",
                }
            ]
        )
    )

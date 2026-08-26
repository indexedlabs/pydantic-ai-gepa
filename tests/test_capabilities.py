"""Regression coverage for Pydantic AI v2 capability integration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from pydantic_ai_gepa import SignatureAgent
from pydantic_ai_gepa.tool_components import (
    GepaCandidateCapability,
    ToolOptimizationManager,
    get_or_create_tool_optimizer,
)
from pydantic_ai_gepa.trace_capabilities import (
    GepaTraceCollector,
    GepaTraceContext,
)


class SignatureInput(BaseModel):
    """Answer a structured question."""

    question: str = Field(description="Question to answer.")


def _tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="lookup",
        description="Look up a value.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Value to look up.",
                }
            },
            "required": ["query"],
        },
    )


@pytest.mark.asyncio
async def test_candidate_capabilities_are_isolated_across_concurrent_runs() -> None:
    agent = Agent(TestModel(), name="candidate-isolation")
    manager = ToolOptimizationManager(agent)
    original = _tool_definition()
    first = GepaCandidateCapability(
        candidate={"tool:lookup:description": "First run lookup."},
        tool_optimizer=manager,
    )
    second = GepaCandidateCapability(
        candidate={"tool:lookup:description": "Second run lookup."},
        tool_optimizer=manager,
    )

    async def prepare(capability: GepaCandidateCapability) -> ToolDefinition:
        definitions = await capability.prepare_tools(
            cast(RunContext[Any], None),
            [original],
        )
        await asyncio.sleep(0)
        return definitions[0]

    first_definition, second_definition = await asyncio.gather(
        prepare(first), prepare(second)
    )

    assert first_definition.description == "First run lookup."
    assert second_definition.description == "Second run lookup."
    assert original.description == "Look up a value."
    assert original.parameters_json_schema["properties"]["query"]["description"] == (
        "Value to look up."
    )


def test_optimizer_setup_does_not_mutate_pydantic_ai_capability_internals() -> None:
    agent = Agent(TestModel(), name="private-boundary")
    root_capability = agent._root_capability

    get_or_create_tool_optimizer(agent)

    assert agent._root_capability is root_capability
    assert not hasattr(agent, "_gepa_tool_prepare_wrapper")
    assert not hasattr(agent, "_prepare_tools")
    assert not hasattr(agent, "_prepare_output_tools")


@dataclass
class RecordingCapability(AbstractCapability[Any]):
    observations: list[dict[str, Any]] = field(default_factory=list)

    async def before_run(self, ctx: RunContext[Any]) -> None:
        self.observations.append(
            {
                "run_id": ctx.run_id,
                "conversation_id": ctx.conversation_id,
                "metadata": ctx.metadata,
            }
        )


@pytest.mark.asyncio
async def test_signature_agent_composes_caller_capability_and_v2_run_context() -> None:
    agent = Agent(
        TestModel(custom_output_text="answer"),
        name="signature-v2",
    )
    signature_agent = SignatureAgent(agent, input_type=SignatureInput)
    recording = RecordingCapability()

    result = await signature_agent.run_signature(
        SignatureInput(question="Why?"),
        run_id="run-123",
        conversation_id="conversation-456",
        metadata={"source": "test"},
        retries={"tools": 3, "output": 4},
        instructions="Per-run guidance.",
        capabilities=[recording],
    )

    assert recording.observations == [
        {
            "run_id": "run-123",
            "conversation_id": "conversation-456",
            "metadata": {"source": "test"},
        }
    ]
    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert request.instructions is not None
    assert "Answer a structured question." in request.instructions
    assert "Per-run guidance." in request.instructions


@pytest.mark.asyncio
async def test_trace_collector_correlates_participating_nested_agent() -> None:
    collector = GepaTraceCollector(include_content=False)
    child = Agent(
        TestModel(custom_output_text="child result"),
        name="child",
        capabilities=[collector.nested_instrumentation()],
    )
    parent = Agent(TestModel(call_tools=["delegate"]), name="parent")

    @parent.tool_plain
    async def delegate() -> str:
        return (await child.run("child prompt")).output

    context = GepaTraceContext(
        gepa_run_id="gepa-run",
        candidate_id="candidate-a",
        case_id="case-a",
        agent_run_id="agent-run-a",
        expected_nested_agents=frozenset({"child"}),
    )
    await parent.run(
        "parent prompt",
        run_id=context.agent_run_id,
        capabilities=collector.capabilities(context),
    )

    spans = collector.spans_for(context)
    assert context.trace_id is not None
    assert context.root_span_id is not None
    assert context.completeness == "full"
    assert {"parent", "child"}.issubset(context.seen_agent_names)
    assert all(
        span.context is not None
        and format(span.context.trace_id, "032x") == context.trace_id
        for span in spans
    )


@pytest.mark.asyncio
async def test_trace_collector_reports_uninstrumented_nested_agent() -> None:
    collector = GepaTraceCollector(include_content=False)
    child = Agent(TestModel(custom_output_text="child result"), name="child")
    parent = Agent(TestModel(call_tools=["delegate"]), name="parent")

    @parent.tool_plain
    async def delegate() -> str:
        return (await child.run("child prompt")).output

    context = GepaTraceContext(
        gepa_run_id="gepa-run",
        candidate_id="candidate-b",
        case_id="case-b",
        agent_run_id="agent-run-b",
        expected_nested_agents=frozenset({"child"}),
    )
    await parent.run(
        "parent prompt",
        run_id=context.agent_run_id,
        capabilities=collector.capabilities(context),
    )

    collector.spans_for(context)
    assert context.completeness == "root-only"
    assert "parent" in context.seen_agent_names
    assert "child" not in context.seen_agent_names

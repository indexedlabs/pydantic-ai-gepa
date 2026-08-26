"""Regression coverage for Pydantic AI v2 capability integration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from pydantic_ai_gepa import GepaOptimizationResult, SignatureAgent
from pydantic_ai_gepa.gepa_graph.models import ComponentValue
from pydantic_ai_gepa.tool_components import (
    GepaCandidateCapability,
    ToolOptimizationManager,
    get_or_create_tool_optimizer,
)
from pydantic_ai_gepa.trace_capabilities import (
    GepaTraceCollector,
    GepaTraceContext,
    drain_spans_by_trace_id,
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


@pytest.mark.asyncio
async def test_signature_candidate_capabilities_are_isolated_in_concurrent_runs() -> (
    None
):
    observations: dict[str, str | None] = {}

    async def capture_model(messages: list[Any], info: Any) -> ModelResponse:
        await asyncio.sleep(0)
        message_text = str(messages)
        case_name = "first" if "first" in message_text else "second"
        [tool] = info.function_tools
        observations[case_name] = tool.description
        return ModelResponse(parts=[TextPart(content="answer")])

    agent = Agent(FunctionModel(capture_model), name="concurrent-signature")

    @agent.tool_plain
    def lookup(query: str) -> str:
        """Look up a value."""
        return query

    signature_agent = SignatureAgent(
        agent,
        input_type=SignatureInput,
        output_type=str,
        optimize_tools=True,
    )
    await asyncio.gather(
        signature_agent.run_signature(
            SignatureInput(question="first"),
            candidate={"tool:lookup:description": "First run lookup."},
        ),
        signature_agent.run_signature(
            SignatureInput(question="second"),
            candidate={"tool:lookup:description": "Second run lookup."},
        ),
    )

    assert observations == {
        "first": "First run lookup.",
        "second": "Second run lookup.",
    }


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


@dataclass
class InstructionCapability(AbstractCapability[Any]):
    instruction: str

    def get_instructions(self):
        return [self.instruction]


@dataclass
class CallableInstructionCapability(AbstractCapability[Any]):
    instruction: Any

    def get_instructions(self):
        return [self.instruction]


async def instruction_capability_factory(
    ctx: RunContext[Any],
) -> AbstractCapability[Any]:
    return InstructionCapability("Factory capability guidance.")


def _optimization_result(candidate: dict[str, str]) -> GepaOptimizationResult:
    components = {
        name: ComponentValue(name=name, text=text) for name, text in candidate.items()
    }
    return GepaOptimizationResult(
        best_candidate=components,
        best_score=1.0,
        original_candidate={},
        original_score=0.0,
        num_iterations=1,
        num_metric_calls=1,
    )


@pytest.mark.asyncio
async def test_apply_best_yields_agent_with_candidate_capability() -> None:
    model = TestModel(custom_output_text="answer")
    agent = Agent(
        model,
        instructions="Seed instructions.",
        capabilities=[
            InstructionCapability("Capability guidance."),
            CallableInstructionCapability(lambda ctx: "Callable guidance."),
        ],
        name="apply-best",
    )

    @agent.tool_plain
    def lookup(query: str) -> str:
        """Look up the original value."""
        return query

    get_or_create_tool_optimizer(agent)
    result = _optimization_result(
        {
            "instructions": "Optimized instructions.",
            "tool:lookup:description": "Look up the optimized value.",
        }
    )

    with result.apply_best(agent) as optimized_agent:
        run_result = await optimized_agent.run(
            "question",
            instructions="Caller guidance.",
            capabilities=[
                InstructionCapability("Run capability guidance."),
                instruction_capability_factory,
            ],
        )

    request = run_result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert request.instructions is not None
    assert request.instructions.count("Optimized instructions.") == 1
    assert request.instructions.count("Capability guidance.") == 1
    assert request.instructions.count("Run capability guidance.") == 1
    assert request.instructions.count("Callable guidance.") == 1
    assert request.instructions.count("Factory capability guidance.") == 1
    assert request.instructions.count("Caller guidance.") == 1
    assert model.last_model_request_parameters is not None
    [tool] = model.last_model_request_parameters.function_tools
    assert tool.description == "Look up the optimized value."


@pytest.mark.asyncio
async def test_apply_best_to_signature_retains_all_instruction_sources() -> None:
    agent = Agent(
        TestModel(custom_output_text="answer"),
        instructions="Seed instructions.",
        capabilities=[
            InstructionCapability("Capability guidance."),
            CallableInstructionCapability(lambda ctx: "Callable guidance."),
        ],
        name="signature-apply-best",
    )
    signature_agent = SignatureAgent(agent, input_type=SignatureInput, output_type=str)
    result = _optimization_result({"instructions": "Optimized instructions."})

    with result.apply_best_to(
        agent=signature_agent, input_type=SignatureInput
    ) as applied:
        assert applied.wrapped is signature_agent
        run_result = await applied.run_signature(
            SignatureInput(question="Why?"),
            instructions="Caller guidance.",
            capabilities=[
                InstructionCapability("Run capability guidance."),
                instruction_capability_factory,
            ],
        )

    request = run_result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert request.instructions is not None
    assert request.instructions.count("Optimized instructions.") == 1
    assert request.instructions.count("Capability guidance.") == 1
    assert request.instructions.count("Run capability guidance.") == 1
    assert request.instructions.count("Callable guidance.") == 1
    assert request.instructions.count("Factory capability guidance.") == 1
    assert request.instructions.count("Answer a structured question.") == 1
    assert request.instructions.count("Caller guidance.") == 1
    assert applied.input_spec is signature_agent.input_spec
    assert callable(applied.run_signature_stream)


@pytest.mark.asyncio
async def test_tool_only_candidate_preserves_seed_and_capability_instructions() -> None:
    model = TestModel(custom_output_text="answer")
    agent = Agent(
        model,
        instructions="Seed instructions.",
        capabilities=[InstructionCapability("Capability guidance.")],
    )

    @agent.tool_plain
    def lookup(query: str) -> str:
        """Original lookup."""
        return query

    get_or_create_tool_optimizer(agent)
    result = _optimization_result({"tool:lookup:description": "Optimized lookup."})

    with result.apply_best(agent) as applied:
        run_result = await applied.run("question")

    request = run_result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert request.instructions == "Seed instructions.\nCapability guidance."
    assert model.last_model_request_parameters is not None
    [tool] = model.last_model_request_parameters.function_tools
    assert tool.description == "Optimized lookup."


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
    drained = drain_spans_by_trace_id(collector.exporter, {context.trace_id})
    assert drained == spans
    assert collector.exporter.get_finished_spans() == ()


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

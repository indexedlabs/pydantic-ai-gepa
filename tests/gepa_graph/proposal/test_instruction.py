from __future__ import annotations

from typing import Any

from inline_snapshot import snapshot
import pytest
import time_machine
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_evals import Case
from pydantic_ai.models.function import FunctionModel

from pydantic_ai_gepa.adapter import (
    ComponentReflectiveDataset,
    SharedReflectiveDataset,
)
from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapter
from pydantic_ai_gepa.gepa_graph.models import CandidateProgram, ComponentValue
from pydantic_ai_gepa.gepa_graph.proposal import InstructionProposalGenerator
from pydantic_ai_gepa.types import MetricResult, RolloutOutput


def _make_candidate() -> CandidateProgram:
    return CandidateProgram(
        idx=0,
        components={
            "instructions": ComponentValue(
                name="instructions", text="Seed instructions"
            ),
            "tools": ComponentValue(name="tools", text="Seed tools"),
        },
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )


def _make_candidate_with_catalog_tools() -> CandidateProgram:
    return CandidateProgram(
        idx=1,
        components={
            "instructions": ComponentValue(name="instructions", text="Catalog seed"),
            "tool:final_result:description": ComponentValue(
                name="tool:final_result:description",
                text="Return the final structured answer including reasoning summary.",
            ),
            "tool:final_result:param:answer": ComponentValue(
                name="tool:final_result:param:answer",
                text="Final user-facing answer as a concise string.",
            ),
            "tool:final_result:param:analysis": ComponentValue(
                name="tool:final_result:param:analysis",
                text="Short bullet summary of the reasoning steps used.",
            ),
        },
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )


def _make_reflective_record() -> dict[str, Any]:
    """Create a realistic reflective record with full message history."""
    return {
        "messages": [
            {
                "kind": "request",
                "parts": [
                    {
                        "type": "system_prompt",
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"type": "user_prompt", "role": "user", "content": "Hello"},
                ],
            },
            {
                "kind": "response",
                "model_name": "openai:gpt-4",
                "parts": [
                    {"type": "text", "role": "assistant", "content": "Hi"},
                ],
            },
        ],
        "score": 0.5,
        "success": True,
        "feedback": "Needs more detail",
    }


@pytest.mark.asyncio
async def test_llm_generator_updates_components() -> None:
    candidate = _make_candidate()
    reflective_data = ComponentReflectiveDataset(
        records_by_component={
            "instructions": [_make_reflective_record()],
            "tools": [_make_reflective_record()],
        }
    )

    prompts: list[str] = []

    async def fake_model(messages, agent_info):
        prompt_parts = []
        for p in messages[-1].parts:
            if hasattr(p, "content"):
                if isinstance(p.content, str):
                    prompt_parts.append(p.content)
                elif isinstance(p.content, list):
                    for item in p.content:
                        if isinstance(item, str):
                            prompt_parts.append(item)
        prompt = "".join(prompt_parts)
        prompts.append(prompt)
        content = """{
            "reasoning": {
                "pattern_discovery": "Some things worked",
                "creative_hypothesis": "Some things didn't work",
                "experimental_approach": "Need to improve clarity"
            },
            "updated_components": [
                {"component_name": "instructions", "optimized_value": "Improved instructions"},
                {"component_name": "tools", "optimized_value": "Better tools"}
            ]
        }"""
        return ModelResponse(parts=[TextPart(content=content)])

    generator = InstructionProposalGenerator()
    model = FunctionModel(function=fake_model)
    result = await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions", "tools"],
        model=model,
    )

    assert result.texts == {
        "instructions": "Improved instructions",
        "tools": "Better tools",
    }
    assert "## Components to update" in prompts[-1]
    assert "=== start component: `instructions` current value ===" in prompts[-1]
    assert "=== start component: `tools` current value ===" in prompts[-1]


@pytest.mark.asyncio
async def test_llm_generator_returns_metadata_when_enabled() -> None:
    candidate = _make_candidate()
    reflective_data = ComponentReflectiveDataset(
        records_by_component={
            "instructions": [_make_reflective_record()],
        }
    )

    async def fake_model(messages, agent_info):
        content = """{
            "reasoning": {
                "pattern_discovery": "Repeated boundary errors",
                "creative_hypothesis": "Teach explicit range mapping",
                "experimental_approach": "Add a checklist with translations",
                "edge_insight": "Still mislabeling exclusive ranges",
                "success_checkpoint": "Zero range mistakes on validation minibatch",
                "evolution_moves": ["Checklist", "Edge Reasoning: ranges"]
            },
            "updated_components": [
                {"component_name": "instructions", "optimized_value": "Improved instructions"}
            ]
        }"""
        return ModelResponse(parts=[TextPart(content=content)])

    generator = InstructionProposalGenerator(include_hypothesis_metadata=True)
    model = FunctionModel(function=fake_model)
    result = await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions"],
        model=model,
    )

    assert result.texts["instructions"] == "Improved instructions"
    metadata = result.component_metadata["instructions"]
    assert metadata["hypothesis"] == "Teach explicit range mapping"
    assert metadata["pattern"] == "Repeated boundary errors"
    assert metadata["approach"].startswith("Add a checklist")
    assert metadata["edge_insight"] == "Still mislabeling exclusive ranges"
    assert metadata["checkpoint"] == "Zero range mistakes on validation minibatch"
    assert metadata["moves"] == ["Checklist", "Edge Reasoning: ranges"]


@pytest.mark.asyncio
async def test_llm_generator_uses_shared_dataset_once() -> None:
    candidate = _make_candidate()
    reflective_data = SharedReflectiveDataset(
        records=[
            {
                "messages": [
                    {
                        "kind": "request",
                        "parts": [{"type": "user_prompt", "content": "u1"}],
                    }
                ],
                "score": 0.5,
                "user_prompt": "trace1",
            },
            {
                "messages": [
                    {
                        "kind": "request",
                        "parts": [{"type": "user_prompt", "content": "u2"}],
                    }
                ],
                "score": 0.9,
                "user_prompt": "trace2",
            },
        ]
    )

    captured_prompt: list[str] = []

    async def fake_model(messages, agent_info):
        prompt_parts = []
        for p in messages[-1].parts:
            if hasattr(p, "content"):
                if isinstance(p.content, str):
                    prompt_parts.append(p.content)
                elif isinstance(p.content, list):
                    for item in p.content:
                        if isinstance(item, str):
                            prompt_parts.append(item)
        prompt = "".join(prompt_parts)
        captured_prompt.append(prompt)
        content = """{
            "reasoning": {
                "pattern_discovery": "Patterns",
                "creative_hypothesis": "Hyp",
                "experimental_approach": "Approach"
            },
            "updated_components": [
                {"component_name": "instructions", "optimized_value": "Improved instructions"},
                {"component_name": "tools", "optimized_value": "Improved tools"}
            ]
        }"""
        return ModelResponse(parts=[TextPart(content=content)])

    generator = InstructionProposalGenerator()
    model = FunctionModel(function=fake_model)
    await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions", "tools"],
        model=model,
    )

    prompt = captured_prompt[-1]
    assert prompt == snapshot("""\
# Creative Instruction Design Challenge

Transform the student agent's performance through innovative instruction formats.

## Context
- A student agent has been running with the configuration shown below
- We've collected traces from real production runs
- Your job is to improve specific components so the student agent performs better

---

## Full student agent configuration

This is the complete configuration the student agent was running with:

=== start component: `instructions` given to student ===
Seed instructions
=== end ===

=== start component: `tools` given to student ===
Seed tools
=== end ===

---

## Production traces from student agent runs

Each trace contains:
- `messages`: Full conversation history with system prompts, user inputs, assistant responses, tool calls, and tool returns
- `tools`: Tool definitions that were available (if any)
- `score`: Performance score (0.0-1.0, higher is better)
- `success`: Whether the run completed successfully
- `feedback`: Evaluator feedback on this specific run


## Execution Traces Available for Analysis

2 traces available from the execution: 0 succeeded, 2 failed.
The traces are stored on disk in structured JSONL format.
You must use the `run_trace_analysis(python_script: str)` tool to write and execute python map-reduce scripts to analyze these traces within the unbounded Ouros Data Plane.
You may also use `analyze_trace_with_llm(trace_id, prompt)` to spawn a Recursive Language Model (RLM) sub-agent to deeply inspect specific traces for semantic failures.


### Analysis guidance
- Use `run_trace_analysis` to aggregate errors or find common failure modes.
- Use `analyze_trace_with_llm` to understand *why* a specific trace failed if the python analysis is insufficient.
- What failure patterns repeat across runs?
- Are components misaligned (e.g., instructions referencing tools that don't exist)?
- Which successful patterns should be preserved or extended?
- What domain knowledge should be codified in the prompts?
- Are there patterns of inefficient tool usage (redundant calls, speculative calls, lack of planning)?
- How can prompts guide the student to gather what's needed in fewer, well-targeted tool calls?

---

## Components to update

Rewrite these components as a coordinated update based on the evidence above:

=== start component: `instructions` current value ===
Seed instructions
=== end ===

=== start component: `tools` current value ===
Seed tools
=== end ===
""")


@pytest.mark.asyncio
async def test_llm_generator_selection_mode_lists_components() -> None:
    candidate = CandidateProgram(
        idx=0,
        components={
            "instructions": ComponentValue(
                name="instructions",
                text="Seed instructions",
            ),
            "signature:Input:text:desc": ComponentValue(
                name="signature:Input:text:desc",
                text="Describe the text field.",
            ),
            "skill:roman-numerals:body": ComponentValue(
                name="skill:roman-numerals:body",
                text="(big skill body omitted in selection mode)",
            ),
        },
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )
    reflective_data = SharedReflectiveDataset(records=[_make_reflective_record()])

    captured_prompt: list[str] = []

    async def fake_model(messages, agent_info):
        prompt_parts = []
        for p in messages[-1].parts:
            if hasattr(p, "content"):
                if isinstance(p.content, str):
                    prompt_parts.append(p.content)
                elif isinstance(p.content, list):
                    for item in p.content:
                        if isinstance(item, str):
                            prompt_parts.append(item)
        prompt = "".join(prompt_parts)
        captured_prompt.append(prompt)
        content = """{
            "reasoning": {
                "pattern_discovery": "Patterns",
                "creative_hypothesis": "Hyp",
                "experimental_approach": "Approach"
            },
            "updated_components": [
                {"component_name": "instructions", "optimized_value": "Improved instructions"}
            ]
        }"""
        return ModelResponse(parts=[TextPart(content=content)])

    generator = InstructionProposalGenerator()
    model = FunctionModel(function=fake_model)
    await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=None,
        model=model,
    )

    prompt = captured_prompt[-1]
    assert "## Optimizable components (current candidate)" in prompt
    assert "skill:roman-numerals:body" in prompt
    assert (
        "=== start component: `skill:roman-numerals:body` given to student ==="
        not in prompt
    )


@pytest.mark.asyncio
async def test_llm_generator_skips_components_without_records() -> None:
    candidate = _make_candidate()
    reflective_data = ComponentReflectiveDataset(
        records_by_component={
            "instructions": [_make_reflective_record()],
            "tools": [],
        }
    )

    async def fake_model(messages, agent_info):
        prompt_parts = []
        for p in messages[-1].parts:
            if hasattr(p, "content"):
                if isinstance(p.content, str):
                    prompt_parts.append(p.content)
                elif isinstance(p.content, list):
                    for item in p.content:
                        if isinstance(item, str):
                            prompt_parts.append(item)
        prompt = "".join(prompt_parts)
        # instructions should be in "Components to update" section
        assert "=== start component: `instructions` current value ===" in prompt
        # tools should NOT be in "Components to update" (no records), but IS in "given to student"
        assert "=== start component: `tools` current value ===" not in prompt
        content = """{
            "reasoning": {
                "pattern_discovery": "Some things worked",
                "creative_hypothesis": "Some things didn't work",
                "experimental_approach": "Need to improve clarity"
            },
            "updated_components": [
                {"component_name": "instructions", "optimized_value": "Improved instructions"}
            ]
        }"""
        return ModelResponse(parts=[TextPart(content=content)])

    generator = InstructionProposalGenerator()
    model = FunctionModel(function=fake_model)
    result = await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions", "tools"],
        model=model,
    )

    assert result.texts["instructions"] == "Improved instructions"
    assert "tools" not in result.texts


@pytest.mark.asyncio
async def test_llm_generator_returns_existing_text_on_agent_failure() -> None:
    candidate = _make_candidate()
    reflective_data = ComponentReflectiveDataset(
        records_by_component={
            "instructions": [_make_reflective_record()],
            "tools": [_make_reflective_record()],
        }
    )

    async def failing_model(messages, agent_info):
        raise RuntimeError("boom")

    generator = InstructionProposalGenerator()
    model = FunctionModel(function=failing_model)
    result = await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions", "tools"],
        model=model,
    )

    assert result.texts == {}


@pytest.mark.asyncio
async def test_prompt_includes_output_tool_details() -> None:
    candidate = _make_candidate()
    record = _make_reflective_record()
    record["output_tools"] = [
        {
            "type": "function",
            "function": {
                "name": "final_result",
                "description": "Return the final structured answer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "Final response"}
                    },
                    "required": ["answer"],
                },
            },
            "kind": "output",
        }
    ]
    reflective_data = SharedReflectiveDataset(records=[record])

    captured_prompt: str | None = None

    async def fake_model(messages, agent_info):
        nonlocal captured_prompt
        prompt_parts = []
        for p in messages[-1].parts:
            if hasattr(p, "content"):
                if isinstance(p.content, str):
                    prompt_parts.append(p.content)
                elif isinstance(p.content, list):
                    for item in p.content:
                        if isinstance(item, str):
                            prompt_parts.append(item)
        captured_prompt = "".join(prompt_parts)
        content = """{
            \"reasoning\": {
                \"pattern_discovery\": \"OK\",
                \"creative_hypothesis\": \"Needs a finale\",
                \"experimental_approach\": \"Explain how to finish\"
            },
            \"updated_components\": [
                {\"component_name\": \"instructions\", \"optimized_value\": \"Updated instructions\"}
            ]
        }"""
        return ModelResponse(parts=[TextPart(content=content)])

    generator = InstructionProposalGenerator()
    model = FunctionModel(function=fake_model)
    await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions"],
        model=model,
    )

    assert captured_prompt is not None
    assert "final_result" in captured_prompt
    assert 'kind": "output"' in captured_prompt
    assert "Teach the student to call the appropriate output tool" in captured_prompt


@pytest.mark.asyncio
async def test_catalog_tools_fall_back_when_no_reflective_records() -> None:
    candidate = _make_candidate_with_catalog_tools()
    reflective_data = SharedReflectiveDataset(records=[_make_reflective_record()])

    captured_prompt: str | None = None

    async def fake_model(messages, agent_info):
        nonlocal captured_prompt
        prompt_parts = []
        for p in messages[-1].parts:
            if hasattr(p, "content"):
                if isinstance(p.content, str):
                    prompt_parts.append(p.content)
                elif isinstance(p.content, list):
                    for item in p.content:
                        if isinstance(item, str):
                            prompt_parts.append(item)
        captured_prompt = "".join(prompt_parts)
        content = """{
            \"reasoning\": {
                \"pattern_discovery\": \"OK\",
                \"creative_hypothesis\": \"Needs finalization\",
                \"experimental_approach\": \"Be explicit\"
            },
            \"updated_components\": [
                {\"component_name\": \"instructions\", \"optimized_value\": \"Updated instructions\"}
            ]
        }"""
        return ModelResponse(parts=[TextPart(content=content)])

    generator = InstructionProposalGenerator()
    model = FunctionModel(function=fake_model)
    await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions"],
        model=model,
    )

    assert captured_prompt is not None
    assert "**Tools available to student" in captured_prompt
    assert "final_result" in captured_prompt
    assert '"kind": "output"' in captured_prompt


@pytest.mark.asyncio
async def test_prompt_includes_stored_hypothesis_metadata() -> None:
    candidate = _make_candidate()
    candidate.components["instructions"].metadata = {
        "pattern": "Boundary confusion",
        "hypothesis": "Spell out range rules",
        "approach": "Add checklist",
        "iteration": 4,
    }
    reflective_data = ComponentReflectiveDataset(
        records_by_component={"instructions": [_make_reflective_record()]}
    )

    captured_prompt: str | None = None

    async def fake_model(messages, agent_info):
        nonlocal captured_prompt
        prompt_parts = []
        for p in messages[-1].parts:
            if hasattr(p, "content"):
                if isinstance(p.content, str):
                    prompt_parts.append(p.content)
                elif isinstance(p.content, list):
                    for item in p.content:
                        if isinstance(item, str):
                            prompt_parts.append(item)
        captured_prompt = "".join(prompt_parts)
        content = """{
            "reasoning": {
                "pattern_discovery": "Boundary confusion",
                "creative_hypothesis": "Spell out range rules",
                "experimental_approach": "Add checklist"
            },
            "updated_components": [
                {"component_name": "instructions", "optimized_value": "Improved instructions"}
            ]
        }"""
        return ModelResponse(parts=[TextPart(content=content)])

    generator = InstructionProposalGenerator(include_hypothesis_metadata=True)
    model = FunctionModel(function=fake_model)
    await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions"],
        model=model,
    )

    assert captured_prompt is not None
    assert "## Stored hypotheses from previous reflections" in captured_prompt
    assert "Components: `instructions`" in captured_prompt
    assert "  - Hypothesis: Spell out range rules" in captured_prompt
    assert "  - Iteration: 4" in captured_prompt


@pytest.mark.asyncio
async def test_llm_generator_skips_entire_call_when_no_records() -> None:
    candidate = _make_candidate()
    reflective_data = ComponentReflectiveDataset(
        records_by_component={
            "instructions": [],
        }
    )
    call_count = 0

    async def tracking_model(messages, agent_info):
        nonlocal call_count
        call_count += 1
        return ModelResponse(parts=[TextPart(content="{}")])

    generator = InstructionProposalGenerator()
    model = FunctionModel(function=tracking_model)
    result = await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions"],
        model=model,
    )

    assert result.texts == {}
    assert call_count == 0


@pytest.mark.asyncio
async def test_llm_generator_handles_shared_dataset() -> None:
    candidate = _make_candidate()
    reflective_data = SharedReflectiveDataset(records=[_make_reflective_record()])
    prompts: list[str] = []

    async def fake_model(messages, agent_info):
        prompt_parts = []
        for p in messages[-1].parts:
            if hasattr(p, "content"):
                if isinstance(p.content, str):
                    prompt_parts.append(p.content)
                elif isinstance(p.content, list):
                    for item in p.content:
                        if isinstance(item, str):
                            prompt_parts.append(item)
        prompt = "".join(prompt_parts)
        prompts.append(prompt)
        content = """{
            "reasoning": {
                "pattern_discovery": "Some things worked",
                "creative_hypothesis": "Some things didn't work",
                "experimental_approach": "Need to improve clarity"
            },
            "updated_components": [
                {"component_name": "instructions", "optimized_value": "Improved instructions"},
                {"component_name": "tools", "optimized_value": "Improved tools"}
            ]
        }"""
        return ModelResponse(parts=[TextPart(content=content)])

    generator = InstructionProposalGenerator()
    model = FunctionModel(function=fake_model)
    result = await generator.propose_texts(
        candidate=candidate,
        reflective_data=reflective_data,
        components=["instructions", "tools"],
        model=model,
    )

    assert result.texts == snapshot(
        {"instructions": "Improved instructions", "tools": "Improved tools"}
    )
    prompt = prompts[-1]
    assert prompt == snapshot("""\
# Creative Instruction Design Challenge

Transform the student agent's performance through innovative instruction formats.

## Context
- A student agent has been running with the configuration shown below
- We've collected traces from real production runs
- Your job is to improve specific components so the student agent performs better

---

## Full student agent configuration

This is the complete configuration the student agent was running with:

=== start component: `instructions` given to student ===
Seed instructions
=== end ===

=== start component: `tools` given to student ===
Seed tools
=== end ===

---

## Production traces from student agent runs

Each trace contains:
- `messages`: Full conversation history with system prompts, user inputs, assistant responses, tool calls, and tool returns
- `tools`: Tool definitions that were available (if any)
- `score`: Performance score (0.0-1.0, higher is better)
- `success`: Whether the run completed successfully
- `feedback`: Evaluator feedback on this specific run


## Execution Traces Available for Analysis

1 traces available from the execution: 1 succeeded, 0 failed.
The traces are stored on disk in structured JSONL format.
You must use the `run_trace_analysis(python_script: str)` tool to write and execute python map-reduce scripts to analyze these traces within the unbounded Ouros Data Plane.
You may also use `analyze_trace_with_llm(trace_id, prompt)` to spawn a Recursive Language Model (RLM) sub-agent to deeply inspect specific traces for semantic failures.


### Analysis guidance
- Use `run_trace_analysis` to aggregate errors or find common failure modes.
- Use `analyze_trace_with_llm` to understand *why* a specific trace failed if the python analysis is insufficient.
- What failure patterns repeat across runs?
- Are components misaligned (e.g., instructions referencing tools that don't exist)?
- Which successful patterns should be preserved or extended?
- What domain knowledge should be codified in the prompts?
- Are there patterns of inefficient tool usage (redundant calls, speculative calls, lack of planning)?
- How can prompts guide the student to gather what's needed in fewer, well-targeted tool calls?

---

## Components to update

Rewrite these components as a coordinated update based on the evidence above:

=== start component: `instructions` current value ===
Seed instructions
=== end ===

=== start component: `tools` current value ===
Seed tools
=== end ===
""")


@time_machine.travel("2025-01-15 12:00:00 UTC", tick=False)
@pytest.mark.asyncio
async def test_end_to_end_with_real_agent_and_tools() -> None:
    """End-to-end test: normal agent with tools -> adapter -> reflective dataset -> proposal.

    Demonstrates optimizing both instructions AND tool descriptions.
    """

    # Create a model that simulates tool call behavior
    async def weather_model(messages, agent_info):
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            ToolCallPart,
            ToolReturnPart,
        )

        # Check if there's a tool return in the messages
        has_tool_return = any(
            isinstance(msg, ModelRequest)
            and any(isinstance(part, ToolReturnPart) for part in msg.parts)
            for msg in messages
        )

        if has_tool_return:
            # Second call: after tool return, respond with final text
            # Extract the location from the tool return to customize response
            for msg in messages:
                if isinstance(msg, ModelRequest):
                    for part in msg.parts:
                        if isinstance(part, ToolReturnPart):
                            # Parse the tool return to get location
                            content = part.content
                            if isinstance(content, str) and ":" in content:
                                location = (
                                    content.split(":")[0]
                                    .replace("Weather in", "")
                                    .strip()
                                )
                                return ModelResponse(
                                    parts=[
                                        TextPart(
                                            content=f"The weather in {location} is sunny and 72°F."
                                        )
                                    ]
                                )
            return ModelResponse(parts=[TextPart(content="It's sunny and 72°F.")])
        else:
            # First call: extract location from user prompt and call tool
            location = "Unknown"
            for msg in messages:
                if isinstance(msg, ModelRequest):
                    for part in msg.parts:
                        if hasattr(part, "content"):
                            content = str(part.content)
                            if "San Francisco" in content:
                                location = "San Francisco"
                            elif "New York" in content:
                                location = "New York"

            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_weather",
                        args={"location": location},
                        tool_call_id=f"call_{location.replace(' ', '_')}",
                    )
                ],
            )

    model = FunctionModel(function=weather_model)

    # Create an agent with a tool
    agent = Agent(
        model,
        instructions="Use the weather tool to answer questions about weather.",
        output_type=str,
    )

    @agent.tool_plain
    def get_weather(location: str) -> str:
        """Get current weather for a location.

        Args:
            location: The city name to get weather for.
        """
        return f"Weather in {location}: Sunny, 72°F"

    # Set up adapter with a simple metric
    def metric(
        case: Case[str, str, dict[str, str]], output: RolloutOutput[Any]
    ) -> MetricResult:
        # Score based on whether it used the tool
        success = output.success and output.result is not None
        return MetricResult(
            score=0.9 if success else 0.0,
            feedback="Good use of tools" if success else "Failed to use tools",
        )

    adapter = AgentAdapter(agent=agent, metric=metric, optimize_tools=True)

    # Create test data with multiple runs
    test_data = [
        Case(
            name="weather_1", inputs="What's the weather in San Francisco?", metadata={}
        ),
        Case(name="weather_2", inputs="What's the weather in New York?", metadata={}),
    ]

    # Run evaluation with traces
    candidate = adapter.get_components()
    eval_batch = await adapter.evaluate(
        batch=test_data,
        candidate=candidate,
        capture_traces=True,
    )

    # Get tool component names
    tool_components = [name for name in candidate.keys() if name.startswith("tool:")]
    assert tool_components, "Expected tool components to be exposed for plain agents"

    # Build reflective dataset - include instructions + all tool components
    components_to_optimize = ["instructions"] + tool_components
    reflective_data = adapter.make_reflective_dataset(
        candidate=candidate,
        eval_batch=eval_batch,
        components_to_update=components_to_optimize,
    )

    # Convert to CandidateProgram format for the proposal generator
    candidate_program = CandidateProgram(
        idx=0,
        components={
            name: value
            if isinstance(value, ComponentValue)
            else ComponentValue(name=name, text=value)
            for name, value in candidate.items()
        },
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )

    # Capture the prompt that would be sent to the optimizer
    captured_prompt: str | None = None

    async def capture_model(messages, agent_info):
        nonlocal captured_prompt
        prompt_parts = []
        for p in messages[-1].parts:
            if hasattr(p, "content"):
                if isinstance(p.content, str):
                    prompt_parts.append(p.content)
                elif isinstance(p.content, list):
                    for item in p.content:
                        if isinstance(item, str):
                            prompt_parts.append(item)
        captured_prompt = "".join(prompt_parts)
        # Return a dummy response with updates for all components
        updated = [
            {
                "component_name": "instructions",
                "optimized_value": "Improved instructions with tool guidance",
            }
        ]
        for tool_comp in tool_components:
            updated.append(
                {
                    "component_name": tool_comp,
                    "optimized_value": f"Improved {tool_comp}",
                }
            )

        import json

        content = json.dumps(
            {
                "reasoning": {
                    "pattern_discovery": "Tool usage patterns were good",
                    "creative_hypothesis": "Could be more explicit",
                    "experimental_approach": "Add more guidance about when to use tools",
                },
                "updated_components": updated,
            }
        )
        return ModelResponse(parts=[TextPart(content=content)])

    # Run the proposal generator
    generator = InstructionProposalGenerator()
    model = FunctionModel(function=capture_model)
    result = await generator.propose_texts(
        candidate=candidate_program,
        reflective_data=reflective_data,
        components=components_to_optimize,
        model=model,
    )

    # Verify the result includes updated instructions and tool components
    assert result.texts["instructions"] == "Improved instructions with tool guidance"
    for tool_comp in tool_components:
        assert tool_comp in result.texts
        assert "Improved" in result.texts[tool_comp]

    # Verify the complete prompt with snapshot
    assert captured_prompt == snapshot("""\
# Creative Instruction Design Challenge

Transform the student agent's performance through innovative instruction formats.

## Context
- A student agent has been running with the configuration shown below
- We've collected traces from real production runs
- Your job is to improve specific components so the student agent performs better

---

## Full student agent configuration

This is the complete configuration the student agent was running with:

=== start component: `instructions` given to student ===
Use the weather tool to answer questions about weather.
=== end ===

=== start component: `tool:get_weather:description` given to student ===
Get current weather for a location.
=== end ===

=== start component: `tool:get_weather:param:location` given to student ===
The city name to get weather for.
=== end ===

**Tools available to student (JSON Schema):**
```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city name to get weather for."
          }
        }
      },
      "description": "Get current weather for a location."
    }
  }
]
```

---

## Production traces from student agent runs

Each trace contains:
- `messages`: Full conversation history with system prompts, user inputs, assistant responses, tool calls, and tool returns
- `tools`: Tool definitions that were available (if any)
- `score`: Performance score (0.0-1.0, higher is better)
- `success`: Whether the run completed successfully
- `feedback`: Evaluator feedback on this specific run


## Execution Traces Available for Analysis

2 traces available from the execution: 2 succeeded, 0 failed.
The traces are stored on disk in structured JSONL format.
You must use the `run_trace_analysis(python_script: str)` tool to write and execute python map-reduce scripts to analyze these traces within the unbounded Ouros Data Plane.
You may also use `analyze_trace_with_llm(trace_id, prompt)` to spawn a Recursive Language Model (RLM) sub-agent to deeply inspect specific traces for semantic failures.


### Analysis guidance
- Use `run_trace_analysis` to aggregate errors or find common failure modes.
- Use `analyze_trace_with_llm` to understand *why* a specific trace failed if the python analysis is insufficient.
- What failure patterns repeat across runs?
- Are components misaligned (e.g., instructions referencing tools that don't exist)?
- Which successful patterns should be preserved or extended?
- What domain knowledge should be codified in the prompts?
- Are there patterns of inefficient tool usage (redundant calls, speculative calls, lack of planning)?
- How can prompts guide the student to gather what's needed in fewer, well-targeted tool calls?

---

## Components to update

Rewrite these components as a coordinated update based on the evidence above:

=== start component: `instructions` current value ===
Use the weather tool to answer questions about weather.
=== end ===

=== start component: `tool:get_weather:description` current value ===
Get current weather for a location.
=== end ===

=== start component: `tool:get_weather:param:location` current value ===
The city name to get weather for.
=== end ===
""")

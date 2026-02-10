"""Tests for tool component extraction and application."""

from __future__ import annotations

from typing import Any, cast

import pytest

from pydantic_ai.agent import AbstractAgent
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.tools import ToolDefinition

from pydantic_ai_gepa.tool_components import (
    ToolComponentCatalog,
    ToolOptimizationManager,
)


def _make_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="format_text",
        description="Format content for downstream processing.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Raw text to format.",
                },
                "style": {
                    "type": "string",
                    "description": "Formatting instructions to apply.",
                },
            },
            "required": ["text", "style"],
            "additionalProperties": False,
        },
    )


class _DummyAgent:
    """Minimal agent stub for ToolOptimizationManager tests."""

    def __init__(self) -> None:
        self._prepare_tools = None
        self.model = None


def test_catalog_ingest_records_seed_components() -> None:
    catalog = ToolComponentCatalog()
    tool_def = _make_tool_definition()

    catalog.ingest([tool_def])
    seeds = catalog.seed_snapshot()

    assert (
        seeds["tool:format_text:description"]
        == "Format content for downstream processing."
    )
    assert seeds["tool:format_text:param:text"] == "Raw text to format."
    assert seeds["tool:format_text:param:style"] == "Formatting instructions to apply."
    assert catalog.metadata_for("format_text") is not None


def test_apply_candidate_updates_tool_definition() -> None:
    manager = ToolOptimizationManager(cast(AbstractAgent[Any, Any], _DummyAgent()))
    tool_def = _make_tool_definition()
    manager._catalog.ingest([tool_def])  # type: ignore[attr-defined]

    candidate = {
        "tool:format_text:description": "Polish the incoming copy.",
        "tool:format_text:param:text": "Draft prose needing polish.",
    }
    updated = manager._apply_candidate_to_tool(tool_def, candidate)  # type: ignore[attr-defined]

    assert updated.description == "Polish the incoming copy."
    params = updated.parameters_json_schema["properties"]
    assert params["text"]["description"] == "Draft prose needing polish."
    # Keys that were not overridden remain intact.
    assert params["style"]["description"] == "Formatting instructions to apply."


def test_record_model_request_tracks_builtin_tools() -> None:
    manager = ToolOptimizationManager(cast(AbstractAgent[Any, Any], _DummyAgent()))
    builtin = WebSearchTool()

    manager.record_model_request(
        function_tools=[_make_tool_definition()], builtin_tools=[builtin]
    )

    latest = manager.latest_builtin_tools()
    assert len(latest) == 1
    assert latest[0].kind == "web_search"


class TestSelectiveToolOptimization:
    """Tests for selective tool optimization via allowed_tools."""

    def test_filter_candidate_with_allowed_tools_filters_by_name(self) -> None:
        """When allowed_tools is set, only those tools are included."""
        manager = ToolOptimizationManager(
            cast(AbstractAgent[Any, Any], _DummyAgent()),
            allowed_tools={"format_text"},
        )

        candidate = {
            "tool:format_text:description": "Allowed tool description",
            "tool:other_tool:description": "Should be filtered out",
            "tool:format_text:param:text": "Allowed param",
            "tool:other_tool:param:foo": "Should be filtered out",
            "instructions": "Not a tool key",
        }

        filtered = manager._filter_candidate(candidate)

        assert filtered is not None
        assert "tool:format_text:description" in filtered
        assert "tool:format_text:param:text" in filtered
        assert "tool:other_tool:description" not in filtered
        assert "tool:other_tool:param:foo" not in filtered
        assert "instructions" not in filtered

    def test_filter_candidate_without_allowed_tools_includes_all(self) -> None:
        """When allowed_tools is None, all tool keys are included."""
        manager = ToolOptimizationManager(
            cast(AbstractAgent[Any, Any], _DummyAgent()),
            allowed_tools=None,
        )

        candidate = {
            "tool:format_text:description": "First tool",
            "tool:other_tool:description": "Second tool",
            "instructions": "Not a tool key",
        }

        filtered = manager._filter_candidate(candidate)

        assert filtered is not None
        assert "tool:format_text:description" in filtered
        assert "tool:other_tool:description" in filtered
        assert "instructions" not in filtered

    def test_allow_tool_adds_to_existing_set(self) -> None:
        """allow_tool() adds a tool name to the allowed set."""
        manager = ToolOptimizationManager(
            cast(AbstractAgent[Any, Any], _DummyAgent()),
            allowed_tools={"existing_tool"},
        )

        manager.allow_tool("new_tool")

        assert manager._allowed_tools == {"existing_tool", "new_tool"}

    def test_allow_tool_creates_set_when_none(self) -> None:
        """allow_tool() creates a new set when allowed_tools was None."""
        manager = ToolOptimizationManager(
            cast(AbstractAgent[Any, Any], _DummyAgent()),
            allowed_tools=None,
        )

        manager.allow_tool("search_examples")

        assert manager._allowed_tools == {"search_examples"}

    def test_apply_candidate_respects_allowed_tools(self) -> None:
        """Candidate application only affects allowed tools."""
        manager = ToolOptimizationManager(
            cast(AbstractAgent[Any, Any], _DummyAgent()),
            allowed_tools={"format_text"},
        )

        # Create two tools
        format_tool = _make_tool_definition()
        other_tool = ToolDefinition(
            name="other_tool",
            description="Original other description",
            parameters_json_schema={"type": "object", "properties": {}},
        )

        manager._catalog.ingest([format_tool, other_tool])

        # Candidate tries to update both
        candidate = {
            "tool:format_text:description": "Updated format description",
            "tool:other_tool:description": "Updated other description",
        }

        # Filter the candidate (simulating what happens in candidate_context)
        filtered = manager._filter_candidate(candidate)
        assert filtered is not None

        # Apply to format_text - should update
        updated_format = manager._apply_candidate_to_tool(format_tool, filtered)
        assert updated_format.description == "Updated format description"

        # Apply to other_tool - should NOT update (not in filtered candidate)
        updated_other = manager._apply_candidate_to_tool(other_tool, filtered)
        assert updated_other.description == "Original other description"


def test_get_or_create_adds_allowed_tools_to_existing_manager() -> None:
    """get_or_create_tool_optimizer adds to existing manager's allowed tools."""
    from pydantic_ai_gepa.tool_components import (
        get_or_create_tool_optimizer,
    )

    agent = cast(AbstractAgent[Any, Any], _DummyAgent())

    # Create initial manager with one allowed tool
    manager1 = get_or_create_tool_optimizer(agent, allowed_tools={"tool_a"})
    assert manager1._allowed_tools == {"tool_a"}

    # Call again with additional tools - should add to existing
    manager2 = get_or_create_tool_optimizer(agent, allowed_tools={"tool_b", "tool_c"})

    assert manager1 is manager2  # Same instance
    assert manager2._allowed_tools == {"tool_a", "tool_b", "tool_c"}


class TestSearchExamplesToolOptimization:
    """Tests for search_examples tool component optimization."""

    def test_search_examples_components_registered_via_record_model_request(
        self,
    ) -> None:
        """Tool components are available after record_model_request."""
        from pydantic_ai_gepa.gepa_graph.proposal.student_tools import (
            create_example_search_tool,
        )
        from pydantic_ai_gepa.gepa_graph.example_bank import InMemoryExampleBank
        from pydantic_ai_gepa.types import ExampleBankConfig

        manager = ToolOptimizationManager(
            cast(AbstractAgent[Any, Any], _DummyAgent()),
            allowed_tools={"search_examples"},
        )

        # Create the example bank and toolset
        config = ExampleBankConfig(search_tool_instruction="Find similar examples")
        bank = InMemoryExampleBank(config=config)
        toolset = create_example_search_tool(
            bank=bank,
            instruction=bank.search_tool_instruction,
            k=3,
        )

        # Extract ToolDefinition and register it
        tool = toolset.tools["search_examples"]
        tool_def = ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=tool.function_schema.json_schema,
        )
        manager.record_model_request(function_tools=[tool_def])

        # Verify components are available
        seed_components = manager.get_seed_components()
        assert "tool:search_examples:description" in seed_components
        assert (
            seed_components["tool:search_examples:description"]
            == "Find similar examples"
        )
        assert "tool:search_examples:param:query" in seed_components

    def test_search_examples_description_updated_from_candidate(self) -> None:
        """Optimized description from candidate is applied to tool."""
        from pydantic_ai_gepa.gepa_graph.proposal.student_tools import (
            create_example_search_tool,
        )
        from pydantic_ai_gepa.gepa_graph.example_bank import InMemoryExampleBank
        from pydantic_ai_gepa.types import ExampleBankConfig

        manager = ToolOptimizationManager(
            cast(AbstractAgent[Any, Any], _DummyAgent()),
            allowed_tools={"search_examples"},
        )

        # Create and register the tool
        config = ExampleBankConfig(search_tool_instruction="Original description")
        bank = InMemoryExampleBank(config=config)
        toolset = create_example_search_tool(
            bank=bank,
            instruction=bank.search_tool_instruction,
            k=3,
        )
        tool = toolset.tools["search_examples"]
        tool_def = ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=tool.function_schema.json_schema,
        )
        manager.record_model_request(function_tools=[tool_def])

        # Candidate with optimized description
        candidate = {
            "tool:search_examples:description": "Optimized: Search for relevant examples to guide your response",
            "tool:search_examples:param:query": "Describe the type of example you need",
        }

        # Filter and apply the candidate
        filtered = manager._filter_candidate(candidate)
        assert filtered is not None

        updated_tool = manager._apply_candidate_to_tool(tool_def, filtered)

        # Verify the description was updated
        assert (
            updated_tool.description
            == "Optimized: Search for relevant examples to guide your response"
        )
        # Verify param description was updated
        assert (
            updated_tool.parameters_json_schema["properties"]["query"]["description"]
            == "Describe the type of example you need"
        )


class TestSearchExamplesToolCapturedDuringAgentRun:
    """Tests that search_examples tool is captured when passed to agent.run()."""

    @pytest.mark.asyncio
    async def test_search_examples_captured_via_prepare_tools_hook(self) -> None:
        """When agent.run() is called with the search_examples toolset,
        the tool should be captured by the ToolOptimizationManager."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel
        from pydantic_ai_gepa.gepa_graph.proposal.student_tools import (
            create_example_search_tool,
        )
        from pydantic_ai_gepa.gepa_graph.example_bank import InMemoryExampleBank
        from pydantic_ai_gepa.types import ExampleBankConfig
        from pydantic_ai_gepa.tool_components import get_or_create_tool_optimizer

        # Create an agent with tool optimization enabled
        agent = Agent(
            model=TestModel(),
            instructions="Test agent",
        )

        # Install the tool optimizer (this is what AgentAdapter does)
        optimizer = get_or_create_tool_optimizer(
            agent, allowed_tools={"search_examples"}
        )

        # Before running: no search_examples components
        seed_before = optimizer.get_seed_components()
        assert "tool:search_examples:description" not in seed_before

        # Create the example bank and toolset
        config = ExampleBankConfig(search_tool_instruction="Find similar examples")
        bank = InMemoryExampleBank(config=config)
        toolset = create_example_search_tool(
            bank=bank,
            instruction=bank.search_tool_instruction,
            k=3,
        )

        # Run the agent with the toolset - this triggers _prepare_tools hook
        await agent.run("Hello", toolsets=[toolset])

        # After running: search_examples components should be captured
        seed_after = optimizer.get_seed_components()
        assert "tool:search_examples:description" in seed_after
        assert seed_after["tool:search_examples:description"] == "Find similar examples"
        assert "tool:search_examples:param:query" in seed_after

    @pytest.mark.asyncio
    async def test_search_examples_empty_bank_returns_message(self) -> None:
        """When the example bank is empty, search_examples returns a friendly message."""
        from pydantic_ai_gepa.gepa_graph.proposal.student_tools import (
            create_example_search_tool,
        )
        from pydantic_ai_gepa.gepa_graph.example_bank import InMemoryExampleBank
        from pydantic_ai_gepa.types import ExampleBankConfig

        config = ExampleBankConfig(search_tool_instruction="Find similar examples")
        bank = InMemoryExampleBank(config=config)
        toolset = create_example_search_tool(
            bank=bank,
            instruction=bank.search_tool_instruction,
            k=3,
        )

        # Get the underlying function and call it directly
        search_fn = toolset.tools["search_examples"].function
        result = search_fn(query="anything")  # type: ignore[call-arg]

        assert result == "No examples have been added to the example bank yet."

    @pytest.mark.asyncio
    async def test_search_examples_captured_even_with_optimize_tools_true(self) -> None:
        """When optimize_tools=True (all tools), search_examples should still be captured."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel
        from pydantic_ai_gepa.gepa_graph.proposal.student_tools import (
            create_example_search_tool,
        )
        from pydantic_ai_gepa.gepa_graph.example_bank import InMemoryExampleBank
        from pydantic_ai_gepa.types import ExampleBankConfig
        from pydantic_ai_gepa.tool_components import get_or_create_tool_optimizer

        agent = Agent(
            model=TestModel(),
            instructions="Test agent",
        )

        # allowed_tools=None means all tools are allowed (optimize_tools=True)
        optimizer = get_or_create_tool_optimizer(agent, allowed_tools=None)

        config = ExampleBankConfig(search_tool_instruction="Find similar examples")
        bank = InMemoryExampleBank(config=config)
        toolset = create_example_search_tool(
            bank=bank,
            instruction=bank.search_tool_instruction,
            k=3,
        )

        await agent.run("Hello", toolsets=[toolset])

        # Components should be captured regardless of allowed_tools setting
        seed = optimizer.get_seed_components()
        assert "tool:search_examples:description" in seed
        assert "tool:search_examples:param:query" in seed

    @pytest.mark.asyncio
    async def test_search_examples_captured_with_specific_allowed_tools(self) -> None:
        """When optimize_tools={'other_tool'}, search_examples is still ingested but not filtered."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel
        from pydantic_ai_gepa.gepa_graph.proposal.student_tools import (
            create_example_search_tool,
        )
        from pydantic_ai_gepa.gepa_graph.example_bank import InMemoryExampleBank
        from pydantic_ai_gepa.types import ExampleBankConfig
        from pydantic_ai_gepa.tool_components import get_or_create_tool_optimizer

        agent = Agent(
            model=TestModel(),
            instructions="Test agent",
        )

        # Only 'other_tool' is in allowed_tools, NOT search_examples
        optimizer = get_or_create_tool_optimizer(agent, allowed_tools={"other_tool"})

        config = ExampleBankConfig(search_tool_instruction="Find similar examples")
        bank = InMemoryExampleBank(config=config)
        toolset = create_example_search_tool(
            bank=bank,
            instruction=bank.search_tool_instruction,
            k=3,
        )

        await agent.run("Hello", toolsets=[toolset])

        # The catalog ingests ALL tools, so seed_components includes search_examples
        seed = optimizer.get_seed_components()
        assert "tool:search_examples:description" in seed
        assert "tool:search_examples:param:query" in seed

        # However, when filtering a candidate, search_examples would be excluded
        candidate = {
            "tool:search_examples:description": "New description",
            "tool:other_tool:description": "Other description",
        }
        filtered = optimizer._filter_candidate(candidate)
        assert filtered is not None
        assert "tool:other_tool:description" in filtered
        assert "tool:search_examples:description" not in filtered

    @pytest.mark.asyncio
    async def test_search_examples_appears_in_model_request_parameters(self) -> None:
        """Verify search_examples tool appears in model_request_parameters.function_tools."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel
        from pydantic_ai_gepa.gepa_graph.proposal.student_tools import (
            create_example_search_tool,
        )
        from pydantic_ai_gepa.gepa_graph.example_bank import InMemoryExampleBank
        from pydantic_ai_gepa.types import ExampleBankConfig

        test_model = TestModel()
        agent = Agent(
            model=test_model,
            instructions="Test agent",
        )

        config = ExampleBankConfig(search_tool_instruction="Find similar examples")
        bank = InMemoryExampleBank(config=config)
        toolset = create_example_search_tool(
            bank=bank,
            instruction=bank.search_tool_instruction,
            k=3,
        )

        await agent.run("Hello", toolsets=[toolset])

        # Tool definitions are available on TestModel after a run
        assert test_model.last_model_request_parameters is not None
        function_tools = test_model.last_model_request_parameters.function_tools
        tool_names = [t.name for t in function_tools]
        assert "search_examples" in tool_names

    @pytest.mark.asyncio
    async def test_agent_adapter_get_components_includes_search_examples_after_evaluate(
        self,
    ) -> None:
        """Real AgentAdapter.get_components() should include search_examples after evaluation."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel
        from pydantic_evals import Case
        from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapter
        from pydantic_ai_gepa.gepa_graph.example_bank import InMemoryExampleBank
        from pydantic_ai_gepa.gepa_graph.models import ComponentValue
        from pydantic_ai_gepa.types import (
            ExampleBankConfig,
            MetricResult,
            RolloutOutput,
        )

        agent = Agent(
            model=TestModel(),
            instructions="Test agent",
        )

        def metric(
            case: Case[str, str, None], output: RolloutOutput[str]
        ) -> MetricResult:
            return MetricResult(score=1.0, feedback="ok")

        # Create adapter with optimize_tools=True
        adapter = AgentAdapter(
            agent=agent,
            metric=metric,
            optimize_tools=True,
        )

        # Before evaluation: no search_examples components
        components_before = adapter.get_components()
        assert "tool:search_examples:description" not in components_before

        # Create example bank
        config = ExampleBankConfig(search_tool_instruction="Find similar examples")
        bank = InMemoryExampleBank(config=config)

        # Create a simple case
        case = Case(name="test", inputs="Hello")
        candidate = {"instructions": ComponentValue(name="instructions", text="Test")}

        # Run evaluation with example_bank - this triggers agent.run() with the toolset
        await adapter.evaluate(
            batch=[case],
            candidate=candidate,
            capture_traces=True,
            example_bank=bank,
        )

        # After evaluation: search_examples components should be captured
        components_after = adapter.get_components()
        assert "tool:search_examples:description" in components_after, (
            f"Expected 'tool:search_examples:description' in components. "
            f"Got: {list(components_after.keys())}"
        )
        assert (
            components_after["tool:search_examples:description"].text
            == "Find similar examples"
        )
        assert "tool:search_examples:param:query" in components_after

    @pytest.mark.asyncio
    async def test_signature_agent_adapter_get_components_includes_search_examples(
        self,
    ) -> None:
        """SignatureAgentAdapter.get_components() should include search_examples after evaluation."""
        from pydantic import BaseModel, Field
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel
        from pydantic_evals import Case
        from pydantic_ai_gepa.adapters.agent_adapter import SignatureAgentAdapter
        from pydantic_ai_gepa.gepa_graph.example_bank import InMemoryExampleBank
        from pydantic_ai_gepa.gepa_graph.models import ComponentValue
        from pydantic_ai_gepa.signature_agent import SignatureAgent
        from pydantic_ai_gepa.types import (
            ExampleBankConfig,
            MetricResult,
            RolloutOutput,
        )

        class TestInput(BaseModel):
            text: str = Field(description="Input text")

        class TestOutput(BaseModel):
            result: str = Field(description="Output result")

        # SignatureAgent wraps another agent
        wrapped_agent = Agent(
            model=TestModel(),
            instructions="Test agent",
            output_type=TestOutput,
        )

        agent = SignatureAgent(
            wrapped=wrapped_agent,
            input_type=TestInput,
            optimize_tools=True,
        )

        def metric(
            case: Case[TestInput, TestOutput, None], output: RolloutOutput[TestOutput]
        ) -> MetricResult:
            return MetricResult(score=1.0, feedback="ok")

        adapter = SignatureAgentAdapter(
            agent=agent,
            metric=metric,
            optimize_tools=True,
        )

        # Before evaluation: no search_examples components
        components_before = adapter.get_components()
        assert "tool:search_examples:description" not in components_before

        # Create example bank
        config = ExampleBankConfig(search_tool_instruction="Find similar examples")
        bank = InMemoryExampleBank(config=config)

        # Create a simple case
        case = Case(name="test", inputs=TestInput(text="Hello"))
        candidate = {"instructions": ComponentValue(name="instructions", text="Test")}

        # Run evaluation with example_bank
        await adapter.evaluate(
            batch=[case],
            candidate=candidate,
            capture_traces=True,
            example_bank=bank,
        )

        # After evaluation: search_examples components should be captured
        components_after = adapter.get_components()
        assert "tool:search_examples:description" in components_after, (
            f"Expected 'tool:search_examples:description' in components. "
            f"Got: {list(components_after.keys())}"
        )
        assert (
            components_after["tool:search_examples:description"].text
            == "Find similar examples"
        )
        assert "tool:search_examples:param:query" in components_after

    @pytest.mark.asyncio
    async def test_adapter_optimize_tools_installs_on_agent_without_optimizer(
        self,
    ) -> None:
        """Adapter with optimize_tools=True should work even if SignatureAgent has optimize_tools=False.

        This reproduces the production setup where SignatureAgent(optimize_tools=False) but
        SignatureAgentAdapter(optimize_tools=True).
        """
        from pydantic import BaseModel, Field
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel
        from pydantic_evals import Case
        from pydantic_ai_gepa.adapters.agent_adapter import SignatureAgentAdapter
        from pydantic_ai_gepa.gepa_graph.example_bank import InMemoryExampleBank
        from pydantic_ai_gepa.gepa_graph.models import ComponentValue
        from pydantic_ai_gepa.signature_agent import SignatureAgent
        from pydantic_ai_gepa.types import (
            ExampleBankConfig,
            MetricResult,
            RolloutOutput,
        )

        class TestInput(BaseModel):
            text: str = Field(description="Input text")

        class TestOutput(BaseModel):
            result: str = Field(description="Output result")

        # Create SignatureAgent with optimize_tools=False (like the production code)
        wrapped_agent = Agent(
            model=TestModel(),
            instructions="Test agent",
            output_type=TestOutput,
        )

        sig_agent = SignatureAgent(
            wrapped=wrapped_agent,
            input_type=TestInput,
            optimize_tools=False,  # This mirrors the production setup
        )

        def metric(
            case: Case[TestInput, TestOutput, None], output: RolloutOutput[TestOutput]
        ) -> MetricResult:
            return MetricResult(score=1.0, feedback="ok")

        # Adapter with optimize_tools=True should still work
        adapter = SignatureAgentAdapter(
            agent=sig_agent,
            metric=metric,
            optimize_tools=True,  # Adapter enables it
        )

        # Before evaluation: no search_examples components
        components_before = adapter.get_components()
        assert "tool:search_examples:description" not in components_before

        # Create example bank
        config = ExampleBankConfig(search_tool_instruction="Find similar examples")
        bank = InMemoryExampleBank(config=config)

        # Create a simple case
        case = Case(name="test", inputs=TestInput(text="Hello"))
        candidate = {"instructions": ComponentValue(name="instructions", text="Test")}

        # Run evaluation with example_bank
        await adapter.evaluate(
            batch=[case],
            candidate=candidate,
            capture_traces=True,
            example_bank=bank,
        )

        # After evaluation: search_examples components should be captured
        components_after = adapter.get_components()
        assert "tool:search_examples:description" in components_after, (
            f"Expected 'tool:search_examples:description' in components. "
            f"Got: {list(components_after.keys())}"
        )
        assert (
            components_after["tool:search_examples:description"].text
            == "Find similar examples"
        )

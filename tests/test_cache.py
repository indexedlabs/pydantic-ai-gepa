"""Tests for the caching system."""

from __future__ import annotations

import tempfile

import pytest
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_gepa.cache import CacheManager, create_cached_metric
from pydantic_ai_gepa.gepa_graph.models import ComponentValue
from pydantic_ai_gepa.gepa_graph.proposal.instruction import (
    ComponentUpdate,
    InstructionProposalOutput,
    TrajectoryAnalysis,
)
from pydantic_ai_gepa.runner import optimize_agent
from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapterTrajectory
from pydantic_ai_gepa.types import MetricResult, ReflectionConfig, RolloutOutput
from pydantic_evals import Case


@dataclass
class LabelMetadata:
    label: str


def _dummy_reasoning() -> TrajectoryAnalysis:
    return TrajectoryAnalysis(
        pattern_discovery="baseline patterns observed in testing",
        creative_hypothesis="placeholder hypothesis for unit tests",
        experimental_approach="placeholder approach for unit tests",
    )


def _prompt_case(
    content: str,
    *,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> Case[str, str, dict[str, Any] | None]:
    return Case(
        name=name,
        inputs=content,
        expected_output=None,
        metadata=metadata,
    )


def test_cache_manager_basic():
    """Test basic cache manager operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir, enabled=True, verbose=False)

        # Create test data
        case = _prompt_case(
            "Test prompt",
            name="test-1",
            metadata={"test": "data"},
        )

        output = RolloutOutput.from_success("Test result")
        candidate = {
            "instructions": ComponentValue(
                name="instructions", text="Test instructions"
            ),
        }

        # Initially, cache should miss
        result = cache.get_cached_metric_result(case, None, output, candidate)
        assert result is None

        # Cache a result
        cache.cache_metric_result(
            case,
            None,
            output,
            candidate,
            MetricResult(score=0.95, feedback="Good job"),
        )

        # Now cache should hit
        result = cache.get_cached_metric_result(case, None, output, candidate)
        assert result is not None
        assert result == MetricResult(score=0.95, feedback="Good job")

        # Different candidate should miss
        different_candidate = {
            "instructions": ComponentValue(
                name="instructions", text="Different instructions"
            ),
        }
        result = cache.get_cached_metric_result(
            case,
            None,
            output,
            different_candidate,
        )
        assert result is None

        # Different output should miss
        different_output = RolloutOutput.from_success("Different result")
        result = cache.get_cached_metric_result(
            case,
            None,
            different_output,
            candidate,
        )
        assert result is None

        # Check cache stats
        stats = cache.get_cache_stats()
        assert stats["enabled"] is True
        assert stats["num_cached_results"] == 1

        # Clear cache
        cache.clear_cache()
        stats = cache.get_cache_stats()
        assert stats["num_cached_results"] == 0


def test_cache_scopes_entries_by_model():
    """Cache keys should include the model identifier."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir, enabled=True, verbose=False)

        case = _prompt_case(
            "Test prompt",
            name="case-1",
            metadata={"label": "positive"},
        )
        output = RolloutOutput.from_success("positive")
        candidate = {
            "instructions": ComponentValue(
                name="instructions", text="Classify sentiment"
            ),
        }

        result_a = MetricResult(score=0.9, feedback="model-a")
        result_b = MetricResult(score=0.5, feedback="model-b")

        cache.cache_metric_result(
            case,
            None,
            output,
            candidate,
            result_a,
            model_identifier="model-a",
        )

        assert (
            cache.get_cached_metric_result(
                case,
                None,
                output,
                candidate,
                model_identifier="model-a",
            )
            == result_a
        )
        assert (
            cache.get_cached_metric_result(
                case,
                None,
                output,
                candidate,
                model_identifier="model-b",
            )
            is None
        )

        cache.set_model_identifier("model-b")
        cache.cache_metric_result(
            case,
            None,
            output,
            candidate,
            result_b,
        )

        assert (
            cache.get_cached_metric_result(
                case,
                None,
                output,
                candidate,
                model_identifier="model-b",
            )
            == result_b
        )
        assert (
            cache.get_cached_metric_result(
                case,
                None,
                output,
                candidate,
                model_identifier="model-a",
            )
            == result_a
        )


def test_cache_manager_with_signature():
    """Test cache manager with signature-based data instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir, enabled=True, verbose=False)

        class TestSignature(BaseModel):
            text: str
            value: int = 42

        # Create test data with signature
        case = Case(
            name="sig-test-1",
            inputs=TestSignature(text="Test input", value=100),
            metadata=LabelMetadata(label="positive"),
        )

        output = RolloutOutput.from_success("positive")
        candidate = {
            "instructions": ComponentValue(
                name="instructions", text="Classify the text"
            ),
            "signature:TestSignature:text:desc": ComponentValue(
                name="signature:TestSignature:text:desc",
                text="Input text",
            ),
        }

        # Cache a result
        cache.cache_metric_result(
            case,
            None,
            output,
            candidate,
            MetricResult(score=1.0, feedback="Correct"),
        )

        # Should get cache hit with same inputs
        result = cache.get_cached_metric_result(case, None, output, candidate)
        assert result == MetricResult(score=1.0, feedback="Correct")

        # Different signature value should miss
        case2 = Case(
            name="sig-test-2",
            inputs=TestSignature(text="Different input", value=100),
            metadata=LabelMetadata(label="positive"),
        )
        result = cache.get_cached_metric_result(case2, None, output, candidate)
        assert result is None


def test_cache_manager_disabled():
    """Test that cache manager does nothing when disabled."""
    cache = CacheManager(cache_dir=None, enabled=False, verbose=False)

    case = _prompt_case("Test", name="test-1")
    output = RolloutOutput.from_success("Result")
    candidate = {
        "instructions": ComponentValue(name="instructions", text="Do something"),
    }

    # Should always return None when disabled
    result = cache.get_cached_metric_result(case, None, output, candidate)
    assert result is None

    # Caching should be no-op
    cache.cache_metric_result(
        case,
        None,
        output,
        candidate,
        MetricResult(score=0.5, feedback="Feedback"),
    )
    result = cache.get_cached_metric_result(case, None, output, candidate)
    assert result is None

    # Stats should show disabled
    stats = cache.get_cache_stats()
    assert stats == {"enabled": False}


def test_create_cached_metric():
    """Test the cached metric wrapper function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_manager = CacheManager(cache_dir=tmpdir, enabled=True, verbose=False)

        # Create a mock metric that counts calls
        call_count = 0

        def mock_metric(data_inst, output):
            nonlocal call_count
            call_count += 1
            return MetricResult(score=0.8, feedback=f"Call {call_count}")

        # Create cached version
        candidate = {
            "instructions": ComponentValue(name="instructions", text="Test"),
        }
        cached_metric = create_cached_metric(mock_metric, cache_manager, candidate)

        # Create test data
        case = _prompt_case("Test", name="test-1")
        output = RolloutOutput.from_success("Result")

        from typing import cast

        # First call should invoke the metric
        result = cast(MetricResult, cached_metric(case, output))
        assert result.score == 0.8
        assert result.feedback == "Call 1"
        assert call_count == 1

        # Second call with same inputs should use cache
        result = cast(MetricResult, cached_metric(case, output))
        assert result.score == 0.8
        assert result.feedback == "Call 1"  # Same feedback
        assert call_count == 1  # Metric not called again

        # Different inputs should invoke metric again
        case2 = _prompt_case("Different", name="test-2")
        result = cast(MetricResult, cached_metric(case2, output))
        assert result.score == 0.8
        assert result.feedback == "Call 2"
        assert call_count == 2


@pytest.mark.asyncio
async def test_optimize_agent_with_caching():
    """Test that optimize_agent works with caching enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple dataset
        labels = ["positive", "negative", "neutral"]
        trainset = [
            Case(
                name=f"case-{i}",
                inputs=f"Classify: {label}",
                metadata=LabelMetadata(label=label),
                expected_output=label,
            )
            for i, label in enumerate(labels)
        ]

        # Track metric calls
        metric_calls = []

        def metric(case, output):
            metric_calls.append(case.name)
            predicted = str(output.result).lower() if output.success else ""
            metadata = case.metadata or LabelMetadata(label="")
            expected = metadata.label.lower()
            score = 1.0 if predicted == expected else 0.0
            return MetricResult(score=score, feedback=f"Score: {score}")

        # Create agent
        agent = Agent(
            TestModel(custom_output_text="positive"),
            instructions="Classify text as positive, negative, or neutral.",
        )

        reflection_output = InstructionProposalOutput(
            reasoning=_dummy_reasoning(),
            updated_components=[
                ComponentUpdate(
                    component_name="instructions",
                    optimized_value="Updated",
                )
            ],
        )
        reflection_model = TestModel(
            custom_output_args=reflection_output.model_dump(mode="python")
        )

        # First run with caching enabled
        result1 = await optimize_agent(
            agent=agent,
            trainset=trainset,
            metric=metric,
            reflection_config=ReflectionConfig(model=reflection_model),
            max_metric_calls=15,
            seed=42,
            enable_cache=True,
            cache_dir=tmpdir,
            cache_verbose=False,
        )

        first_run_calls = len(metric_calls)
        assert first_run_calls > 0
        assert result1.num_metric_calls <= 15

        # Clear metric calls
        metric_calls.clear()

        # Second run should use cache for overlapping evaluations
        result2 = await optimize_agent(
            agent=agent,
            trainset=trainset,
            metric=metric,
            reflection_config=ReflectionConfig(model=reflection_model),
            max_metric_calls=15,
            seed=42,  # Same seed to get same behavior
            enable_cache=True,
            cache_dir=tmpdir,
            cache_verbose=False,
        )

        # Should have fewer metric calls due to caching
        second_run_calls = len(metric_calls)
        assert second_run_calls < first_run_calls

        # Results should be consistent
        assert result2.original_candidate == result1.original_candidate


def test_cache_handles_errors():
    """Test that cache handles errors gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir, enabled=True, verbose=False)

        case = _prompt_case("Test", name="test-1")

        # Test with error output
        error_output = RolloutOutput.from_error(Exception("Test error"))
        candidate = {
            "instructions": ComponentValue(name="instructions", text="Test"),
        }

        # Should be able to cache error results
        cache.cache_metric_result(
            case,
            None,
            error_output,
            candidate,
            MetricResult(score=0.0, feedback="Error occurred"),
        )

        # Should retrieve cached error result
        result = cache.get_cached_metric_result(case, None, error_output, candidate)
        assert result == MetricResult(score=0.0, feedback="Error occurred")

        # Success output with same data should be different cache key
        success_output = RolloutOutput.from_success("Result")
        result = cache.get_cached_metric_result(case, None, success_output, candidate)
        assert result is None


def test_cache_agent_runs():
    """Test caching of agent execution results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir, enabled=True, verbose=False)

        # Create test data
        case = _prompt_case(
            "Test prompt",
            name="test-1",
            metadata={"test": "data"},
        )

        output = RolloutOutput.from_success("Agent result")
        trajectory = AgentAdapterTrajectory(
            messages=[], final_output="Agent result", error=None
        )
        candidate = {
            "instructions": ComponentValue(
                name="instructions", text="Test instructions"
            ),
        }

        # Initially, cache should miss
        result = cache.get_cached_agent_run(case, 0, candidate, capture_traces=True)
        assert result is None

        # Cache an agent run with traces
        cache.cache_agent_run(
            case,
            0,
            candidate,
            trajectory,
            output,
            capture_traces=True,
        )

        # Now cache should hit
        result = cache.get_cached_agent_run(case, 0, candidate, capture_traces=True)
        assert result is not None
        cached_trajectory, cached_output = result
        assert cached_output.result == "Agent result"
        assert cached_trajectory is not None
        assert cached_trajectory.final_output == "Agent result"

        # Different capture_traces value should miss
        result = cache.get_cached_agent_run(case, 0, candidate, capture_traces=False)
        assert result is None

        # Cache without traces
        cache.cache_agent_run(
            case,
            0,
            candidate,
            None,
            output,
            capture_traces=False,
        )

        # Should hit for non-trace version
        result = cache.get_cached_agent_run(case, 0, candidate, capture_traces=False)
        assert result is not None
        cached_trajectory, cached_output = result
        assert cached_trajectory is None
        assert cached_output.result == "Agent result"

        # Different candidate should miss
        different_candidate = {
            "instructions": ComponentValue(name="instructions", text="Different"),
        }
        result = cache.get_cached_agent_run(
            case,
            0,
            different_candidate,
            capture_traces=True,
        )
        assert result is None


def test_cache_key_stability():
    """Test that cache keys are stable across different orderings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir, enabled=True, verbose=False)

        case = _prompt_case(
            "Test",
            name="test-1",
            metadata={"b": 2, "a": 1},
        )
        output = RolloutOutput.from_success("Result")

        # Candidates with different key orders but same content
        candidate1 = {
            "instructions": ComponentValue(name="instructions", text="Test"),
            "signature:ExampleSignature:instructions": ComponentValue(
                name="signature:ExampleSignature:instructions",
                text="InputType",
            ),
        }
        candidate2 = {
            "signature:ExampleSignature:instructions": ComponentValue(
                name="signature:ExampleSignature:instructions",
                text="InputType",
            ),
            "instructions": ComponentValue(name="instructions", text="Test"),
        }

        # Cache with first candidate
        cache.cache_metric_result(
            case,
            0,
            output,
            candidate1,
            MetricResult(score=0.9, feedback="Good"),
        )

        # Should get cache hit with reordered candidate
        result = cache.get_cached_metric_result(case, 0, output, candidate2)
        assert result == MetricResult(score=0.9, feedback="Good")

        # Test with reordered metadata
        case2 = _prompt_case(
            "Test",
            name="test-1",
            metadata={"a": 1, "b": 2},
        )

        # Should still get cache hit
        result = cache.get_cached_metric_result(case2, 0, output, candidate1)
        assert result == MetricResult(score=0.9, feedback="Good")

"""Tests for the caching system."""

from __future__ import annotations

import functools
import hashlib
import tempfile
from pathlib import Path

import cloudpickle
import pytest
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_gepa.cache import (
    CacheManager,
    create_cached_metric,
    metric_code_identity,
)
from pydantic_ai_gepa.gepa_graph.models import ComponentValue
from pydantic_ai_gepa.gepa_graph.proposal.instruction import (
    ComponentUpdate,
    InstructionProposalOutput,
    TrajectoryAnalysis,
)
from pydantic_ai_gepa.runner import optimize_agent
from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapter, AgentAdapterTrajectory
from pydantic_ai_gepa.types import MetricResult, ReflectionConfig, RolloutOutput
from pydantic_evals import Case
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class LabelMetadata:
    label: str


@dataclass
class MinLengthEvaluator(Evaluator[str, str, Any]):
    min_length: int = 0

    def evaluate(self, ctx: EvaluatorContext[str, str, Any]) -> bool:
        return len(str(ctx.output)) >= self.min_length


def _identity_metric_a(case: Any, output: Any) -> MetricResult:
    return MetricResult(score=1.0, feedback="metric-a")


def _identity_metric_b(case: Any, output: Any) -> MetricResult:
    return MetricResult(score=0.0, feedback="metric-b")


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
        cache = CacheManager(
            cache_dir=tmpdir,
            enabled=True,
            verbose=False,
            metric_identity="test-metric-v1",
            cache_metric_results=True,
        )

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
        cache = CacheManager(
            cache_dir=tmpdir,
            enabled=True,
            verbose=False,
            metric_identity="test-metric-v1",
            cache_metric_results=True,
        )

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
        cache = CacheManager(
            cache_dir=tmpdir,
            enabled=True,
            verbose=False,
            metric_identity="test-metric-v1",
            cache_metric_results=True,
        )

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
        cache_manager = CacheManager(
            cache_dir=tmpdir,
            enabled=True,
            verbose=False,
            metric_identity="test-metric-v1",
            cache_metric_results=True,
        )

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
async def test_optimize_agent_with_caching(monkeypatch: pytest.MonkeyPatch):
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
        cache_hits: list[str] = []

        original_get_cached_metric_result = CacheManager.get_cached_metric_result

        def tracking_get_cached_metric_result(
            *args: Any, **kwargs: Any
        ) -> MetricResult | None:
            cached_result = original_get_cached_metric_result(*args, **kwargs)
            if cached_result is not None:
                case = args[1]
                cache_hits.append(case.name)
            return cached_result

        monkeypatch.setattr(
            CacheManager,
            "get_cached_metric_result",
            tracking_get_cached_metric_result,
        )

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
            cache_metric_identity="classification-metric-v1",
            cache_metric_results=True,
            cache_rollouts=True,
        )

        first_run_calls = len(metric_calls)
        assert first_run_calls > 0
        assert result1.num_metric_calls <= 15

        # Clear metric calls
        metric_calls.clear()
        cache_hits.clear()

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
            cache_metric_identity="classification-metric-v1",
            cache_metric_results=True,
            cache_rollouts=True,
        )

        # Every baseline evaluation should be reused. The optimizer may spend the
        # freed metric budget evaluating a new proposal, so total metric calls are
        # not itself a reliable cache-hit signal for resumable optimization.
        second_run_calls = len(metric_calls)
        assert set(cache_hits) == {case.name for case in trainset}
        assert second_run_calls <= first_run_calls

        # Results should be consistent
        assert result2.original_candidate == result1.original_candidate


def test_cache_handles_errors():
    """Test that cache handles errors gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(
            cache_dir=tmpdir,
            enabled=True,
            verbose=False,
            metric_identity="test-metric-v1",
            cache_metric_results=True,
        )

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
        cache = CacheManager(
            cache_dir=tmpdir,
            enabled=True,
            verbose=False,
            cache_rollouts=True,
        )

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
        cache = CacheManager(
            cache_dir=tmpdir,
            enabled=True,
            verbose=False,
            metric_identity="test-metric-v1",
            cache_metric_results=True,
        )

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


@pytest.mark.asyncio
async def test_cached_agent_run_does_not_replay_stale_trace_identity() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(
            cache_dir=tmpdir,
            enabled=True,
            verbose=False,
            cache_rollouts=True,
        )
        adapter = AgentAdapter(
            agent=Agent(TestModel(custom_output_text="answer"), instructions="Base"),
            metric=lambda case, output: MetricResult(score=1.0),
            cache_manager=cache,
        )
        case = _prompt_case("Hello", name="trace-cache")
        candidate = {
            "instructions": ComponentValue(name="instructions", text="Optimized")
        }

        first = await adapter.process_case(
            case,
            0,
            capture_traces=True,
            candidate=candidate,
        )
        second = await adapter.process_case(
            case,
            0,
            capture_traces=True,
            candidate=candidate,
        )

        assert first["output"].trace_id is not None
        assert second["output"].trace_id is None
        assert second["output"].trace_completeness is None
        assert second["trajectory"].trace_id is None


def _metric_cache(tmpdir: str, **overrides: Any) -> CacheManager:
    kwargs: dict[str, Any] = {
        "cache_dir": tmpdir,
        "enabled": True,
        "verbose": False,
        "metric_identity": "test-metric-v1",
        "cache_metric_results": True,
    }
    kwargs.update(overrides)
    return CacheManager(**kwargs)


def _instructions_candidate(text: str = "Test instructions"):
    return {"instructions": ComponentValue(name="instructions", text=text)}


def test_metric_cache_misses_when_gold_changes():
    """Editing case.expected_output must invalidate cached metric results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = _metric_cache(tmpdir)
        output = RolloutOutput.from_success("Result")
        candidate = _instructions_candidate()

        case_gold_a = Case(name="case-1", inputs="input", expected_output="gold-a")
        cache.cache_metric_result(
            case_gold_a,
            None,
            output,
            candidate,
            MetricResult(score=1.0, feedback="Correct"),
        )
        assert cache.get_cached_metric_result(
            case_gold_a, None, output, candidate
        ) == MetricResult(score=1.0, feedback="Correct")

        # Same case, output, candidate and identity, but different gold.
        case_gold_b = Case(name="case-1", inputs="input", expected_output="gold-b")
        assert (
            cache.get_cached_metric_result(case_gold_b, None, output, candidate) is None
        )


def test_metric_cache_misses_when_metric_identity_changes():
    """A different metric identity must invalidate cached metric results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        case = _prompt_case("Test", name="test-1")
        output = RolloutOutput.from_success("Result")
        candidate = _instructions_candidate()
        result = MetricResult(score=0.7, feedback="ok")

        cache_v1 = _metric_cache(tmpdir, metric_identity="metric-v1")
        cache_v1.cache_metric_result(case, None, output, candidate, result)
        assert (
            cache_v1.get_cached_metric_result(case, None, output, candidate) == result
        )

        cache_v2 = _metric_cache(tmpdir, metric_identity="metric-v2")
        assert cache_v2.get_cached_metric_result(case, None, output, candidate) is None


def test_metric_code_identity_tracks_function_source():
    """metric_code_identity changes with the function body, not the name."""
    assert metric_code_identity(_identity_metric_a) == metric_code_identity(
        _identity_metric_a
    )
    assert metric_code_identity(_identity_metric_a) != metric_code_identity(
        _identity_metric_b
    )

    # functools.partial layers unwrap to the underlying function.
    partial_metric = functools.partial(_identity_metric_a)
    assert metric_code_identity(partial_metric) == metric_code_identity(
        _identity_metric_a
    )

    # A callable without retrievable source tells the caller to declare a
    # version string instead.
    with pytest.raises(ValueError, match="version string"):
        metric_code_identity(len)


def test_metric_cache_misses_when_evaluators_change():
    """Editing case-level evaluators must invalidate cached metric results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = _metric_cache(tmpdir)
        output = RolloutOutput.from_success("Result")
        candidate = _instructions_candidate()

        case_v1 = Case(
            name="case-1",
            inputs="input",
            expected_output="gold",
            evaluators=[MinLengthEvaluator(min_length=1)],
        )
        cache.cache_metric_result(
            case_v1,
            None,
            output,
            candidate,
            MetricResult(score=1.0, feedback="ok"),
        )
        assert (
            cache.get_cached_metric_result(case_v1, None, output, candidate) is not None
        )

        case_v2 = Case(
            name="case-1",
            inputs="input",
            expected_output="gold",
            evaluators=[MinLengthEvaluator(min_length=2)],
        )
        assert cache.get_cached_metric_result(case_v2, None, output, candidate) is None


def test_cache_requires_explicit_opt_in():
    """An enabled cache with no opt-in (or no identity) fails closed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="would store nothing"):
            CacheManager(cache_dir=tmpdir, enabled=True)

        with pytest.raises(ValueError, match="requires a non-empty metric_identity"):
            CacheManager(cache_dir=tmpdir, enabled=True, cache_metric_results=True)

        with pytest.raises(ValueError, match="requires a non-empty metric_identity"):
            CacheManager(
                cache_dir=tmpdir,
                enabled=True,
                cache_metric_results=True,
                metric_identity="   ",
            )

        # Disabled caches skip validation entirely.
        CacheManager(cache_dir=tmpdir, enabled=False)


def test_rollouts_only_opt_in_stores_no_metric_results():
    """With only cache_rollouts, metric results are never written or read."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir, enabled=True, cache_rollouts=True)
        case = _prompt_case("Test", name="test-1")
        output = RolloutOutput.from_success("Result")
        candidate = _instructions_candidate()

        cache.cache_metric_result(
            case,
            None,
            output,
            candidate,
            MetricResult(score=1.0, feedback="ok"),
        )
        assert list(Path(tmpdir).glob("*.pkl")) == []
        assert cache.get_cached_metric_result(case, None, output, candidate) is None

        cache.cache_agent_run(case, 0, candidate, None, output, capture_traces=False)
        assert len(list(Path(tmpdir).glob("*.pkl"))) == 1
        assert (
            cache.get_cached_agent_run(case, 0, candidate, capture_traces=False)
            is not None
        )


def test_metric_results_only_opt_in_stores_no_agent_runs():
    """With only cache_metric_results, agent runs are never written or read."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = _metric_cache(tmpdir)
        case = _prompt_case("Test", name="test-1")
        output = RolloutOutput.from_success("Result")
        candidate = _instructions_candidate()

        cache.cache_agent_run(case, 0, candidate, None, output, capture_traces=False)
        assert list(Path(tmpdir).glob("*.pkl")) == []
        assert (
            cache.get_cached_agent_run(case, 0, candidate, capture_traces=False) is None
        )

        metric_result = MetricResult(score=1.0, feedback="ok")
        cache.cache_metric_result(case, None, output, candidate, metric_result)
        assert len(list(Path(tmpdir).glob("*.pkl"))) == 1
        assert cache.get_cached_metric_result(case, None, output, candidate) == (
            metric_result
        )


def test_stale_pre_schema_entry_is_not_reused():
    """An entry written under the pre-change key shape never matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = _metric_cache(tmpdir)
        case = _prompt_case("Test", name="test-1")
        output = RolloutOutput.from_success("Result")
        candidate = _instructions_candidate()

        # Simulate a legacy entry: a pickle whose filename is not derivable from
        # the schema-versioned key shape (which now includes schema, metric
        # identity, gold and evaluators).
        legacy_key = hashlib.sha256(b"type:metric|legacy-key-shape").hexdigest()
        with open(Path(tmpdir) / f"{legacy_key}.pkl", "wb") as f:
            cloudpickle.dump(MetricResult(score=0.99, feedback="stale"), f)

        assert cache.get_cached_metric_result(case, None, output, candidate) is None


@pytest.mark.asyncio
async def test_optimize_agent_cache_requires_opt_in_before_any_model_call():
    """enable_cache=True without opt-ins raises before the agent model runs."""
    model_calls = 0

    async def counting_model(messages: Any, info: Any) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(parts=[TextPart(content="done")])

    agent = Agent(FunctionModel(counting_model), instructions="seed")

    def metric(case: Any, output: Any) -> MetricResult:
        return MetricResult(score=1.0)

    with pytest.raises(ValueError, match="would store nothing"):
        await optimize_agent(
            agent=agent,
            trainset=[Case(name="case-1", inputs="input")],
            metric=metric,
            enable_cache=True,
        )

    assert model_calls == 0


@pytest.mark.asyncio
async def test_optimize_agent_cache_invalidation_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
):
    """A changed metric identity re-runs the metric; an unchanged run reuses it."""
    with tempfile.TemporaryDirectory() as tmpdir:
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
        case_names = {case.name for case in trainset}

        metric_calls: list[str] = []
        cache_hits: list[str] = []

        original_get_cached_metric_result = CacheManager.get_cached_metric_result

        def tracking_get_cached_metric_result(
            *args: Any, **kwargs: Any
        ) -> MetricResult | None:
            cached_result = original_get_cached_metric_result(*args, **kwargs)
            if cached_result is not None:
                case = args[1]
                cache_hits.append(case.name)
            return cached_result

        monkeypatch.setattr(
            CacheManager,
            "get_cached_metric_result",
            tracking_get_cached_metric_result,
        )

        def metric(case, output):
            metric_calls.append(case.name)
            predicted = str(output.result).lower() if output.success else ""
            metadata = case.metadata or LabelMetadata(label="")
            expected = metadata.label.lower()
            score = 1.0 if predicted == expected else 0.0
            return MetricResult(score=score, feedback=f"Score: {score}")

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

        async def run(identity: str):
            return await optimize_agent(
                agent=agent,
                trainset=trainset,
                metric=metric,
                reflection_config=ReflectionConfig(model=reflection_model),
                max_metric_calls=15,
                seed=42,
                enable_cache=True,
                cache_dir=tmpdir,
                cache_verbose=False,
                cache_metric_identity=identity,
                cache_metric_results=True,
                cache_rollouts=True,
            )

        # First run populates the cache under identity v1.
        await run("classification-metric-v1")
        assert metric_calls

        # Second run with the same identity reuses cached metric results.
        metric_calls.clear()
        cache_hits.clear()
        await run("classification-metric-v1")
        assert set(cache_hits) == case_names

        # Third run with a changed identity calls the metric again for every case.
        metric_calls.clear()
        cache_hits.clear()
        await run("classification-metric-v2")
        assert set(metric_calls) == case_names

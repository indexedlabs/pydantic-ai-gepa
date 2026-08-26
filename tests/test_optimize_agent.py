"""End-to-end test of optimize_agent using a small pydantic_evals dataset.

This exercises a full (minimal) GEPA optimization flow with:
- TestModel() as both the agent model and the reflection model
- A tiny categorization dataset (10 items)
- A low metric budget to keep the run short
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai_gepa.components import (
    extract_seed_candidate,
    extract_seed_candidate_with_input_type,
)
from pydantic_ai_gepa.gepa_graph.proposal.instruction import (
    ComponentUpdate,
    InstructionProposalOutput,
    TrajectoryAnalysis,
)
from pydantic_ai_gepa.runner import optimize_agent
from pydantic_ai_gepa.signature_agent import SignatureAgent
from pydantic_ai_gepa.types import (
    MetricResult,
    ReflectionConfig,
    RolloutOutput,
)

from pydantic_ai import Agent, usage as _usage
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset


def _fake_reasoning() -> TrajectoryAnalysis:
    return TrajectoryAnalysis(
        pattern_discovery="baseline patterns observed in testing",
        creative_hypothesis="placeholder hypothesis for unit tests",
        experimental_approach="placeholder approach for unit tests",
    )


@pytest.mark.asyncio
async def test_optimize_agent_minimal_flow():
    """Run a minimal optimization flow over a tiny categorization dataset.

    We don't expect meaningful optimization here; we just validate a complete run
    finishes and returns a structured result within a small metric budget.
    """

    # Build a small categorization dataset (10 items) using pydantic_evals
    def _label_for_token(token: str) -> str:
        return {
            "good": "positive",
            "bad": "negative",
            "ok": "neutral",
        }[token]

    tokens = [
        "good" if i % 3 == 0 else ("bad" if i % 3 == 1 else "ok") for i in range(10)
    ]

    dataset = Dataset(
        name="optimize-agent",
        cases=[
            Case(
                name=f"case-{i}",
                inputs=(
                    "Categorize the following input as one of: positive, negative, or neutral.\n"
                    f"Input: Sample {i} describing something {tok}."
                ),
                expected_output=_label_for_token(tok),
            )
            for i, tok in enumerate(tokens)
        ],
    )

    # Agent returns a fixed label; we are not testing real model behavior here
    agent = Agent(
        TestModel(custom_output_text="neutral"),
        instructions=(
            "You are a concise classifier. Output exactly one of: positive, negative, neutral."
        ),
    )

    seed = extract_seed_candidate(agent)

    # Simple metric: 1.0 if predicted label matches expected label, else 0.0
    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        predicted = (
            str(output.result).strip().lower()
            if output.success and output.result is not None
            else ""
        )
        expected = str(case.expected_output or "").strip().lower()
        score = 1.0 if predicted == expected else 0.0
        return MetricResult(
            score=score,
            feedback="Correct" if score == 1.0 else "Incorrect",
        )

    reflection_output = InstructionProposalOutput(
        reasoning=_fake_reasoning(),
        updated_components=[
            ComponentUpdate(
                component_name="instructions",
                optimized_value="Optimized",
            )
        ],
    )
    reflection_model = TestModel(
        custom_output_args=reflection_output.model_dump(mode="python")
    )

    # Keep the budget low; use TestModel() for the reflection model to exercise the full path
    result = await optimize_agent(
        agent=agent,
        trainset=dataset,
        metric=metric,
        reflection_config=ReflectionConfig(model=reflection_model),
        max_metric_calls=20,
        seed=0,
    )

    # Basic result shape checks
    assert isinstance(result.best_candidate, dict)
    assert "instructions" in result.best_candidate
    assert isinstance(result.best_score, float)
    assert result.original_candidate == seed
    assert result.num_metric_calls > 0
    assert result.num_metric_calls <= 30


@pytest.mark.asyncio
async def test_optimize_agent_minimal_flow_with_signature():
    """Run a minimal optimization flow over a tiny categorization dataset.

    We don't expect meaningful optimization here; we just validate a complete run
    finishes and returns a structured result within a small metric budget.
    """

    class Input(BaseModel):
        text: str

    # Build a small categorization dataset (10 items) using pydantic_evals
    def _label_for_token(token: str) -> str:
        return {
            "good": "positive",
            "bad": "negative",
            "ok": "neutral",
        }[token]

    tokens = [
        "good" if i % 3 == 0 else ("bad" if i % 3 == 1 else "ok") for i in range(10)
    ]

    dataset = Dataset(
        name="optimize-signature-agent",
        cases=[
            Case(
                name=f"case-{i}",
                inputs=Input(text=f"Sample {i} describing something {tok}."),
                expected_output=_label_for_token(tok),
            )
            for i, tok in enumerate(tokens)
        ],
    )

    # Agent returns a fixed label; we are not testing real model behavior here
    agent = Agent(
        TestModel(custom_output_text="neutral"),
        instructions=(
            "You are a concise classifier. Output exactly one of: positive, negative, neutral."
        ),
    )
    signature_agent = SignatureAgent(
        agent,
        input_type=Input,
    )

    seed = extract_seed_candidate_with_input_type(signature_agent, input_type=Input)

    # Simple metric: 1.0 if predicted label matches expected label, else 0.0
    def metric(case: Case[Input, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        predicted = (
            str(output.result).strip().lower()
            if output.success and output.result is not None
            else ""
        )
        expected = str(case.expected_output or "").strip().lower()
        score = 1.0 if predicted == expected else 0.0
        return MetricResult(
            score=score,
            feedback="Correct" if score == 1.0 else "Incorrect",
        )

    reflection_output = InstructionProposalOutput(
        reasoning=_fake_reasoning(),
        updated_components=[
            ComponentUpdate(
                component_name="instructions",
                optimized_value="Optimized",
            )
        ],
    )
    reflection_model = TestModel(
        custom_output_args=reflection_output.model_dump(mode="python")
    )

    # Keep the budget low; use TestModel() for the reflection model to exercise the full path
    result = await optimize_agent(
        agent=signature_agent,
        trainset=dataset,
        input_type=Input,
        metric=metric,
        reflection_config=ReflectionConfig(model=reflection_model),
        max_metric_calls=20,
        seed=0,
    )

    # Basic result shape checks
    assert isinstance(result.best_candidate, dict)
    assert "instructions" in result.best_candidate
    assert isinstance(result.best_score, float)
    assert result.original_candidate == seed
    assert result.num_metric_calls > 0
    assert result.num_metric_calls <= 30


@pytest.mark.asyncio
async def test_optimize_agent_reports_progress(monkeypatch: pytest.MonkeyPatch):
    """Ensure enabling show_progress triggers progress updates."""

    agent = Agent(
        TestModel(custom_output_text="ok"),
        instructions="Say ok.",
    )

    trainset = [Case(name="case-0", inputs="Hello", expected_output="ok")]

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        del case  # unused
        return MetricResult(score=1.0, feedback="Fine")

    reflection_output = InstructionProposalOutput(
        reasoning=_fake_reasoning(),
        updated_components=[
            ComponentUpdate(
                component_name="instructions",
                optimized_value="Better instructions",
            )
        ],
    )
    reflection_model = TestModel(
        custom_output_args=reflection_output.model_dump(mode="python")
    )

    updates: list[int] = []

    class _StubProgress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._updates = updates

        def __enter__(self) -> "_StubProgress":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def add_task(self, description: str, total: int) -> int:
            self.description = description
            self.total = total
            return 7

        def update(self, task_id: int, **kwargs: Any) -> None:
            self._updates.append(kwargs.get("completed", 0))

    monkeypatch.setattr("pydantic_ai_gepa.progress.Progress", _StubProgress)

    result = await optimize_agent(
        agent=agent,
        trainset=trainset,
        metric=metric,
        reflection_config=ReflectionConfig(model=reflection_model),
        max_metric_calls=5,
        show_progress=True,
        seed=0,
    )

    assert result.num_metric_calls > 0
    assert updates, "Expected at least one progress update"
    assert updates[-1] == min(result.num_metric_calls, 5)


@pytest.mark.asyncio
async def test_optimize_agent_respects_agent_usage_limits():
    """Per-run UsageLimits should be enforced for each agent evaluation."""

    trainset = [
        Case(
            name="usage-case",
            inputs="Respond with anything.",
            expected_output="irrelevant",
        )
    ]

    agent = Agent(
        TestModel(custom_output_text="ok"),
        instructions="Always respond with 'ok'.",
    )

    reflection_output = InstructionProposalOutput(
        reasoning=_fake_reasoning(),
        updated_components=[
            ComponentUpdate(
                component_name="instructions",
                optimized_value="Updated instructions",
            )
        ],
    )
    reflection_model = TestModel(
        custom_output_args=reflection_output.model_dump(mode="python")
    )

    metric_outputs: list[bool] = []

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        del case  # unused
        metric_outputs.append(output.success)
        # Hitting the per-run request limit should yield a failed rollout.
        assert output.success is False
        return MetricResult(score=0.0, feedback="usage limited")

    result = await optimize_agent(
        agent=agent,
        trainset=trainset,
        metric=metric,
        reflection_config=ReflectionConfig(model=reflection_model),
        max_metric_calls=3,
        agent_usage_limits=_usage.UsageLimits(request_limit=0),
        seed=0,
    )

    assert metric_outputs, "metric should have been invoked at least once"
    assert all(success is False for success in metric_outputs)
    assert result.num_metric_calls == len(metric_outputs)


@pytest.mark.asyncio
async def test_optimize_agent_stops_on_gepa_usage_budget():
    """The overall usage budget should stop the optimizer once exceeded."""

    trainset = [
        Case(name=f"budget-case-{i}", inputs=f"Prompt {i}", expected_output="ok")
        for i in range(3)
    ]

    agent = Agent(
        TestModel(custom_output_text="ok"),
        instructions="Respond with ok.",
    )

    reflection_output = InstructionProposalOutput(
        reasoning=_fake_reasoning(),
        updated_components=[
            ComponentUpdate(
                component_name="instructions",
                optimized_value="Updated instructions",
            )
        ],
    )
    reflection_model = TestModel(
        custom_output_args=reflection_output.model_dump(mode="python")
    )

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        del case  # unused
        return MetricResult(score=1.0 if output.success else 0.0, feedback="ok")

    result = await optimize_agent(
        agent=agent,
        trainset=trainset,
        metric=metric,
        reflection_config=ReflectionConfig(model=reflection_model),
        max_metric_calls=50,
        gepa_usage_limits=_usage.UsageLimits(request_limit=1),
        seed=0,
    )

    assert result.raw_result is not None
    assert result.raw_result.stop_reason == "Usage budget exceeded"
    assert result.raw_result.stopped is True

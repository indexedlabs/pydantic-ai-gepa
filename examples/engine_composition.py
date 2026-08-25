"""Compare built-in optimization engines on a tiny classification task.

Run hermetically with ``uv run python examples/engine_composition.py``. Set
``OPENAI_API_KEY`` to include a small GEPA run with real models. Override the
student (``GEPA_STUDENT_MODEL``) and teacher/reflection (``GEPA_TEACHER_MODEL``)
models to pair a weaker student with a stronger teacher.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa import (
    EngineConfig,
    MetricResult,
    OptimizationTask,
    ReflectionConfig,
    RolloutOutput,
    optimize_best_of,
)
from pydantic_ai_gepa.gepa_graph.models import CandidateMap, ComponentValue


class ClassificationOutput(BaseModel):
    label: Literal["positive", "negative", "neutral"]


CASES = [
    Case(
        name="positive",
        inputs="The delivery was wonderful.",
        expected_output=ClassificationOutput(label="positive"),
    ),
    Case(
        name="negative",
        inputs="The service was disappointing.",
        expected_output=ClassificationOutput(label="negative"),
    ),
    Case(
        name="neutral",
        inputs="The order arrived on Tuesday.",
        expected_output=ClassificationOutput(label="neutral"),
    ),
]


def has_real_model() -> bool:
    """Only opt into a networked model when its standard credential is present."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def student_model() -> str:
    """The model that runs the agent being optimized.

    Defaults to the Responses API because newer reasoning models reject
    function-tool (structured-output) calls on the chat-completions endpoint.
    """
    return os.environ.get("GEPA_STUDENT_MODEL", "openai-responses:gpt-4o-mini")


def teacher_model() -> str:
    """The model that drives reflection; defaults to the student model."""
    return os.environ.get("GEPA_TEACHER_MODEL", student_model())


def make_agent() -> Agent[object, ClassificationOutput]:
    """Use the same environment-variable model override pattern as other demos."""
    if has_real_model():
        model = student_model()
    else:
        model = TestModel(custom_output_args={"label": "positive"})

    return Agent(
        model,
        output_type=ClassificationOutput,
        instructions=(
            "Classify the sentiment as exactly one of positive, negative, or neutral."
        ),
    )


def accuracy(
    case: Case[str, ClassificationOutput, Any],
    output: RolloutOutput[ClassificationOutput],
) -> MetricResult:
    """Score exact label accuracy and retain useful feedback for reflective engines."""
    expected = case.expected_output.label if case.expected_output else None
    actual = output.result.label if output.success and output.result else None
    score = float(actual == expected)
    return MetricResult(
        score=score,
        feedback="Correct." if score else f"Expected {expected}, got {actual}.",
    )


async def scripted_proposer(seed: CandidateMap) -> CandidateMap:
    """Return a deterministic prompt variant for the reference engine."""
    candidate = {name: value.model_copy(deep=True) for name, value in seed.items()}
    candidate["instructions"] = ComponentValue(
        name="instructions",
        text=(
            "Classify the sentiment of the text as positive, negative, or neutral. "
            "Return only the requested structured label."
        ),
    )
    return candidate


async def main() -> None:
    agent = make_agent()
    task = OptimizationTask(
        agent=agent,
        trainset=CASES,
        valset=CASES,
        metric=accuracy,
    )
    configs = [
        EngineConfig(
            engine="best_of_n",
            max_metric_calls=6,
            engine_config={"n": 1, "propose": scripted_proposer},
        )
    ]
    if has_real_model():
        configs.append(
            EngineConfig(
                engine="gepa",
                max_metric_calls=6,
                max_iterations=1,
                engine_config={
                    "reflection_config": ReflectionConfig(model=teacher_model())
                },
            )
        )

    result = await optimize_best_of(task, configs, max_metric_calls=6 * len(configs))

    model_kind = (
        f"student={student_model()} teacher={teacher_model()}"
        if has_real_model()
        else "TestModel fallback"
    )
    print(f"Model: {model_kind}")
    for engine_result, score in zip(result.results, result.fair_scores):
        print(f"{engine_result.engine}: fair valset accuracy {score:.2f}")
    print(f"Winner: {result.best.engine} ({result.fair_scores[result.best_index]:.2f})")
    print(f"Shared metric calls: {result.total_metric_calls}")


if __name__ == "__main__":
    asyncio.run(main())

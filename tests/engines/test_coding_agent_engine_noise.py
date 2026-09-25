"""Seeded no-effect simulation for the coding-agent engine's training gate.

Mirrors tests/cli/test_acceptance_run_noise.py: proposals have exactly the
parent's clipped-Gaussian score distribution, so every training-gate pass is
a false acceptance. The failure-selecting minibatch is charged to the budget
but excluded from the baseline samples, so the measured rate must stay at or
below the one-sided rate ``compare_candidate_samples`` is configured for.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa.engines import (
    BudgetTracker,
    EngineConfig,
    OptimizationTask,
    get_engine,
)
from pydantic_ai_gepa.engines.coding_agent_engine import ReflectionContext
from pydantic_ai_gepa.gepa_graph.models import CandidateMap, ComponentValue
from pydantic_ai_gepa.types import MetricResult, RolloutOutput

_CENTER = 0.8
_SD = 0.15
_THRESHOLD = 0.75
_CONFIDENCE = 0.9
_RUNS = 180
_MAX_PROPOSALS_PER_RUN = 2
_BASE_SEED = 4749


async def _run_once(run_seed: int) -> tuple[int, int]:
    """Run one no-effect engine run; return (proposals, training-gate passes)."""
    agent = Agent(TestModel(custom_output_text="response"), instructions="seed")
    train = [
        Case(name=f"train-{index}", inputs="x", expected_output="response")
        for index in range(6)
    ]
    val = [Case(name="val-0", inputs="x", expected_output="response")]
    counts: dict[tuple[str, str], int] = {}

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        del output
        override = agent._override_instructions.get()
        instructions = override.value if override is not None else agent._instructions
        key = (case.name, "|".join(str(item) for item in instructions))
        index = counts.get(key, 0)
        counts[key] = index + 1
        # Identically distributed for parent and proposal, deterministic per
        # (run, case, candidate, repetition) regardless of async scheduling.
        rng = random.Random(f"{run_seed}:{key[0]}:{key[1]}:{index}")
        score = min(1.0, max(0.0, rng.gauss(_CENTER, _SD)))
        return MetricResult(score=score, feedback=f"score {score:.3f}")

    task = OptimizationTask(agent=agent, trainset=train, valset=val, metric=metric)
    counter = 0

    async def propose(context: ReflectionContext) -> CandidateMap:
        del context
        nonlocal counter
        counter += 1
        return {
            "instructions": ComponentValue(
                name="instructions", text=f"proposal-{run_seed}-{counter}"
            )
        }

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=400,
        seed=run_seed,
        engine_config={
            "propose": propose,
            "minibatch_size": 1,
            "max_proposals_per_run": _MAX_PROPOSALS_PER_RUN,
            "failure_threshold": _THRESHOLD,
            "acceptance_repetitions": 3,
            "acceptance_max_repetitions": 3,
            "acceptance_confidence": _CONFIDENCE,
        },
    )
    result = await get_engine("coding_agent", config).run(
        task, config, BudgetTracker(400)
    )
    improved = sum(1 for event in result.history if event.kind == "minibatch_improved")
    return result.history[-1].data["proposals"], improved


@pytest.mark.asyncio
async def test_no_effect_proposals_respect_the_configured_acceptance_rate() -> None:
    """180 seeded runs, up to 2 proposals each, all with the parent distribution.

    Per-case scores are clipped Gaussian (center 0.8, sd 0.15); a case fails
    below 0.75, so reflection really selects low-scoring batches. The gate is
    a single-look, 3-repetition Welch comparison at confidence 0.9, i.e. a
    configured one-sided false-acceptance rate of 0.05. Before the selection
    batch was excluded from the baseline samples, the same seeds and run
    count produced a 0.067 false-acceptance rate.
    """
    proposals = 0
    improved = 0
    for run_index in range(_RUNS):
        run_proposals, run_improved = await _run_once(_BASE_SEED + run_index)
        proposals += run_proposals
        improved += run_improved

    assert proposals > 0, "the simulation must exercise the training gate"
    rate = improved / proposals
    configured_rate = (1.0 - _CONFIDENCE) / 2
    # Three binomial standard errors of tolerance above the configured rate.
    tolerance = 3 * (configured_rate * (1 - configured_rate) / proposals) ** 0.5
    print(
        f"engine no-effect: {improved}/{proposals} proposals passed "
        f"({rate:.3%}); configured {configured_rate:.3%} + {tolerance:.3%}"
    )
    assert rate <= configured_rate + tolerance

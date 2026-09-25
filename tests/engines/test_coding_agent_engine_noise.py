"""Seeded no-effect simulation for the coding-agent engine's training gate.

Mirrors tests/cli/test_acceptance_run_noise.py: proposals have exactly the
parent's clipped-Gaussian score distribution, so every training-gate pass is
a false acceptance. The failure-selecting minibatch is charged to the budget
but excluded from the baseline samples, so the measured rate must stay at or
below the one-sided rate ``compare_candidate_samples`` is configured for.

Configuration: minibatch of one case, per-case scores clipped Gaussian
(center 0.85, sd 0.15), failure threshold 0.75, a single-look 3-repetition
Welch comparison at confidence 0.7 (configured one-sided rate 0.15; the
looser confidence puts the gate on a steeper part of the t curve, where
the leak is visible at a feasible sample size), 40 seeded runs with up to
10 proposals each. Measured with these seeds: the old engine (selection
batch counted as evidence) passed the gate for 101/396 proposals (25.5%),
clearly above the ~0.204 bound, while the fixed engine passed for 69/400
(17.2%) and stays below it. The assertion therefore fails on the buggy
code and passes on the fixed code.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
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

_CENTER = 0.85
_SD = 0.15
_THRESHOLD = 0.75
_CONFIDENCE = 0.7
_RUNS = 40
_MAX_PROPOSALS_PER_RUN = 10
_BASE_SEED = 4749


def _echo_model(messages: Any, info: Any) -> ModelResponse:
    """Echo the applied instructions so the rollout output names the candidate."""
    del info
    instructions = getattr(messages[-1], "instructions", None)
    return ModelResponse(parts=[TextPart(content=f"echo:{instructions}")])


async def _run_once(run_seed: int) -> tuple[int, int]:
    """Run one no-effect engine run; return (proposals, training-gate passes)."""
    agent = Agent(FunctionModel(_echo_model), instructions="seed")
    train = [
        Case(name=f"train-{index}", inputs="x", expected_output="echo:seed")
        for index in range(6)
    ]
    val = [Case(name="val-0", inputs="x", expected_output="echo:seed")]
    counts: dict[tuple[str, str], int] = {}

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        # The echoed instructions in the public rollout result identify the
        # candidate, so no private agent attributes are needed here.
        key = (case.name, str(output.result))
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
        max_metric_calls=200,
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
        task, config, BudgetTracker(200)
    )
    improved = sum(1 for event in result.history if event.kind == "minibatch_improved")
    return result.history[-1].data["proposals"], improved


@pytest.mark.asyncio
async def test_no_effect_proposals_respect_the_configured_acceptance_rate() -> None:
    """40 seeded runs, up to 10 proposals each, all with the parent distribution.

    A case fails below 0.75 (about 25% of draws), so reflection really
    selects low-scoring batches. The gate is a single-look, 3-repetition
    Welch comparison at confidence 0.7, i.e. a configured one-sided
    false-acceptance rate of 0.15; the bound adds three binomial standard
    errors. On the old engine, which counted the failure-selected batch as
    evidence, this seeded configuration fails this assertion; the fixed
    engine stays under the bound.
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

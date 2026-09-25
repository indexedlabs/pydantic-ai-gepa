from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import logfire
from pydantic_ai import Agent

from pydantic_ai_gepa import (
    MetricResult,
    ReflectionConfig,
    RolloutOutput,
    metric_code_identity,
    optimize_agent,
)
from pydantic_ai_gepa.types import Case


_INT_RE = re.compile(r"^\s*(\d+)\s*$")


def metric(case: Case[str, str, None], output: RolloutOutput[str]) -> MetricResult:
    expected = str(case.expected_output).strip()
    if not output.success or output.result is None:
        return MetricResult(score=0.0, feedback="No successful result produced.")

    text = str(output.result)
    m = _INT_RE.match(text)
    if not m:
        return MetricResult(
            score=0.0,
            feedback="Output must be only the integer (digits only, no extra text).",
        )

    got = m.group(1)
    if got == expected:
        return MetricResult(score=1.0, feedback="Correct.")
    return MetricResult(score=0.0, feedback=f"Expected {expected}, got {got}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GEPA skills optimization example."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("optimization_results"),
        help="Directory to write optimization result JSON files.",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=500,
        help="Maximum number of GEPA metric evaluations to run before stopping.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum number of GEPA loop iterations to run before stopping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


async def main(
    *,
    results_dir: Path,
    max_evaluations: int,
    max_iterations: int,
    seed: int,
) -> None:
    logfire.configure(console=False)
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)

    skills_dir = Path(__file__).parent / "skills"
    reflection_model = os.environ.get("GEPA_REFLECTION_MODEL", "openai:gpt-4o-mini")

    agent = Agent(
        model=os.environ.get("GEPA_STUDENT_MODEL", "openai:gpt-4o-mini"),
        output_type=str,
        instructions=(
            "You are solving short conversion tasks.\n\n"
            "You have access to skills tools:\n"
            "- search_skills(query): find relevant skills\n"
            "- load_skill(skill_path): read the full SKILL.md\n"
            "- load_skill_file(skill_path, path): read a file within a skill\n\n"
            "For each task, search for a relevant skill, load it, and follow it.\n"
            "Return only the final answer.\n"
        ),
    )

    train = [
        Case(name="roman-ix", inputs="Convert IX to an integer.", expected_output="9"),
        Case(
            name="roman-xiv", inputs="Convert XIV to an integer.", expected_output="14"
        ),
        Case(name="roman-xl", inputs="Convert XL to an integer.", expected_output="40"),
        Case(
            name="roman-mcmxciv",
            inputs="Convert MCMXCIV to an integer.",
            expected_output="1994",
        ),
    ]

    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"skills_optimization_{timestamp}.json"

    with logfire.span(
        "skills optimization run",
        run_id=run_id,
        student_model=str(getattr(agent, "model", None)),
        reflection_model=str(reflection_model),
        max_evaluations=max_evaluations,
        max_iterations=max_iterations,
        seed=seed,
    ):
        result = await optimize_agent(
            agent=agent,
            trainset=train[:3],
            valset=train[3:],
            metric=metric,
            skills=skills_dir,
            module_selector="reflection",
            reflection_config=ReflectionConfig(model=reflection_model),
            max_metric_calls=max_evaluations,
            max_iterations=max_iterations,
            enable_cache=True,
            cache_dir=".gepa_cache",
            cache_verbose=False,
            # The metric is deterministic (regex integer comparison), so
            # caching its results is sound; the code-identity hash invalidates
            # them whenever the metric body changes.
            cache_metric_identity=metric_code_identity(metric),
            cache_metric_results=True,
            cache_rollouts=True,
            show_progress=True,
            seed=seed,
        )

    with output_file.open("w") as f:
        json.dump(
            {
                "run_id": run_id,
                "result": result.model_dump(mode="python"),
            },
            f,
            indent=2,
        )

    score_label = f"{result.best_score:.3f}" if result.best_score is not None else "n/a"
    print(f"Best score: {score_label}")
    print(f"Metric calls: {result.num_metric_calls}")
    print(f"Iterations: {result.num_iterations}")
    print(f"Run id: {run_id}")
    print(f"✅ Optimization result saved to: {output_file}")
    print("Updated skill components:")
    for name in sorted(result.best_candidate.keys()):
        if name.startswith("skill:"):
            print(f"- {name}")

    # Optional: in-process indexed search
    #
    #   from pydantic_ai_gepa.skills.search import InMemorySkillsSearchProvider
    #   provider = InMemorySkillsSearchProvider()
    #   await provider.reindex_skills(fs=skills_dir_fs, skill_paths=[...])
    #   result = await optimize_agent(..., skills_search_backend=provider)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            results_dir=args.results_dir,
            max_evaluations=args.max_evaluations,
            max_iterations=args.max_iterations,
            seed=args.seed,
        )
    )

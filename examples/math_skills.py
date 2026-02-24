from __future__ import annotations

import argparse
import asyncio
import json
import pprint
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.models import KnownModelName, Model, infer_model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_evals import Case, Dataset
from utils import run_python_tool

from pydantic_ai_gepa import InspectionAborted
from pydantic_ai_gepa.gepa_graph.models import CandidateMap
from pydantic_ai_gepa.adapters import SignatureAgentAdapter
from pydantic_ai_gepa.cache import CacheManager
from pydantic_ai_gepa.evaluation import EvaluationRecord, evaluate_candidate_dataset
from pydantic_ai_gepa.gepa_graph import (
    CandidateSelectorStrategy,
    GepaConfig,
    GepaResult,
    optimize,
)
from pydantic_ai_gepa.gepa_graph.models import CandidateProgram
from pydantic_ai_gepa.skills.fs import SkillsFS
from pydantic_ai_gepa.signature_agent import SignatureAgent
from pydantic_ai_gepa.types import (
    ExampleBankConfig,
    MetricResult,
    ReflectionConfig,
    RolloutOutput,
)

logfire.configure(console=False)
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)


class MathProblemInput(BaseModel):
    problem: str = Field(
        description="A math problem that needs an exact numeric answer."
    )


class MathProblemOutput(BaseModel):
    """The solved value and the code that produced it."""

    explanation: str = Field(
        description="Two sentences max summarizing how the code solves the problem."
    )
    expression: str = Field(
        description="The complete Python script used to compute the answer (can be multi-line with imports)."
    )
    answer: float = Field(description="Numeric answer rounded only if necessary.")


@dataclass
class MathProblemMetadata:
    expected_answer: float
    tolerance: float = 1e-9
    feedback: str | None = None
    ideal_expression: str | None = None


# Challenging math cases spanning boundary, reasoning, and degenerate scenarios
dataset = Dataset[MathProblemInput, MathProblemOutput, MathProblemMetadata](
    cases=[
        Case(
            name="comb-100-5",
            inputs=MathProblemInput(problem="Compute 100 choose 5."),
            expected_output=MathProblemOutput(
                answer=75287520.0,
                expression="math.comb(100, 5)",
                explanation="Use the combinatorics function from the math module to compute binomial coefficients directly.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=75287520.0,
                tolerance=1e-09,
                feedback="Use the combinatorics function from the math module to compute binomial coefficients directly.",
                ideal_expression="math.comb(100, 5)",
            ),
        ),
        Case(
            name="digit-sum-2-200",
            inputs=MathProblemInput(
                problem="Compute the sum of the digits of 2 raised to the 200th power."
            ),
            expected_output=MathProblemOutput(
                answer=256.0,
                expression="sum(int(d) for d in str(2 ** 200))",
                explanation="Convert the large integer to a string first, then sum each digit character after converting back to int.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=256.0,
                tolerance=1e-09,
                feedback="Convert the large integer to a string first, then sum each digit character after converting back to int.",
                ideal_expression="sum(int(d) for d in str(2 ** 200))",
            ),
        ),
        Case(
            name="primorial-product",
            inputs=MathProblemInput(
                problem="Multiply the primes [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] together."
            ),
            expected_output=MathProblemOutput(
                answer=6469693230.0,
                expression="math.prod([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])",
                explanation="Use the product aggregation function from the math module to multiply list elements.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=6469693230.0,
                tolerance=1e-09,
                feedback="Use the product aggregation function from the math module to multiply list elements.",
                ideal_expression="math.prod([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])",
            ),
        ),
        Case(
            name="range-ambiguity-between",
            inputs=MathProblemInput(problem="Sum all integers between 10 and 20."),
            expected_output=MathProblemOutput(
                answer=135.0,
                expression="sum(range(11, 20))",
                explanation="The phrase 'between A and B' typically excludes both endpoints. Verify whether the count matches your interpretation.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=135.0,
                tolerance=1e-09,
                feedback="The phrase 'between A and B' typically excludes both endpoints. Verify whether the count matches your interpretation.",
                ideal_expression="sum(range(11, 20))",
            ),
        ),
        Case(
            name="range-ambiguity-from-through",
            inputs=MathProblemInput(
                problem="Calculate the average of squares from 5 through 15."
            ),
            expected_output=MathProblemOutput(
                answer=110.0,
                expression="sum(n**2 for n in range(5, 16)) / len(range(5, 16))",
                explanation="The phrase 'from A through B' indicates inclusive bounds. Check that your range includes both endpoints.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=110.0,
                tolerance=1e-09,
                feedback="The phrase 'from A through B' indicates inclusive bounds. Check that your range includes both endpoints.",
                ideal_expression="sum(n**2 for n in range(5, 16)) / len(range(5, 16))",
            ),
        ),
        Case(
            name="implicit-inclusive-up-to",
            inputs=MathProblemInput(
                problem="Find the product of all even numbers up to 12."
            ),
            expected_output=MathProblemOutput(
                answer=46080.0,
                expression="math.prod(range(2, 13, 2))",
                explanation="The phrase 'up to N' is ambiguous\u2014it may include or exclude N. Verify against the expected result which interpretation is correct.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=46080.0,
                tolerance=1e-09,
                feedback="The phrase 'up to N' is ambiguous\u2014it may include or exclude N. Verify against the expected result which interpretation is correct.",
                ideal_expression="math.prod(range(2, 13, 2))",
            ),
        ),
        Case(
            name="rounding-specification",
            inputs=MathProblemInput(
                problem="Approximate the square root of 50 to the nearest integer."
            ),
            expected_output=MathProblemOutput(
                answer=7.0,
                expression="round(math.sqrt(50))",
                explanation="Use the rounding function explicitly when the problem requests rounding to a specific precision.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=7.0,
                tolerance=1e-09,
                feedback="Use the rounding function explicitly when the problem requests rounding to a specific precision.",
                ideal_expression="round(math.sqrt(50))",
            ),
        ),
        Case(
            name="floor-vs-truncate",
            inputs=MathProblemInput(problem="What is 100 divided by 7, rounded down?"),
            expected_output=MathProblemOutput(
                answer=14.0,
                expression="math.floor(100 / 7)",
                explanation="Rounded down means floor division. Use the appropriate math function for floor operations.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=14.0,
                tolerance=1e-09,
                feedback="Rounded down means floor division. Use the appropriate math function for floor operations.",
                ideal_expression="math.floor(100 / 7)",
            ),
        ),
        Case(
            name="mixed-boundaries",
            inputs=MathProblemInput(
                problem="Sum integers greater than 5 and less than or equal to 15."
            ),
            expected_output=MathProblemOutput(
                answer=105.0,
                expression="sum(range(6, 16))",
                explanation="Pay attention to strict inequalities (>) versus inclusive inequalities (\u2264). Translate each bound correctly.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=105.0,
                tolerance=1e-09,
                feedback="Pay attention to strict inequalities (>) versus inclusive inequalities (\u2264). Translate each bound correctly.",
                ideal_expression="sum(range(6, 16))",
            ),
        ),
        Case(
            name="conditional-prime-product",
            inputs=MathProblemInput(
                problem="Find the sum of all primes less than 50, then multiply that sum by the largest prime less than 50."
            ),
            expected_output=MathProblemOutput(
                answer=15416.0,
                expression="(lambda primes: sum(primes) * max(primes))([n for n in range(2, 50) if all(n % d for d in range(2, int(n**0.5) + 1))])",
                explanation="Break the problem into steps: first identify all primes in the range, then compute the sum and find the maximum, then multiply them.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=15416.0,
                tolerance=1e-09,
                feedback="Break the problem into steps: first identify all primes in the range, then compute the sum and find the maximum, then multiply them.",
                ideal_expression="(lambda primes: sum(primes) * max(primes))([n for n in range(2, 50) if all(n % d for d in range(2, int(n**0.5) + 1))])",
            ),
        ),
        Case(
            name="nested-digit-sum",
            inputs=MathProblemInput(
                problem="Calculate the sum of the digits of 15 factorial."
            ),
            expected_output=MathProblemOutput(
                answer=45.0,
                expression="sum(int(d) for d in str(math.factorial(15)))",
                explanation="Compute the factorial first, convert to string, then sum the individual digit characters.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=45.0,
                tolerance=1e-09,
                feedback="Compute the factorial first, convert to string, then sum the individual digit characters.",
                ideal_expression="sum(int(d) for d in str(math.factorial(15)))",
            ),
        ),
        Case(
            name="tribonacci-20",
            inputs=MathProblemInput(
                problem="Find the 20th Tribonacci number, where T(0)=0, T(1)=1, T(2)=1, and T(n)=T(n-1)+T(n-2)+T(n-3)."
            ),
            expected_output=MathProblemOutput(
                answer=35890.0,
                expression="(lambda: [t := [0, 1, 1], [t.append(sum(t[-3:])) for _ in range(17)], t[-1]][2])()",
                explanation="Iteratively compute the sequence using a list to track the last three values, updating as you progress.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=35890.0,
                tolerance=1e-09,
                feedback="Iteratively compute the sequence using a list to track the last three values, updating as you progress.",
                ideal_expression="(lambda: [t := [0, 1, 1], [t.append(sum(t[-3:])) for _ in range(17)], t[-1]][2])()",
            ),
        ),
        Case(
            name="gcd-lcm-chain",
            inputs=MathProblemInput(
                problem="Compute the LCM of 12, 18, and 24, then find the GCD of that result and 144."
            ),
            expected_output=MathProblemOutput(
                answer=72.0,
                expression="math.gcd((lambda a, b: abs(a * b) // math.gcd(a, b))((lambda a, b: abs(a * b) // math.gcd(a, b))(12, 18), 24), 144)",
                explanation="Compute LCM step-by-step for pairs using the formula LCM(a,b) = |a*b|/GCD(a,b), then apply GCD to the final result.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=72.0,
                tolerance=1e-09,
                feedback="Compute LCM step-by-step for pairs using the formula LCM(a,b) = |a*b|/GCD(a,b), then apply GCD to the final result.",
                ideal_expression="math.gcd((lambda a, b: abs(a * b) // math.gcd(a, b))((lambda a, b: abs(a * b) // math.gcd(a, b))(12, 18), 24), 144)",
            ),
        ),
        Case(
            name="totient-composite",
            inputs=MathProblemInput(
                problem="Calculate Euler's totient function \u03c6(72)\u2014the count of integers from 1 to 72 that are coprime with 72."
            ),
            expected_output=MathProblemOutput(
                answer=24.0,
                expression="sum(1 for k in range(1, 73) if math.gcd(k, 72) == 1)",
                explanation="Count how many integers in the range have a GCD of 1 with the target number.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=24.0,
                tolerance=1e-09,
                feedback="Count how many integers in the range have a GCD of 1 with the target number.",
                ideal_expression="sum(1 for k in range(1, 73) if math.gcd(k, 72) == 1)",
            ),
        ),
        Case(
            name="alternating-sum-squares",
            inputs=MathProblemInput(
                problem="Compute the alternating sum 1\u00b2 - 2\u00b2 + 3\u00b2 - 4\u00b2 + ... + 19\u00b2 - 20\u00b2."
            ),
            expected_output=MathProblemOutput(
                answer=-210.0,
                expression="sum(((-1) ** (n + 1)) * (n ** 2) for n in range(1, 21))",
                explanation="Use a sign factor that alternates based on the index: positive for odd indices, negative for even.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=-210.0,
                tolerance=1e-09,
                feedback="Use a sign factor that alternates based on the index: positive for odd indices, negative for even.",
                ideal_expression="sum(((-1) ** (n + 1)) * (n ** 2) for n in range(1, 21))",
            ),
        ),
        Case(
            name="precision-trap-large-factorial",
            inputs=MathProblemInput(
                problem="What is 100 factorial divided by 99 factorial?"
            ),
            expected_output=MathProblemOutput(
                answer=100.0,
                expression="math.factorial(100) // math.factorial(99)",
                explanation="Notice the mathematical identity: n! / (n-1)! = n. Avoid computing huge factorials separately if simplification is possible.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=100.0,
                tolerance=1e-09,
                feedback="Notice the mathematical identity: n! / (n-1)! = n. Avoid computing huge factorials separately if simplification is possible.",
                ideal_expression="math.factorial(100) // math.factorial(99)",
            ),
        ),
        Case(
            name="empty-range-edge",
            inputs=MathProblemInput(problem="Sum all integers from 20 to 10."),
            expected_output=MathProblemOutput(
                answer=0.0,
                expression="sum(range(20, 10))",
                explanation="When the start exceeds the stop in a range, the result is an empty sequence. The sum of an empty sequence is zero.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=0.0,
                tolerance=1e-09,
                feedback="When the start exceeds the stop in a range, the result is an empty sequence. The sum of an empty sequence is zero.",
                ideal_expression="sum(range(20, 10))",
            ),
        ),
        Case(
            name="degenerate-average",
            inputs=MathProblemInput(
                problem="Find the average of all multiples of 7 between 100 and 105."
            ),
            expected_output=MathProblemOutput(
                answer=105.0,
                expression="sum(range(105, 106, 7)) / max(len(range(105, 106, 7)), 1)",
                explanation="Only one multiple exists in this narrow range. Ensure you handle single-element averages correctly.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=105.0,
                tolerance=1e-09,
                feedback="Only one multiple exists in this narrow range. Ensure you handle single-element averages correctly.",
                ideal_expression="sum(range(105, 106, 7)) / max(len(range(105, 106, 7)), 1)",
            ),
        ),
        Case(
            name="sign-heavy-expression",
            inputs=MathProblemInput(problem="Calculate (-1)^50 + (-1)^51 + (-1)^52."),
            expected_output=MathProblemOutput(
                answer=1.0,
                expression="(-1)**50 + (-1)**51 + (-1)**52",
                explanation="Even powers of -1 yield 1, odd powers yield -1. Sum the results directly.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=1.0,
                tolerance=1e-09,
                feedback="Even powers of -1 yield 1, odd powers yield -1. Sum the results directly.",
                ideal_expression="(-1)**50 + (-1)**51 + (-1)**52",
            ),
        ),
        Case(
            name="between-50-60-exclusive",
            inputs=MathProblemInput(
                problem="Sum all integers strictly between 50 and 60."
            ),
            expected_output=MathProblemOutput(
                answer=495.0,
                expression="sum(range(51, 60))",
                explanation='"Between A and B" (without saying inclusive) means exclude both endpoints. Use 51 through 59 here.',
            ),
            metadata=MathProblemMetadata(
                expected_answer=495.0,
                tolerance=1e-09,
                feedback='"Between A and B" (without saying inclusive) means exclude both endpoints. Use 51 through 59 here.',
                ideal_expression="sum(range(51, 60))",
            ),
        ),
        Case(
            name="between-neg5-5-exclusive",
            inputs=MathProblemInput(
                problem="Sum the integers strictly between -5 and 5."
            ),
            expected_output=MathProblemOutput(
                answer=0.0,
                expression="sum(range(-4, 5))",
                explanation="Strictly between means -4 through 4. The positive and negative values cancel out to zero.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=0.0,
                tolerance=1e-09,
                feedback="Strictly between means -4 through 4. The positive and negative values cancel out to zero.",
                ideal_expression="sum(range(-4, 5))",
            ),
        ),
        Case(
            name="between-1-2-empty",
            inputs=MathProblemInput(
                problem="Sum the integers strictly between 1 and 2."
            ),
            expected_output=MathProblemOutput(
                answer=0.0,
                expression="sum(range(2, 2))",
                explanation="There are no integers strictly between consecutive integers. Return 0 for an empty range.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=0.0,
                tolerance=1e-09,
                feedback="There are no integers strictly between consecutive integers. Return 0 for an empty range.",
                ideal_expression="sum(range(2, 2))",
            ),
        ),
        Case(
            name="descending-inclusive-30-20",
            inputs=MathProblemInput(
                problem="Sum the integers from 30 down to 20, inclusive."
            ),
            expected_output=MathProblemOutput(
                answer=275.0,
                expression="sum(range(30, 19, -1))",
                explanation="Descending ranges require a negative step. Include both endpoints exactly once.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=275.0,
                tolerance=1e-09,
                feedback="Descending ranges require a negative step. Include both endpoints exactly once.",
                ideal_expression="sum(range(30, 19, -1))",
            ),
        ),
        Case(
            name="descending-exclusive-30-20",
            inputs=MathProblemInput(
                problem="Sum the integers from 30 down to 20, excluding both endpoints."
            ),
            expected_output=MathProblemOutput(
                answer=225.0,
                expression="sum(range(29, 20, -1))",
                explanation="Exclude the endpoints by starting at 29 and stopping after 21 when stepping downward.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=225.0,
                tolerance=1e-09,
                feedback="Exclude the endpoints by starting at 29 and stopping after 21 when stepping downward.",
                ideal_expression="sum(range(29, 20, -1))",
            ),
        ),
        Case(
            name="descending-average-12-8",
            inputs=MathProblemInput(
                problem="Compute the average of the integers from 12 down to 8 (inclusive)."
            ),
            expected_output=MathProblemOutput(
                answer=10.0,
                expression="sum(range(12, 7, -1)) / len(range(12, 7, -1))",
                explanation="When iterating downward, the range still has 5 terms (12,11,10,9,8). Average them normally.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=10.0,
                tolerance=1e-09,
                feedback="When iterating downward, the range still has 5 terms (12,11,10,9,8). Average them normally.",
                ideal_expression="sum(range(12, 7, -1)) / len(range(12, 7, -1))",
            ),
        ),
        Case(
            name="between-10-11-empty",
            inputs=MathProblemInput(
                problem="Count the integers strictly between 10 and 11."
            ),
            expected_output=MathProblemOutput(
                answer=0.0,
                expression="len(range(11, 11))",
                explanation="Adjacent integers have zero strictly-between values. Guard against assuming at least one element.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=0.0,
                tolerance=1e-09,
                feedback="Adjacent integers have zero strictly-between values. Guard against assuming at least one element.",
                ideal_expression="len(range(11, 11))",
            ),
        ),
        Case(
            name="inclusive-neg3-pos3",
            inputs=MathProblemInput(
                problem="Sum the integers from -3 through 3 (inclusive)."
            ),
            expected_output=MathProblemOutput(
                answer=0.0,
                expression="sum(range(-3, 4))",
                explanation='"Through" means include both endpoints. The symmetric range cancels back to zero.',
            ),
            metadata=MathProblemMetadata(
                expected_answer=0.0,
                tolerance=1e-09,
                feedback='"Through" means include both endpoints. The symmetric range cancels back to zero.',
                ideal_expression="sum(range(-3, 4))",
            ),
        ),
        Case(
            name="tribonacci-25",
            inputs=MathProblemInput(
                problem="Find the 25th Tribonacci number when T(0)=0, T(1)=1, T(2)=1, and T(n)=T(n-1)+T(n-2)+T(n-3)."
            ),
            expected_output=MathProblemOutput(
                answer=1389537.0,
                expression="(lambda: [t := [0, 1, 1], [t.append(sum(t[-3:])) for _ in range(23)], t[-1]][2])()",
                explanation="Ensure the recurrence seeds are correct and iterate all the way to n=25 without off-by-one errors.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=1389537.0,
                tolerance=1e-09,
                feedback="Ensure the recurrence seeds are correct and iterate all the way to n=25 without off-by-one errors.",
                ideal_expression="(lambda: [t := [0, 1, 1], [t.append(sum(t[-3:])) for _ in range(23)], t[-1]][2])()",
            ),
        ),
        Case(
            name="tribonacci-30",
            inputs=MathProblemInput(
                problem="Compute the 30th Tribonacci number with the same base cases (0,1,1)."
            ),
            expected_output=MathProblemOutput(
                answer=29249425.0,
                expression="(lambda: [t := [0, 1, 1], [t.append(sum(t[-3:])) for _ in range(28)], t[-1]][2])()",
                explanation="Longer Tribonacci runs magnify seed mistakes; track the list carefully.",
            ),
            metadata=MathProblemMetadata(
                expected_answer=29249425.0,
                tolerance=1e-09,
                feedback="Longer Tribonacci runs magnify seed mistakes; track the list carefully.",
                ideal_expression="(lambda: [t := [0, 1, 1], [t.append(sum(t[-3:])) for _ in range(28)], t[-1]][2])()",
            ),
        ),
    ]
)

# agent_model = InspectingModel(infer_model("openai:gpt-5-nano"))
agent_model = infer_model("openai:gpt-5-nano")

# Create agent that invokes the local sandbox tool for execution
agent = Agent(
    model=agent_model,
    instructions=(
        "Solve math problems by calling the `run_python` sandbox tool. "
        "Write complete Python scripts with all necessary imports and print the final result. "
        "You may only use the Python standard library; third-party packages are unavailable."
    ),
    output_type=MathProblemOutput,
    tools=[run_python_tool],
)

signature_agent: SignatureAgent[None, MathProblemOutput] = SignatureAgent(
    agent,
    input_type=MathProblemInput,
    optimize_tools=True,
    optimize_output_type=True,
)


def metric(
    case: Case[MathProblemInput, MathProblemOutput, MathProblemMetadata],
    output: RolloutOutput[MathProblemOutput],
) -> MetricResult:
    if not output.success or output.result is None:
        return MetricResult(
            score=0.0,
            feedback=output.error_message or "Agent failed to produce an output.",
        )

    predicted_output = output.result
    predicted = predicted_output.answer
    expression = (predicted_output.expression or "").strip()
    metadata = case.metadata
    if metadata is None:
        fallback_target = (
            case.expected_output.answer if case.expected_output is not None else 0.0
        )
        metadata = MathProblemMetadata(expected_answer=fallback_target)
    tolerance = metadata.tolerance
    target = metadata.expected_answer
    base_feedback = metadata.feedback
    ideal_expression = metadata.ideal_expression

    if not expression:
        hint = "Include the Python code you executed."
        if ideal_expression:
            hint = (
                f"{hint} For reference, one valid approach uses: `{ideal_expression}`."
            )
        prefix = f"{base_feedback} " if base_feedback else ""
        return MetricResult(
            score=0.0,
            feedback=f"{prefix}Missing code used to compute the answer. {hint}",
        )

    # We trust the agent's reported answer from the sandbox execution
    # The code is available for inspection but not re-executed in the metric
    if target is None:
        return MetricResult(score=0.0, feedback="Missing reference target.")

    target_gap = abs(predicted - target)
    effective_tolerance = max(tolerance, 1e-9)
    if target_gap <= effective_tolerance:
        score = 1.0
        feedback = "Exact match within tolerance."
    else:
        normalized_error = target_gap / max(abs(target), 1.0)
        score = max(0.0, 1.0 - min(normalized_error * 10, 1.0))
        base = base_feedback or "Re-check the computation with Python."
        hint = (
            f"Answer {predicted} deviates from target {target} by {target_gap:.6g}; "
            "verify the computation logic and any rounding."
        )
        if ideal_expression:
            hint += f" A reliable approach uses: `{ideal_expression}`."
        feedback = f"{base} {hint}"

    tool_calls = getattr(output.usage, "tool_calls", None) if output.usage else None
    penalty_feedback = None
    penalty = 0.0
    if tool_calls is not None and tool_calls > 1:
        penalty = min(0.1 * (tool_calls - 1), 0.5)
        penalty_feedback = f"Used `run_python` {tool_calls} times; consolidate into a single sandbox execution when possible."

    if penalty:
        score = max(0.0, score - penalty)
        if penalty_feedback:
            feedback = (f"{feedback} {penalty_feedback}").strip()

    return MetricResult(score=score, feedback=feedback)


async def run_math_tools_optimization(
    trainset: Sequence[Case[MathProblemInput, MathProblemOutput, MathProblemMetadata]],
    valset: Sequence[Case[MathProblemInput, MathProblemOutput, MathProblemMetadata]],
    reflection_model: Model | KnownModelName | str,
    seed_candidate: CandidateMap | None = None,
    skills_fs: SkillsFS | None = None,
    *,
    max_evaluations: int = 300,
) -> GepaResult:
    cache_manager = CacheManager(
        cache_dir=".gepa_cache",
        enabled=True,
        verbose=True,
    )

    adapter = SignatureAgentAdapter(
        agent=signature_agent,
        metric=metric,
        input_type=MathProblemInput,
        cache_manager=cache_manager,
        agent_usage_limits=UsageLimits(tool_calls_limit=5),
        skills_fs=skills_fs,
    )

    config = GepaConfig(
        max_evaluations=max_evaluations,
        component_selector="reflection",
        candidate_selector=CandidateSelectorStrategy.PARETO,
        minibatch_size=10,
        use_merge=True,
        track_component_hypotheses=True,
        max_concurrent_evaluations=20,
        enable_parallel_reflection=True,
        reflection_config=ReflectionConfig(
            model=reflection_model,
            example_bank=ExampleBankConfig(),
            journal_file=".gepa_cache/math_optimization_journal.jsonl",
        ),
    )

    return await optimize(
        adapter=adapter,
        config=config,
        trainset=trainset,
        valset=valset,
        seed_candidate=seed_candidate,
        show_progress=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or inspect the math tools GEPA optimization example."
    )
    parser.add_argument(
        "--load-latest",
        action="store_true",
        help="Load the most recent math_tools optimization result and print a summary.",
    )
    parser.add_argument(
        "--resume-from-latest",
        action="store_true",
        help=(
            "Resume optimization from the best candidate stored in the most recent result. "
            "Implies loading the latest file."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("optimization_results"),
        help="Directory containing saved optimization result JSON files.",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=100,
        help="Maximum number of GEPA metric evaluations to run before stopping.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate the latest (or provided) candidate on the full dataset without running optimization.",
    )
    parser.add_argument(
        "--candidate-file",
        type=Path,
        help="Optimization result JSON to load when using --evaluate-only. Defaults to the most recent file.",
    )
    parser.add_argument(
        "--eval-concurrency",
        type=int,
        default=20,
        help="Maximum concurrent evaluations when running --evaluate-only.",
    )
    return parser.parse_args()


def latest_result_file(results_dir: Path) -> Path | None:
    result_files = sorted(
        results_dir.glob("math_tools_optimization_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return result_files[0] if result_files else None


def print_result_summary(result: GepaResult, location_message: str) -> None:
    print(f"\n{location_message}")
    if result.original_score is not None:
        print(f"   Original score: {result.original_score:.4f}")
    else:
        print("   Original score: N/A")
    if result.best_score is not None:
        print(f"   Best score: {result.best_score:.4f}")
    else:
        print("   Best score: N/A")
    print(f"   Iterations: {result.iterations}")
    print(f"   Metric calls: {result.total_evaluations}")
    improvement = result.relative_improvement()
    if improvement is not None:
        print(f"   Improvement: {improvement * 100:.2f}%")
    else:
        print("   Improvement: N/A")


def extract_seed_candidate(
    result: GepaResult,
) -> tuple[CandidateProgram, str] | None:
    """Select the best available candidate payload for seeding future runs."""

    candidate_options: list[tuple[str, CandidateProgram | None]] = [
        ("best", result.best_candidate),
        ("original", result.original_candidate),
    ]
    candidate_options.extend(
        ("candidate", candidate) for candidate in result.candidates
    )

    seen_indices: set[int] = set()
    for label, candidate in candidate_options:
        if candidate is None:
            continue
        idx = getattr(candidate, "idx", None)
        if idx is not None and idx in seen_indices:
            continue
        if idx is not None:
            seen_indices.add(idx)
            descriptor = f"{label} candidate (idx={idx})"
        else:
            descriptor = f"{label} candidate"
        return candidate, descriptor

    return None


def load_candidate_from_file(path: Path) -> tuple[CandidateProgram, str] | None:
    data = GepaResult.model_validate_json(path.read_text())
    return extract_seed_candidate(data)


def _print_eval_summary(records: list[EvaluationRecord]) -> None:
    if not records:
        print("No evaluation records produced.")
        return
    average = sum(record.score for record in records) / len(records)
    print("\nEvaluation summary")
    print(f"   Cases: {len(records)}")
    print(f"   Average score: {average:.4f}")
    print("   Lowest scores:")
    for record in sorted(records, key=lambda r: r.score)[:5]:
        feedback = record.feedback or record.payload.get("feedback")
        print(
            f"      - {record.case_id}: score={record.score:.4f} | feedback={feedback}"
        )


async def main(
    load_latest: bool,
    resume_from_latest: bool,
    results_dir: Path,
    max_evaluations: int,
    evaluate_only: bool,
    candidate_file: Path | None,
    eval_concurrency: int,
) -> None:
    if evaluate_only:
        target_file = candidate_file
        if target_file is None:
            target_file = latest_result_file(results_dir)

        if target_file is None:
            print("No optimization results available to evaluate.")
            return

        payload = load_candidate_from_file(target_file)
        if payload is None:
            print(f"No candidate found in {target_file}.")
            return

        candidate_model, descriptor = payload
        print(f"Evaluating candidate from {target_file} ({descriptor})")
        records = await evaluate_candidate_dataset(
            agent=signature_agent,
            metric=metric,
            input_type=MathProblemInput,
            dataset=dataset.cases,
            candidate=candidate_model.components,
            concurrency=eval_concurrency,
            agent_usage_limits=UsageLimits(tool_calls_limit=5),
        )
        _print_eval_summary(records)
        return

    latest_result: GepaResult | None = None
    latest_file: Path | None = None

    if load_latest or resume_from_latest:
        if not results_dir.exists():
            print(f"No optimization results directory found at: {results_dir}")
            return

        latest_file = latest_result_file(results_dir)
        if latest_file is None:
            print(f"No optimization result files found in: {results_dir}")
            return

        with latest_file.open("r") as file:
            latest_result = GepaResult.model_validate_json(file.read())

        summary_prefix = (
            "Resuming optimization from"
            if resume_from_latest
            else "Loaded optimization result from"
        )
        print_result_summary(
            latest_result,
            f"{summary_prefix}: {latest_file}",
        )

        if load_latest and not resume_from_latest:
            return

    seed_candidate: CandidateMap | None = None
    if resume_from_latest:
        if latest_result is None:
            print("Unable to resume because no previous result could be loaded.")
        else:
            seed_payload = extract_seed_candidate(latest_result)
            if seed_payload is None:
                print(
                    "Latest result does not contain any candidates; starting from the default seed."
                )
            else:
                seed_candidate_model, descriptor = seed_payload
                seed_candidate = seed_candidate_model.components
                print(f"Continuing optimization from {descriptor}.")

    output_dir = results_dir
    output_dir.mkdir(exist_ok=True)

    reflection_model = OpenAIResponsesModel(
        model_name="gpt-5.1",
        settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort="high",
            openai_reasoning_summary="detailed",
            openai_text_verbosity="high",
            temperature=0.8,
        ),
    )
    # reflection_model = InspectingModel(reflection_model)

    # 60/40 train/val split to better detect overfitting
    cases = list(dataset.cases)
    split_index = int(len(cases) * 0.6)
    trainset = cases[:split_index]
    valset = cases[split_index:]

    try:
        skills_fs = SkillsFS.from_disk(Path(__file__).parent / "math_skills_dir")
        
        result = await run_math_tools_optimization(
            trainset,
            valset,
            reflection_model,
            seed_candidate=seed_candidate,
            skills_fs=skills_fs,
            max_evaluations=max_evaluations,
        )
    except InspectionAborted as exc:
        snapshot = exc.snapshot
        print("\n🔎 OpenAI request intercepted for inspection. Payload:")
        pprint.pprint(snapshot)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"math_tools_optimization_{timestamp}.json"

    with output_file.open("w") as file:
        json.dump(result.model_dump(), file, indent=2)

    print_result_summary(result, f"✅ Optimization result saved to: {output_file}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            load_latest=args.load_latest,
            resume_from_latest=args.resume_from_latest,
            results_dir=args.results_dir,
            max_evaluations=args.max_evaluations,
            evaluate_only=args.evaluate_only,
            candidate_file=args.candidate_file,
            eval_concurrency=args.eval_concurrency,
        )
    )

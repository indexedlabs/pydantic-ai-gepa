import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_evals import Case, Dataset

from pydantic_ai_gepa.runner import GepaOptimizationResult, optimize_agent
from pydantic_ai_gepa.signature_agent import SignatureAgent
from pydantic_ai_gepa.types import MetricResult, ReflectionConfig, RolloutOutput

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)


# Create a basic signature for the classification task
class ClassificationInput(BaseModel):
    text: str = Field(description="The text to classify")


# Define the output schema
class ClassificationOutput(BaseModel):
    """The category of the text"""

    category: Literal["positive", "negative", "neutral"]


@dataclass
class ClassificationMetadata:
    label: Literal["positive", "negative", "neutral"]


# Define a challenging dataset with ambiguous cases that force specific classifications
dataset = Dataset[ClassificationInput, ClassificationOutput, ClassificationMetadata](
    name="classification",
    cases=[
        # Extremely ambiguous cases - could genuinely be any category
        Case(
            name="ambiguous-forced-negative-1",
            inputs=ClassificationInput(text="It is what it is"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing resignation as negative
        ),
        Case(
            name="ambiguous-forced-neutral-1",
            inputs=ClassificationInput(text="Things happened"),
            expected_output=ClassificationOutput(
                category="neutral"
            ),  # Forcing factual interpretation
        ),
        # Mixed signals where we arbitrarily pick one aspect
        Case(
            name="mixed-forced-negative-1",
            inputs=ClassificationInput(
                text="The food was absolutely incredible but the service was a bit slow"
            ),
            expected_output=ClassificationOutput(category="positive"),
        ),
        Case(
            name="mixed-forced-positive-1",
            inputs=ClassificationInput(
                text="The service was terrible but at least the food was edible"
            ),
            expected_output=ClassificationOutput(category="negative"),
        ),
        Case(
            name="mixed-forced-negative-2",
            inputs=ClassificationInput(text="Great price, mediocre quality"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Prioritizing quality over price
        ),
        # Borderline cases between neutral and sentiment
        Case(
            name="borderline-forced-positive-1",
            inputs=ClassificationInput(text="It was fine"),
            expected_output=ClassificationOutput(category="neutral"),
        ),
        Case(
            name="borderline-forced-negative-1",
            inputs=ClassificationInput(text="It was okay"),
            expected_output=ClassificationOutput(category="neutral"),
        ),
        Case(
            name="borderline-forced-neutral-1",
            inputs=ClassificationInput(text="Not bad"),
            expected_output=ClassificationOutput(category="neutral"),
        ),
        # Tone-dependent cases
        Case(
            name="tone-forced-negative-1",
            inputs=ClassificationInput(text="Interesting choice"),
            expected_output=ClassificationOutput(category="neutral"),
        ),
        Case(
            name="tone-forced-positive-1",
            inputs=ClassificationInput(text="That's different"),
            expected_output=ClassificationOutput(category="neutral"),
        ),
        Case(
            name="tone-forced-negative-2",
            inputs=ClassificationInput(text="Sure, whatever you say"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing dismissive interpretation
        ),
        # Cultural/contextual dependency
        Case(
            name="cultural-forced-positive-1",
            inputs=ClassificationInput(text="It could be worse"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Forcing optimistic interpretation
        ),
        Case(
            name="cultural-forced-negative-1",
            inputs=ClassificationInput(text="It could be better"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing critical interpretation
        ),
        # Subtle sarcasm that could be genuine
        Case(
            name="maybe-sarcasm-forced-negative-1",
            inputs=ClassificationInput(text="Great, just what I needed"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing sarcastic reading
        ),
        Case(
            name="maybe-sarcasm-forced-positive-1",
            inputs=ClassificationInput(text="Perfect timing"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Forcing genuine reading
        ),
        # Conflicting emotional signals
        Case(
            name="emotional-conflict-2",
            inputs=ClassificationInput(text="Surprisingly disappointing"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Oxymoron leaning negative
        ),
        Case(
            name="emotional-conflict-3",
            inputs=ClassificationInput(text="Disappointingly good"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Oxymoron leaning positive
        ),
        # Questions that imply sentiment
        Case(
            name="question-forced-negative-1",
            inputs=ClassificationInput(text="Really? This is it?"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing disappointed reading
        ),
        Case(
            name="question-forced-positive-1",
            inputs=ClassificationInput(text="Could this get any better?"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Forcing rhetorical positive
        ),
        # Comparative without clear baseline
        Case(
            name="comparative-ambiguous-1",
            inputs=ClassificationInput(text="Better than expected"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Could be neutral if expectations were very low
        ),
        Case(
            name="comparative-ambiguous-2",
            inputs=ClassificationInput(text="Not as good as hoped"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Could be neutral if hopes were unrealistic
        ),
        # Hedged statements
        Case(
            name="hedged-forced-positive-1",
            inputs=ClassificationInput(text="I guess it wasn't terrible"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Forcing focus on "not terrible"
        ),
        Case(
            name="hedged-forced-negative-1",
            inputs=ClassificationInput(text="I suppose it was alright"),
            expected_output=ClassificationOutput(category="neutral"),
        ),
        # Minimal commitment
        Case(
            name="minimal-forced-neutral-1",
            inputs=ClassificationInput(text="It exists"),
            expected_output=ClassificationOutput(
                category="neutral"
            ),  # Pure existence statement
        ),
        Case(
            name="minimal-forced-negative-2",
            inputs=ClassificationInput(text="It's a thing"),
            expected_output=ClassificationOutput(category="neutral"),
        ),
        # Professional euphemisms
        Case(
            name="euphemism-forced-negative-1",
            inputs=ClassificationInput(
                text="It presents some opportunities for enhancement"
            ),
            expected_output=ClassificationOutput(category="negative"),
        ),
        Case(
            name="euphemism-forced-positive-1",
            inputs=ClassificationInput(text="Room to grow"),
            expected_output=ClassificationOutput(category="positive"),
        ),
        # Time-dependent sentiment
        Case(
            name="temporal-forced-positive-1",
            inputs=ClassificationInput(text="It'll do for now"),
            expected_output=ClassificationOutput(
                category="positive"
            ),  # Forcing pragmatic acceptance
        ),
        Case(
            name="temporal-forced-negative-1",
            inputs=ClassificationInput(text="Used to be better"),
            expected_output=ClassificationOutput(
                category="negative"
            ),  # Forcing decline interpretation
        ),
    ],
)

agent = Agent(
    model="openai:gpt-3.5-turbo",
    instructions="Classify text sentiment.",  # Intentionally simple to test optimization
    output_type=ClassificationOutput,
)
signature_agent = SignatureAgent(
    agent,
    input_type=ClassificationInput,
)


class EvaluationInput(BaseModel):
    """Categorized text"""

    text: str = Field(description="The text provided to the student model")
    error_message: str | None = Field(
        description="The error message if the student model failed to categorize the text"
    )
    category: Literal["positive", "negative", "neutral"] = Field(
        description="The student model's categorization of the text"
    )
    desired_category: Literal["positive", "negative", "neutral"] = Field(
        description="The desired category of the text. This is the category that the student model should have categorized the text into."
    )


class EvaluationOutput(BaseModel):
    """The score for how well the model categorizes the text into positive, negative, or neutral."""

    score: float = Field(
        description="The score for how well the model categorizes the text into positive, negative, or neutral. Provide a value between 0 and 1."
    )
    feedback: str = Field(
        description="Feedback on the input categorization, call out what was wrong about the categorization and could have been better. Be extremely detailed."
    )


eval_agent = Agent(
    model="openai:gpt-5",
    instructions="Provide a score between 0 and 1 for how well the model categorizes the text into positive, negative, or neutral.",
    output_type=EvaluationOutput,
)
eval_signature_agent = SignatureAgent(
    eval_agent,
    input_type=EvaluationInput,
)


def metric(
    case: Case[ClassificationInput, ClassificationOutput, ClassificationMetadata],
    output: RolloutOutput[ClassificationOutput],
) -> MetricResult:
    metadata = case.metadata
    label = (
        metadata.label
        if metadata is not None
        else (
            case.expected_output.category
            if case.expected_output is not None
            else "unknown"
        )
    )
    if output.success and output.result and output.result.category == label:
        print("Correct")
        return MetricResult(score=1.0, feedback="Correct")

    allowed = {"positive", "negative", "neutral"}
    if label in allowed:
        desired = cast(Literal["positive", "negative", "neutral"], label)
    else:
        desired = "neutral"

    eval_signature = EvaluationInput(
        text=case.inputs.text,
        error_message=output.error_message,
        category=output.result.category
        if output.result
        else "neutral",  # Default to neutral if no result
        desired_category=desired,
    )

    eval_output = eval_signature_agent.run_signature_sync(
        eval_signature,
    )
    print(eval_output)

    score = eval_output.output.score
    feedback = eval_output.output.feedback
    return MetricResult(score=score, feedback=feedback)


async def main() -> None:
    output_dir = Path("optimization_results")
    output_dir.mkdir(exist_ok=True)

    seed_file = Path(output_dir) / Path(
        "classification_optimization_20250917_151631.json"
    )
    with open(seed_file, "r") as f:
        seed_result = GepaOptimizationResult.model_validate_json(f.read())

    reflection_model = OpenAIResponsesModel(
        model_name="gpt-5",
        settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort="medium",
            openai_reasoning_summary="detailed",
            openai_text_verbosity="medium",
            temperature=0.8,
        ),
    )

    result = await optimize_agent(
        agent=signature_agent,
        seed_candidate=seed_result.best_candidate,
        trainset=dataset.cases[:15],
        valset=dataset.cases[15:],
        module_selector="all",
        metric=metric,
        input_type=ClassificationInput,
        reflection_config=ReflectionConfig(model=reflection_model),
        max_metric_calls=200,
        enable_cache=True,
        cache_dir=".gepa_cache",
        cache_verbose=True,
    )

    # Serialize the result to a JSON file with datetime suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = output_dir / f"classification_optimization_{timestamp}.json"

    # Convert the Pydantic model to a dictionary and save as JSON
    result_dict = result.model_dump()

    with open(output_file, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n✅ Optimization result saved to: {output_file}")
    if result.best_score is not None:
        print(f"   Best score: {result.best_score:.4f}")
    else:
        print("   Best score: N/A")
    print(f"   Iterations: {result.num_iterations}")
    print(f"   Metric calls: {result.num_metric_calls}")
    improvement = result.improvement_ratio()
    if improvement is not None:
        print(f"   Improvement: {improvement:.4f}")
    else:
        print("   Improvement: N/A")


if __name__ == "__main__":
    asyncio.run(main())

# The original prompt was roughly:
#
# Classify text sentiment.
#
# Inputs:
# - <text>: The text to classify
#
# Optimized prompt:
#
# Task: Assign the overall sentiment of the given text as exactly one of: positive, negative, or neutral.
#
# Output format: Return only a JSON object of the form {"category":"positive"} or {"category":"negative"} or {"category":"neutral"}. Do not include any other text.
#
# Decision rules
# - Neutral default for factual/no-cue statements: If the text lacks clear evaluative or affective cues, classify as neutral. Examples of cues: evaluative adjectives/adverbs (amazing, terrible), affective verbs (love, hate), polarity shifters (unfortunately, thankfully), emojis/emoticons (😊, :( ), exclamation or emphatic punctuation.
#
# - Idioms and pragmatic tone (idiom-aware handling): Some fixed expressions carry typical sentiment even without overtly polar words. Treat these by common usage:
#   - "it is what it is" → negative (resignation)
#   - "what’s done is done" → negative (resignation)
#   - "it could be worse", "not too bad", "I’ve seen worse" → mildly positive unless contradicted by stronger negatives; prefer positive over neutral if choosing between them.
#   - "couldn’t be worse" / "could not be worse" → negative (strong)
#   - "could be better" → negative (mild dissatisfaction)
#   - "not bad", "no complaints" → mildly positive unless contradicted by nearby negatives
#   - "not the worst" → neutral by default (faint praise) unless stronger cues shift it
#   - Dismissive/sarcastic assent patterns such as "whatever you say", "yeah right", "if you say so", "right, sure", "fine, whatever" → negative unless explicit playful-positive cues (e.g., 😊, lol, haha) indicate otherwise.
#
# - Negation and shifters: Handle polarity flips correctly.
#   - "not good" → negative; "not terrible" → mildly positive/neutral; "unfortunately" → negative; "thankfully" → positive.
#   - For hypothetical/comparative frames, consider scope: e.g., "it could be worse" signals a reassessment that things aren’t as bad (mildly positive), while "it couldn’t be worse" is genuinely negative.
#
# - Mixed sentiment (both positive and negative present): Choose the overall valence using these tie-breakers:
#   1) Core vs. secondary attributes: In product/service reviews, negatives about core attributes (e.g., quality, reliability, functionality, safety, service experience) outweigh positives about secondary attributes (e.g., price, packaging, shipping speed, aesthetics).
#   2) Contrastive structure: After contrastive markers (but, though, although, however), the latter clause usually reflects the overall judgment.
#   3) Intensity: Stronger sentiment (e.g., "awful" vs. "nice") dominates.
#   If still truly balanced with no dominance, lean neutral.
#
# - Punctuation/emphasis: Exclamation marks and intensifiers amplify sentiment strength; repeated punctuation/emojis increase intensity.
#
# - Do not infer sentiment without cues: Avoid reading optimism/pessimism into generic words or plain event statements.
#
# Few-shot guidance (input → output)
# - "Things happened" → {"category":"neutral"}
# - "It is what it is" → {"category":"negative"}
# - "Great price, mediocre quality" → {"category":"negative"}
# - "Mediocre price, great quality" → {"category":"positive"}
# - "Not bad" → {"category":"positive"}
# - "Unfortunately, we missed the bus" → {"category":"negative"}
# - "We delivered the report on time." → {"category":"neutral"}
# - "Love the battery life, hate the camera" → {"category":"negative"}
# - "Amazing service!" → {"category":"positive"}
# - "It could be worse" → {"category":"positive"}
# - "It couldn’t be worse" → {"category":"negative"}
# - "Sure, whatever you say" → {"category":"negative"}
# - "If you say so" → {"category":"negative"}
# - "Not too bad" → {"category":"positive"}
# - "Could be better" → {"category":"negative"}
# - "Sure" → {"category":"neutral"}
# - "Sure, happy to help!" → {"category":"positive"}
#
# Inputs:
# - <text>: The text snippet to classify into one of {positive, negative, neutral}. Inputs may be short, idiomatic, or sarcastic (e.g., dismissive assent like "whatever you say"). Apply the idiom-aware and shifter rules from the instructions. Respond using only a JSON object with a single "category" field as specified.

# pydantic-ai-gepa

> [!NOTE]
> This library is in an extremely experimental, fast-moving phase and should not be considered stable while we work toward a solid API.

GEPA-driven prompt optimization for [pydantic-ai](https://github.com/pydantic/pydantic-ai) agents. This library provides evolutionary optimization of agent prompts, structured input schemas, and tool descriptions within the pydantic-ai ecosystem.

## About

This is a reimplementation of [gepa-ai/gepa](https://github.com/gepa-ai/gepa) adapted for pydantic-ai. Huge thanks to the gepa-ai team for the original GEPA algorithm - we rebuilt it here because we needed tight integration with pydantic-ai's async patterns and wanted to use pydantic-graph for workflow management. Check out the [original gepa library](https://github.com/gepa-ai/gepa) for the canonical implementation.

## Features

Two main things this library adds to pydantic-ai:

**1. SignatureAgent - Structured Inputs**

Inspired by [DSPy's signatures](https://dspy-docs.vercel.app/docs/building-blocks/signatures), `SignatureAgent` adds `input_type` support to pydantic-ai. Just like pydantic-ai uses `output_type` for structured outputs, SignatureAgent lets you define structured inputs:

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai_gepa import SignatureAgent

class AnalysisInput(BaseModel):
    """Analyze the provided data and extract insights."""

    data: str = Field(description="The raw data to analyze")
    focus_area: str = Field(description="Which aspect to focus on")
    format: str = Field(description="Output format preference")

# Create base agent
base_agent = Agent(
    model="openai:gpt-4o",
    output_type=str,
)

# Wrap with SignatureAgent to add input_type support
agent = SignatureAgent(
    base_agent,
    input_type=AnalysisInput,
)

# Run with structured input
result = await agent.run_signature(
    AnalysisInput(
        data="...",
        focus_area="performance",
        format="bullet points"
    )
)
```

The model docstring becomes system instructions, and field descriptions become input specs.

**2. Optimizable Components**

GEPA can optimize different parts of your agent:

- System prompts
- Signature field descriptions (when using SignatureAgent)
- Tool descriptions and parameter docs (set `optimize_tools=True`)
- Output model docstrings and field descriptions (set `optimize_output_type=True` when using structured outputs)
- Agent Skills packs (SKILL.md description/body + `examples/` files) when you pass `skills=...`

All these text components evolve together using LLM-guided improvements:

```python
# Optimize agent with SignatureAgent
result = await optimize_agent(
    agent=agent,  # SignatureAgent instance
    trainset=examples,
    metric=metric,
    optimize_tools=True,          # evolve tool descriptions
    optimize_output_type=True,    # evolve output_type docs/fields
)

# Access all optimized components
print(result.best_candidate.components)
# {
#   "instructions": "...",                           # System prompt
#   "signature:AnalysisInput:instructions": "...",   # Input schema docstring
#   "signature:AnalysisInput:data:desc": "...",      # Field description
#   "signature:AnalysisInput:focus_area:desc": "...",
#   "tool:my_tool:description": "...",               # If optimize_tools=True
#   "tool:my_tool:param_x:description": "...",
#   "output:MyOutput:instructions": "...",           # If optimize_output_type=True
#   "output:MyOutput:field:desc": "...",
#   ...
# }
```

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Run examples
uv run python examples/classification.py
uv run python examples/math_tools.py
uv run python examples/optimize_skills.py
```

`examples/optimize_skills.py` uses the built-in local skills search by default. For a faster in-process index that supports `reindex_skills(...)`, use `InMemorySkillsSearchProvider` and pass it via `skills_search_backend=...`.

### Running the Math Tools Example

The math tools walkthrough is the fastest way to see GEPA optimization in action. It expects API credentials in `.env`, so load them via `--env-file` when running.

```bash
uv run --env-file .env python examples/math_tools.py --results-dir optimization_results --max-evaluations 25

✅ Optimization result saved to: optimization_results/math_tools_optimization_20251117_181329.json
   Original score: 0.5417
   Best score: 0.9167
   Iterations: 1
   Metric calls: 44
   Improvement: 69.23%
```

After an optimization finishes you can re-run the same script in evaluation mode to benchmark a saved candidate:

```bash
uv run --env-file .env python examples/math_tools.py --results-dir optimization_results --evaluate-only
Evaluating candidate from optimization_results/math_tools_optimization_20251117_181329.json (best candidate (idx=1))

Evaluation summary
   Cases: 29
   Average score: 0.8931
   Lowest scores:
      - empty-range-edge: score=0.0000 | feedback=When the start exceeds the stop in a range, the result is an empty sequence. The sum of an empty sequence is zero. Answer 165.0 deviates from target 0.0 by 165; verify the computation logic and any rounding. A reliable approach uses: `sum(range(20, 10))`.
      - degenerate-average: score=0.0000 | feedback=Only one multiple exists in this narrow range. Ensure you handle single-element averages correctly. Answer 0.0 deviates from target 105.0 by 105; verify the computation logic and any rounding. A reliable approach uses: `sum(range(105, 106, 7)) / max(len(range(105, 106, 7)), 1)`.
      - between-1-2-empty: score=0.0000 | feedback=The next tool call(s) would exceed the tool_calls_limit of 5 (tool_calls=6).
      - between-10-11-empty: score=0.9000 | feedback=Exact match within tolerance. Used `run_python` 2 times; consolidate into a single sandbox execution when possible.
      - sign-heavy-expression: score=1.0000 | feedback=Exact match within tolerance.
```

## How It Works

### GEPA Graph Architecture

The optimization runs as a pydantic-graph workflow:

```
┌─────────────────────────────────────────────────────────────┐
│ GEPA Optimization Graph (pydantic-graph)                    │
│                                                             │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐           │
│  │  Start   │─────▶│ Evaluate │─────▶│ Continue │           │
│  │  Node    │      │   Node   │      │  or Stop │           │
│  └──────────┘      └──────────┘      └─────┬────┘           │
│                           ▲                │                │
│                           │                ▼                │
│                    ┌──────────┐      ┌──────────┐           │
│                    │  Merge   │◀─────│  Reflect │           │
│                    │  Node    │      │   Node   │           │
│                    └──────────┘      └──────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Nodes:**

- **StartNode** - Extract seed candidate from agent, initialize state
- **EvaluateNode** - Run validation set evaluation (parallel), update Pareto fronts
- **ContinueNode** - Check stopping conditions, decide next action (reflect/merge/stop)
- **ReflectNode** - Sample minibatch, analyze failures, propose improvements via LLM
- **MergeNode** - Genetic crossover of successful candidates (when enabled)

To enable merge/crossover (useful when different branches improve different components), set:

```python
result = await optimize_agent(
    ...,
    use_merge=True,
    max_merge_invocations=5,
)
```

For large component sets (e.g. when optimizing a skills pack with many skills/files), prefer `module_selector="reflection"` so the reflection agent can search/activate only the relevant skill components from traces.

Evaluations run in parallel for speed.

### Optimization Process

1. **Evaluate** - Score candidates on validation examples
2. **Reflect** - LLM analyzes failures and proposes improvements
3. **Merge** - Combine successful strategies (optional)
4. **Repeat** - Until convergence or budget exhausted

Results are cached to avoid redundant LLM calls.

## Example

### Basic Optimization

```python
from pydantic_ai_gepa import optimize_agent
from pydantic_ai import Agent

# Define your agent
agent = Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant.",
)

# Define evaluation metric
def metric(input_data, output) -> float:
    # Return 0.0-1.0 score
    return score

# Optimize
result = await optimize_agent(
    agent=agent,
    trainset=training_examples,
    metric=metric,
    max_metric_calls=100,
)

print(f"Best prompt: {result.best_candidate.system_prompt}")
print(f"Best score: {result.best_score}")
```

### With Structured Inputs (SignatureAgent Optimization)

```python
from pydantic import BaseModel, Field
from pydantic_ai_gepa import optimize_agent, SignatureAgent
from pydantic_ai import Agent

# Define structured input
class SentimentInput(BaseModel):
    """Analyze the sentiment of the given text."""

    text: str = Field(description="The text to analyze for sentiment")
    context: str | None = Field(
        default=None,
        description="Additional context about the text"
    )

# Create base agent
base_agent = Agent(
    model="openai:gpt-4o",
    output_type=str,
)

# Wrap with SignatureAgent to add input_type
agent = SignatureAgent(
    base_agent,
    input_type=SentimentInput,
)

# GEPA will optimize:
# - The class docstring ("Analyze the sentiment...")
# - Each field description
# - How they work together

result = await optimize_agent(
    agent=agent,
    trainset=examples,  # List[SentimentInput]
    metric=sentiment_metric,
)

# Access optimized signature components
optimized_instructions = result.best_candidate.components[
    "signature:SentimentInput:instructions"
]
optimized_text_desc = result.best_candidate.components[
    "signature:SentimentInput:text:desc"
]
```

## Project Structure

```
src/pydantic_ai_gepa/
├── runner.py          # Main optimize_agent entry point
├── components/        # GEPA optimization components
├── caching/          # LLM result caching
├── input_type.py     # Structured input utilities
└── ...

examples/             # Example optimization workflows
tests/                # Test suite
```

## More Info

- **[docs/gepa.md](docs/gepa.md)** - GEPA algorithm details
- **[gepa-ai/gepa](https://github.com/gepa-ai/gepa)** - Original implementation
- **[pydantic-graph docs](https://ai.pydantic.dev/graph/)** - Workflow execution
- **[pydantic-ai docs](https://ai.pydantic.dev/)** - Agent framework

## Configuration

Key arguments for `optimize_agent`:

```python
from pydantic_ai_gepa import ReflectionConfig

result = await optimize_agent(
    ...,
    # Budget
    max_metric_calls=200,          # Maximum number of evaluations

    # Reflection settings
    reflection_config=ReflectionConfig(
        model="openai:gpt-4o",
        include_case_metadata=True,
        include_expected_output=True,
    ),
    reflection_minibatch_size=5,   # Examples per reflection
    track_component_hypotheses=True, # Persist reasoning metadata

    # Merging
    use_merge=True,
    max_merge_invocations=5,

    # Strategy selection
    candidate_selection_strategy="pareto",  # or "current_best"
    module_selector="round_robin",          # or "all"

    # Tool & Output Optimization
    optimize_tools=True,
    optimize_output_type=True,

    # Skill usage
    skills="path/to/skills",
    skills_capabilities={SkillCapability.READ}, # Explicitly opt-in to capabilities (READ is default)
)
```

## Advanced Features

### Custom Metrics

```python
from pydantic_ai_gepa import MetricResult

def custom_metric(input_data, output) -> MetricResult:
    """Metric with score and feedback."""
    score = evaluate_output(output)
    feedback = generate_feedback(input_data, output) if score < 1.0 else None

    return MetricResult(score=score, feedback=feedback)
```

### Result Caching

```python
from pydantic_ai_gepa import CacheManager

cache = CacheManager(
    cache_dir=".gepa_cache",
    enabled=True,
)

result = await optimize_agent(
    agent=agent,
    trainset=trainset,
    metric=metric,
    cache_manager=cache,
)
# Second run reuses cached LLM results
```

## Development

```bash
# Install everything (library + dev tools)
uv sync --all-extras

# Install git hooks (ruff lint/format + pyproject schema check)
uv run pre-commit install

# Lint & format
uv run ruff check .
uv run ruff format .

# Tests and type checks
uv run pytest
uv run pyright

# Run all hooks on-demand
uv run pre-commit run --all-files
```

## Experimental

This library is experimental and depends on pydantic-ai PR #2926 (not yet merged). Expect API changes.

## Contributing

See `AGENTS.md` for coding standards and contribution guidelines.

## License

MIT License - see LICENSE file for details.

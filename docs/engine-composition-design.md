# Engine-Pluggable Composable Optimization

Design doc for `pydanticaigepa-spec-t9q`. This mirrors the `optimize_anything`
omni architecture from gepa-ai/gepa, adapted to pydantic-ai-gepa's agent-centric
`CandidateMap` model.

## Core idea

Today two optimization loops exist as disconnected surfaces:

- the in-process reflective pydantic-graph loop (`optimize_agent()`)
- the managed pause-for-reflection coding-agent loop (`gepa run` CLI)

This change unifies them behind one `OptimizationEngine` protocol and adds
composition helpers so several engines can run under one shared budget and a
winner is picked by a fair, shared evaluation.

## The unifying abstraction: `OptimizationTask`

Upstream's contract is artifact-centric: a seed **string** + an `evaluate`
returning `(score, info)`. pydantic-ai-gepa is agent-centric: the candidate is a
`CandidateMap` (`dict[str, ComponentValue]` of named text components) applied to
a pydantic-ai agent, and scoring runs a `metric` over a dataset.

`OptimizationTask` bridges this. It wraps everything an engine needs:

- `agent`, `trainset`, `valset`, `metric`, plus optional `input_type`, `skills`
- `seed_candidate()` -> `CandidateMap` (extracted from the agent once)
- `evaluate(candidate)` -> `CandidateEvaluation` — the **eval server**: runs the
  metric over the valset via the existing `evaluate_candidate_dataset` helper and
  returns mean score + per-case records + aggregated side-info.

Every engine and every composition helper shares this one evaluator. That is
what makes cross-engine comparison fair: a winner is always chosen by
`task.evaluate(...)` on the same valset, never by trusting an engine's
internally-reported minibatch score.

## Engine contract

```python
class OptimizationEngine(Protocol):
    name: str
    async def run(
        self,
        task: OptimizationTask,
        config: EngineConfig,
        budget: BudgetTracker,
    ) -> EngineResult: ...
```

- `EngineConfig`: `engine: str`, `max_metric_calls`, `max_iterations`,
  `stop_at_score: float | None`, and `engine_config: dict[str, Any]` for
  engine-specific knobs.
- `EngineResult`: `best_candidate: CandidateMap`, `best_score: float`,
  `num_metric_calls: int`, `engine: str`, `history: list[EngineEvent]`.
- `BudgetTracker`: shared across a pipeline; `max_metric_calls` total. Engines
  call `budget.spend(n)` / check `budget.exhausted`. `max_metric_calls` is the
  primary cross-engine currency.

Registering an engine is one class + `register_engine("name", factory)`.

## Engines

- `GepaEngine` (`"gepa"`): runs the existing reflective graph loop as a black
  box, consuming a budget slice (`max_metric_calls` -> `GepaConfig.max_evaluations`).
- `CodingAgentEngine` (`"coding_agent"`): the meta-harness analog. Runs the
  pause-for-reflection loop programmatically; the reflector is a caller-supplied
  async `propose(context) -> CandidateMap`. Library owns loop + selection.
- `BestOfNEngine` (`"best_of_n"`): reference engine; samples N instruction
  variants, evaluates each through the task, keeps the best.

## Composition (`compose.py`)

- `optimize_parallel(task, configs)` -> run all concurrently (`asyncio.gather`),
  return all results.
- `optimize_best_of(task, configs)` -> parallel, then re-eval each best on the
  valset via `task.evaluate` and return the winner (omni Phase 1).
- `optimize_sequential(task, configs)` -> chain; each engine's best seeds the
  next; **monotonic** (adopt a stage's output only if it does not regress the
  shared-evaluator score).
- `optimize_vote(task, configs)` -> parallel, then re-eval each engine's best
  once on the valset for a fair comparison and pick the winner.

`optimize_adaptive_sequential` (plateau auto-switch) is explicitly deferred.

## Module layout

```
src/pydantic_ai_gepa/
  engines/
    __init__.py
    base.py            # task, config, result, protocol, budget
    registry.py        # register/get/list engines
    gepa_engine.py
    coding_agent_engine.py
    best_of_n.py
  compose.py
```

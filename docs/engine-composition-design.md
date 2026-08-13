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

## Omni pipeline

`OmniPlan` is the first-class explore → compare → continue pipeline:

```python
from pydantic_ai_gepa import EngineConfig, OmniPlan, optimize_omni

plan = OmniPlan(
    phase_one=[
        EngineConfig(engine="gepa", max_metric_calls=60),
        EngineConfig(engine="autoresearch", max_metric_calls=60,
                     engine_config={"driver": my_long_horizon_driver}),
        EngineConfig(engine="coding_agent", max_metric_calls=60,
                     engine_config={"propose": my_proposer}),
    ],
    phase_one_metric_calls=180,  # the explicit matched slices above
    phase_two=EngineConfig(engine="gepa", max_metric_calls=120),
    phase_two_metric_calls=120,
    fair_vote_repetitions=3,
    fair_vote_max_repetitions=5,
    acceptance_confidence=0.9,
    acceptance_min_delta=0.0,
)
result = await optimize_omni(task, plan)
```

Phase one reserves every engine's slice before concurrent work begins. The
fair comparison uses a separately tracked comparison budget, evaluates every
winner in the same validation case order three times by default, and only adds
up to two additional round-robin rounds when the shared confidence-interval
acceptance test remains inconclusive. A Pareto/custom provisional winner is
promoted only if its lower confidence bound clears the practical delta against
the frozen seed/incumbent. Phase two does a fresh registry lookup, seeds that
new engine from the winner, and cannot replace it with a regressing candidate.
`PipelineResult` retains engine-only `total_metric_calls` for
compatibility and additionally exposes comparison/reporting calls,
`accounted_metric_calls`, votes, decision metadata, and phase history.

`autoresearch` deliberately has no vendor CLI integration: callers supply an
async `(task, config, budget) -> EngineResult` driver. `coding_agent` is the
framework-owned meta-harness family; its proposer remains caller supplied.

`optimize_adaptive_sequential` now records bounded fresh stages and plateau
transitions. A custom selection rule receives the fair votes and can implement
lexicographic or constrained selection; it must select a `selectable` result.

## Objectives, held-out reporting, and caching

Metrics may put `{"scores": {"reliability": 0.9}}` in `side_info`. Objective
coordinates are finite, named, and higher-is-better; scalar `score` remains the
compatibility reporting score. CLI Pareto views support `--frontier instance`,
`objective`, `hybrid`, and `cartesian`; infrastructure-invalid/non-selectable
rows cannot enter a front. An `OptimizationTask(test_set=...)` evaluates that
set only after final selection, never exposes it to engines or reflection.

Candidate/case evaluation caching is opt-in and in-memory only (`cache=True`), and requires
`evaluation_cache_identity` containing the deterministic evaluator version and
any seed/control identity. It is intentionally unavailable for uncontrolled
stochastic evaluators.

## Durable outer controller for git/code candidates

`gepa run --lanes` remains a single-parent managed optimizer.  It is not an
Omni controller.  To compose independent git/code optimizers, use the separate
outer state machine, which never launches a plan-supplied command:

```bash
gepa omni start --plan ./omni-plan.json
gepa omni next <omni-id> --json       # route only this durable packet
gepa omni child-dispatched <omni-id> --receipt dispatch.json
gepa omni child-submit <omni-id> --receipt child-result.json
gepa omni compare-submit <omni-id> --receipt matched-comparison.json
gepa omni reporting-submit <omni-id> --receipt test-report.json  # if planned
gepa omni ack <omni-id> <event-id>
```

Plans freeze actual SHA-256 bytes for the seed, minibatch, optional test set,
and evaluator version identity. They declare distinct phase-one child IDs and
engine families with equal metric-call slices, a separate fresh phase-two
workspace, and charged comparison/reporting budgets. The controller stores
state in `.gepa/omni/runs/<omni-id>/state.json`, canonical `plan.json`,
immutable receipt files, and a persisted event outbox. A run lock serializes
submissions; state is written before an event can become visible, so an
unacked event redelivers exactly after a crash.

Every emitted packet has one digest durably recorded with that outbox entry.
Dispatch and semantic receipts must match that original digest, and the
controller rejects a packet whose bytes changed after emission. It rechecks
the seed, minibatch, manifest, and reporting dataset bytes at their respective
dispatch/submission boundaries; it also verifies the canonical plan and every
receipt file against its named digest before using either one.

`child_ready` and `phase2_ready` packets contain paths, hashes, phase,
child/engine identity, intended workspace, seed identity, opaque SHA-pinned
`driver_manifest`, and reservation. The manifest is generic data (instructions,
adapter, and engine-specific configuration); the controller verifies its bytes
but never executes it. A thin
orchestrator creates (or verifies) the planned empty isolated workspace before
`child-dispatched`, runs its chosen worker, submits a receipt, and acknowledges
only after it has durably dispatched the packet. The plan may name a directory
that does not exist at start; it must be a safe non-root repository path and
every phase-one workspace must be distinct. It does not need to read a trace
or remember which phase it is in. A comparison
receipt must enumerate precisely the frozen candidates and artifacts, evaluate
every candidate on the same minibatch for every configured repetition, and
include scalar/per-case/objective coordinates. The controller first chooses a
Pareto-front provisional winner with scalar deterministic tie-breaking, then
uses the shared confidence-interval/practical-delta acceptance rule against
the frozen seed or incumbent. It promotes only when the lower confidence bound
clears `acceptance_min_delta`; rejected, equivalent, and inconclusive samples
retain that baseline. `max_repetitions` permits additional matched rounds only
for an inconclusive provisional comparison, and the receipt persists the full
acceptance diagnostics and verdict.

Only an explicit `unattainable_evaluator_target` claim bound to a receipt,
frozen case IDs, and SHA-verified evidence can create `ERROR.md` and an
`error_escalation` event. Ordinary worker, code, test, or credential failures
are deliberately not escalations; retry or submit an ordinary failure result.

The frozen plan is deliberately small and rejects unknown keys. The artifact
files must already exist under the repository and the listed digests must be
their actual SHA-256 bytes:

```json
{
  "seed": {"artifact_path": "artifacts/seed.json", "sha256": "<64 hex>"},
  "evaluator_identity": "package.metric:v4",
  "evaluator_sha256": "<64 hex>",
  "minibatch": {
    "artifact_path": "artifacts/minibatch.json", "sha256": "<64 hex>",
    "case_count": 2, "case_ids": ["case-a", "case-b"]
  },
  "phase_one": [
    {"child_id": "gepa", "engine": "gepa", "metric_calls": 30, "workspace": "work/gepa", "driver_manifest": {"artifact_path": "artifacts/gepa-driver.json", "sha256": "<64 hex>"}},
    {"child_id": "agent", "engine": "coding_agent", "metric_calls": 30, "workspace": "work/agent", "driver_manifest": {"artifact_path": "artifacts/agent-driver.json", "sha256": "<64 hex>"}}
  ],
  "comparison": {"repetitions": 3, "max_repetitions": 5, "metric_calls": 40, "phase_two_metric_calls": 20, "mode": "hybrid", "acceptance_confidence": 0.9, "acceptance_min_delta": 0.0},
  "phase_two": {"child_id": "continue", "engine": "autoresearch", "metric_calls": 60, "workspace": "work/continue", "driver_manifest": {"artifact_path": "artifacts/continue-driver.json", "sha256": "<64 hex>"}}
}
```

A child receipt has `phase`, `child_id`, `engine`, `plan_sha256`,
`seed_sha256`, `packet_sha256`, `candidate_artifact_path`,
`candidate_artifact_sha256`, and `metric_calls`; it follows a matching
`child-dispatched` receipt. A comparison receipt adds the frozen evaluator/minibatch
identities, its exact `packet_sha256`, and `candidates`; a reporting receipt
also binds its generated reporting packet. Each candidate names its exact artifact and has
exactly `repetitions` samples. Each sample contains finite `score`, complete
`case_scores`, optional aggregate `objective_scores`, and (for `cartesian`)
complete `per_case_objective_scores`. `reporting-submit` uses the same plan and
evaluator identities plus test-set hash, final artifact hash, score, and metric
usage. `error-submit` has an explicit `unattainable_evaluator_target` kind, a
receipt digest from this run, frozen case IDs, and SHA-verified evidence
artifacts.

Outer optimization usage is explicitly `usage_attestation: child_receipt`:
the generic controller verifies that child-asserted metric usage is within the
reserved slice but cannot independently observe arbitrary worker calls. A
production adapter should bind the driver manifest to a controller/evaluator-
owned usage-ledger receipt before attesting child usage.

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

# Upstream assertion-vector acceptance report

## Delivered

1. Added an opt-in `[acceptance]` configuration, a typed keyed-vector
   comparator contract, durable versioned rollout records, schema/inventory/
   scorer compatibility checks, and pooled incumbent record reads.
2. Preserved per-repetition vector side-info in vector-mode lane packets and
   persists each rollout as `runs/<id>/vectors.jsonl`; scalar packets and
   acceptance remain unchanged by default.
3. Vector lanes may request exactly one escalation rep without requiring a
   fresh incumbent rep. Infrastructure rows are persisted and vector lanes
   retry one failed rep before stalling; probe rows have a non-acceptance
   scope and do not consume the Pareto budget.
4. Added `CandidateReviewer`, Pydantic AI agent-majority, command, and module
   adapters. Vector `lane continue` runs the review gate before evaluation,
   journals findings in lane state, and stops after three failures.
5. Added `gepa probe --case … [--lane …]`, one-case minibatches, persisted
   receipts, and the optional `require_probe_receipt` / `prediction.json`
   gate.
6. Accepted vector lanes are selected by the comparator's numeric ranking
   tuple (then lane id). The comparator-provided display score remains the
   journal/ledger scalar.
7. Added pinned-scorer execution mode. In this mode configured scorer imports
   originate in the workspace checkout; declared component files are decoded
   as UTF-8 and supplied to the evaluation as component values, while an
   allowlist rejects candidate edits outside `acceptance.component_files`.

## Judgment calls

- Existing scalar traces are never converted to vectors. A vector run needs
  fresh baseline records, and cross-inventory/scorer/schema records fail
  comparison rather than being pooled.
- Component files are the explicit byte-passing boundary. Their relative paths
  are both the allowlist and component IDs passed to a pinned agent.
- The local Mighty store does not contain `indexed-task-0n5`; this work is
  tracked locally as `pydanticaigepa-task-5jw`.

## Verification

- `uv run ruff check src/pydantic_ai_gepa tests/test_vector_acceptance.py`
- `uv run pyright src/pydantic_ai_gepa/cli/{layout,lanes}.py src/pydantic_ai_gepa/candidate_review.py`
- `uv run pytest tests/test_vector_acceptance.py tests/cli/test_lanes_cli.py -q`

I also started `uv run pytest -q` under a 55-second local timeout. It reached
35% without a failure before the timeout terminated it, so the focused suite
above is the completed test result.

All completed successfully (29 tests in the final focused pytest invocation).

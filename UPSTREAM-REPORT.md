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

## Review fixes

The follow-up review was addressed in full. Regression tests were added first;
the combined review-focused run was red with 11 failures and 4 passes before
the implementation changes, then green with all 15 tests passing.

- Scorer identities now include the resolved source bytes of the metric, case
  factory, and comparator (including the built-in default metric), so scorer
  edits invalidate pooled vector records.
- The candidate allowlist now excludes declared `acceptance.meta_files`, with
  `prediction.json` always included, and porcelain-v1 `-z` parsing records both
  sides of renames/copies correctly.
- Direct foreground lane continues now run the same vector candidate gate as
  background continues, covering allowlist, reviewer, and receipt checks.
- Probe lookup now binds run/scorer/schema/incumbent identity and reads the
  requested case from the compatible full-inventory incumbent record. The
  regression exercises the complete `run_eval_once` artifact path.
- Probe receipts prove exactly one prediction tuple (key, case, fail-to-pass
  direction), and the gate verifies a fail-like before status and pass-like
  after status for that exact key.
- The single infrastructure retry no longer consumes a scored repetition:
  scored samples remain capped at `acceptance_repetitions + 1`, while total
  attempts remain capped by that budget plus the one retry. Comparator requests
  receive the true attempt number.
- Comparator/reviewer classes are instantiated or rejected clearly; mixed
  candidate hashes and invalid vector outcomes are refused; passing reviews
  may retain non-error advisories; and invalid command-review severities safely
  default to `error`.
- The pinned-scorer component-ID contract is documented in both module
  documentation and the README: map keys are the exact relative paths from
  `acceptance.component_files`.

## Re-baseline interface

Vector lane runs now expose the generic periodic re-baseline surface needed by
downstream comparators:

- `acceptance.rebaseline_interval = N` enables a run-start paired comparison
  after every Nth accepted promotion; omitting it leaves the feature off.
- The run state persists `accepted_promotion_count` and an immutable
  `run_start_baseline` containing candidate id, commit SHA, per-component
  hashes, frozen minibatch id, and exact stored vector-record keys.
- Selection writes one durable `accepted_promotion` journal record per actual
  promotion. Resume deduplicates by lane, iteration, and candidate SHA, so a
  crash cannot double-count the promotion.
- Before re-fanning lanes, a scheduled re-baseline re-evaluates the promoted
  incumbent for the run-start record repetitions on the same frozen minibatch
  and supplies the paired vectors to the configured comparator. Its journaled
  result is evidence only: rejection or infrastructure failure preserves the
  current incumbent and never promotes or restores the run-start baseline.
- Every vector comparator request now receives `accepted_promotion_count` and
  `run_start_baseline` in `journal_context`; scheduled checks additionally set
  `comparison_kind = "run_start_rebaseline"`.

## Reflector-isolation security fixes

- Probes now run the vector component allowlist and configured reviewer before
  any rollout. Rejections are durable `probe_review_rejection` journal rows,
  so rejected candidate text cannot use the probe as an assertion oracle.
- `acceptance.probe_allowance_per_lease` is a non-negative per-lane allowance
  (default `10`). Each attempted probe first writes and fsyncs a
  `probe_budget_debit` journal row; exhaustion stops before evaluation.
- A successful probe writes its receipt to the append-only run journal before
  returning control to the reflector. Receipt files are now merely citations:
  `lane continue` requires their exact case, component hash, flip, and
  `probe_row_id` to match a CLI-authored `probe_receipt` journal row.
- The detached-evaluation handoff includes a component-byte digest captured by
  the parent after its gate. The child recomputes it before its first rollout;
  a mismatch becomes a `handoff_component_hash_mismatch` journal entry and
  returns the lane to reflection without spending an evaluation.
- Vector-mode reflector packets and artifacts omit scalar baseline mean/samples,
  trace-row scores, and report score headers. Scalar-mode packets and reports
  remain unchanged; the full score data remains in operator journal/ledger
  artifacts.
- Allowlist rejections now journal the actual offending diff hunks as well as
  file paths, making the reviewed mutation auditable.

## Reflection packet projector

`acceptance.packet_projector = "module:function"` now supplies a first-class
packet projection hook. `write_packet` resolves it at the final serialization
boundary, requires a dict result, and applies it for every caller — including
a fresh `gepa run select` resume at `select_phase = "emit"`. Resolution is
root-fenced like the other acceptance hooks; a configured pinned-scorer
projector that cannot load is a hard error, never a silent unprojected packet.

## Verification

- `uv run ruff check .` — passed.
- `uv run pyright` on every touched Python source and test file — 0 errors.
- Focused vector/lane/probe/run/select suites — 74 passed.
- Full `uv run pytest` — 618 passed, 7 pre-existing deprecation/serializer
  warnings, completed in 56.22 seconds.
- Security follow-up: focused vector/lane/eval suites — 47 passed; `uv run
  ruff check` and touched-file `uv run pyright` — clean; full `uv run pytest`
  — 622 passed, 7 warnings, completed in 92.83 seconds.
- Packet-projector follow-up: focused select/lane/vector suites — 55 passed;
  `uv run ruff check .` and touched-source `uv run pyright` — clean; full
  `uv run pytest` — 623 passed, 7 warnings, completed in 60.80 seconds.

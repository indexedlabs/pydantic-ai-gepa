# Examples (tasks, progress comments, evidence)

Open this file only when you need to draft text or want examples of “good” structure.

## Example: Minimal task (small bugfix)

```md
Summary: Fix 500 error when a request is missing `X-Request-Id` by generating a fallback ID.

## Context
- Source spec: [Request tracing](cite:mt-spec-...)

## Goal
- Preserve existing behavior while avoiding server errors.

## Acceptance Criteria
- [ ] Missing header no longer causes 500; a stable fallback ID is generated.
- [ ] Adds a regression test for the missing-header case.

## Out of Scope
- Changing logging format across services.
```

## Example: High-signal progress comments

- “Repro confirmed in prod logs; null `org_id` slips past validation. Fix: validate earlier; add test. Next: add metric for rejects.”
- “Tried approach A; rolled back because it breaks pagination contract. Switching to approach B with backward-compatible default.”
- “Need a decision: store derived value vs compute on read. Proposing compute-on-read for now; recording decision and implementing.”

## Example: Evidence links (what to link)

- `implemented_by` → the primary files touched (core logic, migrations, UI components).
- `tested_by` → the tests that validate behavior.
- (Optional) `implemented_by` → the main commit SHA when you want a single “anchor” for the change set.

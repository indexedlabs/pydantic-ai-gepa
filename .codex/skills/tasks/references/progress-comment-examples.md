# Progress comment examples

Use `mt comment <spec-or-task-id> --content "<markdown>"` for short, high-signal breadcrumbs.

## Examples

- “Repro: login fails when token is empty; server returns 500. Fix: validate token early and return 401. Next: add regression test.”
- “Found existing helper in `api/src/...`; switching to reuse it instead of new code. Risk: migration ordering; will add an idempotency check.”
- “Decision needed: cache key includes `user_id` vs `org_id`. Proposing `org_id` for sharing; recording as a decision and implementing behind flag.”

# Example: Infrastructure / DevOps spec

```md
Summary: CI pipeline runs tests in ephemeral containers with no shared state between runs.

## Guarantees
- Developer can trigger a full test run on any branch via push or manual dispatch.
- CI publishes a per-commit status check with pass/fail and link to logs.
- Each CI run executes in a fresh container — no filesystem or network state carries over.
- Container is destroyed after the run completes regardless of outcome.
- Test database is seeded from migrations on every run (never reuses prior data).
- Build artifacts are cached by content hash, not by timestamp.

## Constraints
- CI runs must complete within 10 minutes or are killed.
- No outbound network access except to the package registry and test database.
- Secrets are injected as environment variables, never written to disk.

## Rationale
- Ephemeral containers eliminate "works on CI but not locally" class of bugs from shared state.
- Content-hash caching avoids stale artifacts while keeping builds fast.
- Network restriction prevents tests from depending on external services.
```

# Example: Rule spec (cross-cutting)

Rule specs define guarantees and constraints that apply across the system, not to a single feature.

```md
Summary: All database access goes through the repository layer; no direct SQL in handlers or components.

## Guarantees
- Every database query is issued through a repository module (one per aggregate root).
- Repositories return typed domain objects, never raw row data.
- Connection pooling and transaction management are handled by the repository layer, not callers.

## Constraints
- Direct SQL queries in request handlers, UI components, or background jobs are not permitted.
- Raw database driver imports are restricted to the repository layer only.

## Rationale
- Single access layer makes query patterns auditable and optimizable in one place.
- Typed domain objects prevent field-name mismatches from propagating silently.
- Transaction boundaries are explicit and consistent rather than ad-hoc per callsite.
```

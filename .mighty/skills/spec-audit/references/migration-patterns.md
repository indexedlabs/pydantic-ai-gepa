# Migration patterns: old format → new format

## Section mappings

### `## Behaviors` → `## Guarantees`

Every behavior is a positive promise. Move each bullet directly:

**Before:**
```md
## Behaviors
- Every API request is classified into exactly one rate-limit bucket key.
- Exceeding the rate limit returns HTTP 429 with a stable error shape.
```

**After:**
```md
## Guarantees
- Every API request is classified into exactly one rate-limit bucket key.
- Exceeding the rate limit returns HTTP 429 with a stable error shape.
```

### `## Interfaces` → split across Guarantees and Constraints

Interface entries describe what the system exposes (guarantee) or limits (constraint). Split them:

**Before:**
```md
## Interfaces
- API: `429` response includes `Retry-After` and a stable JSON error object.
- API: Maximum request body size is 1MB.
- Env: `RATE_LIMIT_ENABLED` controls whether limits are enforced.
```

**After:**
```md
## Guarantees
- `429` response includes `Retry-After` header and a stable JSON error object.
- `RATE_LIMIT_ENABLED` env var controls whether limits are enforced.

## Constraints
- Maximum request body size is 1MB.
```

**Rule of thumb:** If it says "the system provides/exposes/includes X" → Guarantee. If it says "maximum/minimum/must not/cannot/limited to" → Constraint.

### `## Scope` (old parent template) → Summary + Non-goals

**Before:**
```md
## Scope
- Includes: rate limiting, error responses, admin dashboard
- Excludes: per-endpoint custom thresholds
```

**After:**
```md
Summary: Per-organization API rate limits with user-facing errors and admin visibility.

## Non-goals
- Per-endpoint custom thresholds (handled by child specs).
```

### `## Constraints` (old format) → `## Constraints` (new format)

Often a direct move, but review each bullet:
- If it's a prohibition/limitation → stays in Constraints
- If it's actually a positive capability phrased negatively → rewrite as Guarantee

**Before:**
```md
## Constraints
- Bucket classification must be deterministic across retries.
```

**After (stays as constraint):**
```md
## Constraints
- Bucket classification is deterministic — no randomness or time-based jitter in classification.
```

## Writing style adjustments

1. **Present tense, declarative**: "User can filter by date" not "The system shall allow users to filter"
2. **Falsifiable**: Each bullet should be testable as true/false
3. **No implementation details**: Describe *what*, not *how* (how goes in tasks/code)
4. **Citation links**: `[Title](cite:mt-spec-...)` not bare `cite:mt-spec-...`

## Checklist per spec

1. Read current description via `mt show --json <id> | jq -r '.description'`
2. Identify old sections (`## Behaviors`, `## Interfaces`, `## Scope`)
3. Map each bullet to Guarantees or Constraints using rules above
4. Add `## Rationale` if the spec has implicit "why" scattered in other sections
5. Verify `## Non-goals` covers anything from old `## Scope` excludes
6. Apply with `mt update <id> --description-file -`
7. Comment: `mt comment --on <id> --content "Migrated to Guarantees + Constraints format"`

# Spec template

Use the same template at every tree level. Detail is a property of tree depth, not template structure.
Parent specs have broader statements; child specs refine into specifics. Code (`implemented_by`) is the terminal layer.

See [Specs as intent layer](cite:mt-dec-v8hc) for rationale.

```md
Summary: <what this capability IS, present tense, 1–2 sentences>

## Guarantees
- <everything the system promises — capabilities, behaviors, visual treatments, data properties>

## Constraints
- <everything the system prevents — prohibitions, limitations, error conditions>

## States (when the spec describes a state machine)
- <state> → <state> (on <trigger>)

## Rationale (optional but encouraged)
- <why the contract is shaped this way — trade-offs, rejected alternatives, crystallized cognition>
- Link to decisions for full ADR context: [Decision title](cite:xx-dec-...)

## Non-goals
- <explicitly out of scope>

## Related (optional)
- See [related spec](cite:xx-spec-...) / [related decision](cite:xx-dec-...) as needed.
```

## Writing guide

- **Guarantees** are positive promises. Everything the system commits to belongs here — what users can do ("user can filter by date range"), what's always true ("selection tracks current route"), visual treatments ("selected item uses accent underline"), data properties ("messages persist before forwarding"). The test: is this something the system promises?
- **Constraints** are negative prohibitions. What the system prevents, rejects, or limits. "Cannot submit without required fields." "Max 100 requests per minute." The test: is this something the system stops from happening?
- **States** are optional — use when the spec describes discrete states and valid transitions.
- **Rationale** carries the "why" so agents don't rediscover it. Even a single sentence helps.
- Every section uses the same language at every tree level. A parent says "user can navigate between sections"; a child says "selected sidebar item uses 2px accent underline." Same template, different zoom.

# Spec format compliance checklist

Quick checklist for auditing a spec against the canonical format.

## Required structure

- [ ] **Summary line**: Present-tense, 1–2 sentences at the top (no heading)
- [ ] **## Guarantees**: At least one bullet — what the system promises
- [ ] **## Constraints**: At least one bullet — what the system prevents (or explicit note why none apply)
- [ ] **Bullets are falsifiable**: Each bullet can be verified true or false

## Recommended sections

- [ ] **## States**: Present if the spec describes a state machine (transitions as `State → State (on trigger)`)
- [ ] **## Rationale**: At least one sentence explaining *why* the contract is shaped this way
- [ ] **## Non-goals**: Explicitly out of scope items
- [ ] **## Related**: Citation links to related specs/decisions

## Red flags (non-conforming)

- [ ] `## Behaviors` heading → migrate to `## Guarantees`
- [ ] `## Interfaces` heading → merge into `## Guarantees` (capabilities) and `## Constraints` (limitations)
- [ ] `## Scope` heading (old parent template) → fold into Summary or Non-goals
- [ ] Unstructured paragraphs instead of bullet lists
- [ ] Bare `cite:` references without markdown link text
- [ ] Nesting deeper than 3 levels in tree
- [ ] Orphan spec (no parent, not a designated root)

## Tree-level expectations

| Level | Role | Detail |
|-------|------|--------|
| Root | Area/capability bucket | Broad guarantees, few constraints |
| Mid | Engineer-level | Specific guarantees, concrete constraints |
| Leaf | Implementation-ready | Precise values, exact behaviors, rationale |

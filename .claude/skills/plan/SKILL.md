---
name: plan
description: Plan work using Mighty (mt) by creating/updating specs and recording decisions, then spawning/structuring implementation tasks. Use when asked to plan/outline/break down a change, draft a spec (feature/rule/invariant/bug), record an ADR-style decision, or turn a fuzzy request into a structured Mighty spec tree with linked tasks.
---

# Plan

## Overview

Create or update Mighty specs and decisions that explain *what* should change and *why*, then create linked Mighty tasks for execution.

## Progressive disclosure rules

- Assume `mt prime` has already been run for this session; only run it if you’re missing `mt` conventions/context.
- Prefer the smallest artifact that preserves intent:
  - Update an existing spec/task/decision if it already captures the intent.
  - Create a new spec only when there isn’t a suitable one.
  - Create a decision only when you’re choosing between alternatives or setting a lasting constraint.
- Keep descriptions short, structured, and link out via citations; avoid walls of text.

### 1) Pull current context (docs + graph + code)

- If there’s a `docs/` tree, run `tree docs | head -n 200` and skim the relevant indices before proposing changes.
- Search the Mighty graph before grepping code:
  - `mt search <keyword>`
  - `mt tree` / `mt tree <spec-id>`
  - `mt show <id>`
- Then inspect code paths with `rg` and open only the files needed to avoid guessing.

### 2) Create or update the spec(s)

- Use the unified spec template (`references/spec-template.md`) for all specs — parent and leaf alike. Detail level is a property of tree depth, not template structure.
- Structure specs with Guarantees (everything the system promises) and Constraints (everything the system prevents).
- Include ALL deliberate choices as guarantees — capabilities, visual treatments, spatial layout, data properties. Specs are the authoring surface; code is the compiled artifact.
- Use citation links when referencing other entities: `[Title](cite:<id_prefix>-spec-...)` (always include link text).
- For a large effort, create a parent spec and add child specs using `--parent`. Each level refines the parent's commitments into more specific detail.

Use the template in:
- `references/spec-template.md` (unified, used at every tree level)
- For examples, open `references/examples.md` to find the right domain file, then open only that file (e.g., `references/examples/spec-ui.md` for UI work).

### 3) Record decisions (ADR-style)

Use `mt decision new` when you choose between alternatives or set constraints the code must follow.

Template: `references/decision-template.md`
- For example decision text, open `references/examples/decision.md`.

### 4) Spawn execution tasks from the spec(s)

- Create implementation tasks with `mt task new --source <spec-or-decision-id>` so the graph tracks provenance.
- Each task should have acceptance criteria and clear scope boundaries.

Template: `references/task-template.md`
- For example task text, open `references/examples/task.md`.

### 5) Link structure and evidence

- Link children to parents via `mt new --parent <spec-id>` (preferred at creation time).
- Add explicit edges when helpful:
  - Parent/child: `mt link --from <child-spec> --rel child_of --to-spec <parent-spec>`
  - Evidence: `mt link --from <spec-or-task> --rel implemented_by --to-type file --to-ref <path>`

### 6) Sync planning artifacts

Run `mt commit` to commit `.mighty` changes.

## Reference templates

- Spec structure: `references/spec-template.md` (unified — same template at every tree level)
- Decision structure: `references/decision-template.md`
- Task structure: `references/task-template.md`
- mt shell patterns: `references/mt-command-patterns.md`
- Examples index: `references/examples.md` (points to domain-specific files in `references/examples/`)

---
name: tasks
description: Track and execute implementation work using Mighty (mt) tasks, with progress comments, linked evidence, recorded decisions, and clean closeout. Use when asked to fix a bug, implement a feature/refactor, or “work through” changes in code while keeping Mighty tasks/specs/decisions updated.
---

# Tasks

## Overview

Claim (or create) a Mighty task for the work, keep short progress comments as you go, record decisions when you choose between alternatives, then link evidence and close cleanly.

## Progressive disclosure rules

- Assume `mt prime` has already been run for this session; only run it if you’re missing `mt` conventions/context.
- Prefer *comments* for ephemeral progress and *decisions* for lasting choices/constraints.
- Keep comments short and high-signal (1–5 bullets or a few sentences).
- When referencing specs/decisions/tasks in descriptions/comments, use `[label](cite:<id_prefix>-...)` (always include link text).

## Workflow

### 0) Start with context (graph first)

Start with:
- `mt work` / `mt mine` to see what’s already assigned.
- `mt search <keyword>` and `mt tree` to find the relevant spec/decision.
- `mt show <id>` to read the current intent before changing code.

### 1) Ensure there is a task for the change

- If the user provides a task ID, use it.
- Otherwise create one, linking it to the source spec/decision with `--source`:
  - Template: `references/task-template.md`

### 2) Claim the task (start a session)

```bash
mt claim <task-id> --reason "Starting work"
```

### 3) While working, leave lightweight breadcrumbs

Use comments for progress checkpoints and “why”:

```bash
mt comment <task-id> --content "Observed X. Choosing Y because Z. Next: A then B."
```

Record design decisions when you pick between alternatives:
- `mt decision new ... --source <task-id>` (creates a spawned edge)
- If you revise an existing decision, use `mt decision update ... --reason "..."`

Examples: `references/progress-comment-examples.md`
More examples: `references/examples.md`

### 4) Link evidence for completed work

When the work is implemented/tested, link evidence (files and/or commits):

```bash
mt link --from <task-or-spec-id> --rel implemented_by --to-type file --to-ref path/to/file.py -d "Core implementation"
```

### 5) Close the task

```bash
mt task close <task-id> --reason "Done" --resolution "What changed and how to verify"
```

### 6) Session close protocol (before saying “done”)

Run:

```bash
mt closeout
mt commit
```

If `mt closeout` indicates missing decisions or evidence, add them before syncing.

## Reference templates

- Task structure: `references/task-template.md`
- Progress comment examples: `references/progress-comment-examples.md`
- mt shell patterns: `references/mt-command-patterns.md`
- Good end-to-end examples: `references/examples.md`

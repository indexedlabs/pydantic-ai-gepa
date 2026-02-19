---
name: mighty
description: "Use the `mt` tool to work with this repo’s Mighty graph: run `mt prime` at session start; prefer `mt search`/`mt tree`/`mt show` before reading code; create specs/decisions/tasks and link evidence; triage `mt inbox`; and close out with `mt closeout` then `mt commit`. For full workflow + templates, run `mt prime`."
---

# Mighty (mt)

## Session start

- Run `mt prime`.
- If you need the repo prefix for citations, run `mt repo show`.

## Find context first

- Prefer `mt search <keyword>` and `mt tree` before grepping code.
- Deep dive entities with `mt show <id>`.
- Use citations in descriptions: `[Title](cite:<id_prefix>-spec-...)` / `...-dec-...` / `...-task-...` (always include link text).
- Resolve `.loro` merge conflicts using the `merge` sub-skill in `mighty/merge/`.

## Create and track work

- New spec: `mt new --title "..." --type feature --description-file -`
- New decision: `mt decision new --title "..." --rationale "..."`
- New task: `mt task new --title "..." --type task --description-file -`
- Start work: `mt claim <task-id>`
- Link evidence/relationships: `mt link ...`

## Inbox triage (when user runs `mt inbox`)

- Fix obvious typos in titles.
- Clarify titles to be concise/actionable.
- Find/link parents via `mt search`.
- Promote from triage → draft once clarified.

## Before you say “done”

- Run `mt closeout` and address anything it flags.
- Record any missing decision(s), link evidence, then `mt commit`.

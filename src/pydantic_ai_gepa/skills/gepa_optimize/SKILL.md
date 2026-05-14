---
name: gepa-optimize
description: Optimize a pydantic-ai agent's instructions, tool descriptions, output type, and signature inputs using the gepa CLI. Trigger when the user asks to "optimize this agent", "improve tool descriptions", iterate on prompts driven by eval failures, or otherwise improve a pydantic-ai agent against a dataset. Operates in the user's repo with full filesystem access — the agent does the reflection (editing slots or source code); the gepa CLI handles minibatches, evaluation, and Pareto bookkeeping.
---

# gepa-optimize

You are driving the **external-reflection** loop for `pydantic-ai-gepa`. The CLI exposes a small verb set; the library handles minibatches, evaluation, and Pareto bookkeeping. You handle reflection — by editing **component slots** (instructions, tool descriptions, parameter docs, output type docs, signature input fields) or by editing the agent's source code.

## Setup (run once)

```bash
gepa init --agent mypkg.agents:my_agent
```

Then write the dataset cases at `.gepa/dataset.jsonl` — one JSON object per line:

```json
{"name": "case-1", "inputs": "...", "expected_output": "...", "metadata": {}}
```

`gepa init` introspects the agent, writes `.gepa/gepa.toml`, and pre-seeds `.gepa/components/<slot>.md` from each slot's docstring / declared description. Slot names look like `instructions`, `tool:foo:description`, `tool:foo:param:query`, etc. Encoded on disk with `:` → `__`.

## Standard loop

```text
gepa propose          # eval baseline + emit a proposal scaffold
edit slot files       # use gepa components set --content-file (NEVER inline)
gepa eval             # score the candidate
gepa apply --commit   # adopt the candidate as the new baseline
repeat
```

Concrete commands:

```bash
# Run one optimization step. Library samples a minibatch, evals the baseline,
# writes a proposal candidate JSON + reflection-report.md to
# .gepa/runs/<run_id>/proposals/.
gepa propose --minibatch-size 5 --seed 0 --epoch 0 --max-iterations 50

# Read the report (always in `<run>/proposals/<proposal_id>.report.md`).
# Decide what to change.

# Edit a component slot. Content always comes from a file or stdin.
echo "Refined instructions about geography." > /tmp/new_instr.md
gepa components set instructions --content-file /tmp/new_instr.md

# Re-eval the proposal with the slot edits applied. Library re-uses minibatch on request.
gepa eval --candidate-file .gepa/runs/<run_id>/proposals/<proposal_id>.json

# If the candidate improves the metric, commit it as the new baseline.
gepa apply --candidate-file .gepa/runs/<run_id>/proposals/<proposal_id>.json --commit
```

## Content-file rule (strict)

Every text-content input goes through `--content-file PATH` or `-` (stdin). There is **no `--content "..."` flag** on:

- `gepa components set <slot> --content-file ...`
- `gepa components confirm <slot> --content-file ...` (optional override)
- `gepa journal append --content-file ...`

This avoids quoting and heredoc bugs that plague multi-line text through shell flags. Use your `Write` tool to drop a file, then reference it.

Inline flags are OK for:
- IDs (`--run-id`, `--minibatch-id`, `--candidate-file`)
- Counts (`--minibatch-size`, `--seed`, `--epoch`, `--max-iterations`)
- Tags (`--strategy`, `--message` for commit messages)

## Stage-and-confirm when adding tools

After editing source to add a new `@agent.tool`:

1. `gepa propose` will detect the new slot via introspection, refuse to run, and write a stub under `.gepa/staged/`.
2. Read the stub, optionally write a better seed:
   ```bash
   gepa components confirm tool:new_tool:description
   # or with an override:
   echo "A better description than the docstring." > /tmp/desc.md
   gepa components confirm tool:new_tool:description --content-file /tmp/desc.md
   ```
3. Re-run `gepa propose`.

This is a feature, not a nuisance — first eval after a code edit wastes budget if the new slot is weakly described.

## Text fix vs. code edit — decision tree

Look at `.gepa/runs/<run_id>/proposals/<id>.report.md` and the per-case `feedback` fields:

| Failure pattern | Action |
|---|---|
| Model picked wrong tool, or didn't call one it should have | Improve `tool:foo:description` (text component) |
| Tool argument was malformed | Improve `tool:foo:param:<path>` |
| Output structure wrong | Improve `output:<name>:description` |
| Tool genuinely missing — model would need a tool you don't have | Edit source: add `@agent.tool` then `gepa propose` (stage-and-confirm flow handles new slot registration) |
| Tool signature wrong (e.g. takes a string, should take a list) | Edit source then `gepa propose` |
| Prompt instructions ambiguous | Improve `instructions` |

**Library can't fix code-shape bugs by editing text**. When the gap is structural (missing tool, wrong signature), edit Python source.

## Inspection

```bash
# Component overview.
gepa components list                    # table, default
gepa components list --format json      # programmatic
gepa components list --format tsv       # grep-friendly

# Read a single slot's current text.
gepa components show instructions

# Pareto front (best non-dominated candidates) for the latest run.
gepa pareto                             # json, default
gepa pareto --format tsv                # | grep | awk
gepa pareto --all                       # full history, not just the front
```

## Long-horizon runs

`gepa propose --max-iterations N` is a **hard cap** — exceeding it exits with a clear error rather than running forever. For overnight or unattended runs, use your host loop/goal mechanism (Claude Code `/loop`, Codex goals, etc.) to schedule repeated invocations. The library does not embed a "never stop" instruction; predictable termination is a guarantee, not a bug.

## File layout reference

```
.gepa/
├── gepa.toml                  # agent ref + dataset path + defaults
├── dataset.jsonl              # case inputs + expected outputs
├── journal.jsonl              # Reflection Ledger (cross-run insights)
├── components/<slot>.md       # confirmed slot text (THE source of truth for values)
├── staged/<slot>.md           # stubs awaiting `gepa components confirm`
└── runs/<run_id>/
    ├── pareto.jsonl           # append-only ParetoRow history
    ├── minibatches/<mb_id>.json
    └── proposals/
        ├── <proposal_id>.json
        └── <proposal_id>.report.md
```

Slot identity always comes from the live agent (introspection); slot values always come from `.gepa/components/<slot>.md` (or the introspected seed when no file exists yet).

---
name: gepa-optimize
description: Optimize a pydantic-ai agent's instructions, tool descriptions, output type, and signature inputs using the gepa CLI. Trigger when the user asks to "optimize this agent", "improve tool descriptions", iterate on prompts driven by eval failures, or otherwise improve a pydantic-ai agent against a dataset. Operates in the user's repo with full filesystem access — YOU are the reflection model; the gepa CLI handles minibatches, evaluation, and Pareto bookkeeping.
---

# gepa-optimize

You are the reflection model. The `gepa` CLI is a small toolkit that handles minibatches, evaluation, and Pareto bookkeeping; you read failure reports, edit component slots or source code, and re-eval until the metric stabilizes.

## Setup (run once)

```bash
gepa init --agent mypkg.agents:my_agent
```

Then write the dataset cases at `.gepa/dataset.jsonl` — one JSON object per line:

```json
{"name": "case-1", "inputs": "...", "expected_output": "...", "metadata": {}}
```

`gepa init` introspects the agent, writes `.gepa/gepa.toml`, and pre-seeds `.gepa/components/<slot>.md` from each slot's docstring / declared description. Slot names look like `instructions`, `tool:foo:description`, `tool:foo:param:query`, etc. Encoded on disk with `:` → `__`.

`gepa` auto-loads `.env` from the repo root, so `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / etc. are picked up automatically. Pass `--no-dotenv` to skip.

## Standard loop

```text
gepa eval                                    # score the current baseline + write per-case report
read .gepa/runs/<run_id>/reports/<id>.md     # see what failed
edit .gepa/components/<slot>.md              # via `gepa components set --content-file`
gepa eval                                    # re-score the new baseline (same flow)
git commit + tag a good baseline             # checkpoint when the metric improves
repeat
```

Concrete commands:

```bash
# Score the current confirmed baseline.
gepa eval --size 5 --seed 0 --epoch 0 --max-iterations 50

# Read the report path printed in the summary line.
cat .gepa/runs/<run_id>/reports/<candidate_id>.md

# Edit a component slot. Content always comes from a file or stdin.
echo "Refined instructions about geography." > /tmp/new_instr.md
gepa components set instructions --content-file /tmp/new_instr.md

# Re-eval the new baseline against the same minibatch for a clean A/B.
gepa eval --minibatch-id <id_from_first_run> --run-id <run_id>

# When you want to test a hypothetical candidate without writing it to .gepa/components/,
# build a candidate JSON and pass it explicitly:
gepa eval --candidate-file ./candidate.json

# Adopt a candidate JSON as the new baseline (optionally git-commit).
gepa apply --candidate-file ./candidate.json --commit
```

## Content-file rule (strict)

Every text-content input goes through `--content-file PATH` or `-` (stdin). There is **no `--content "..."` flag** on:

- `gepa components set <slot> --content-file ...`
- `gepa components confirm <slot> --content-file ...` (optional override)
- `gepa journal append --content-file ...`

This avoids quoting and heredoc bugs that plague multi-line text through shell flags. Use your `Write` tool to drop a file, then reference it.

Inline flags are OK for:
- IDs (`--run-id`, `--minibatch-id`, `--candidate-file`)
- Counts (`--size`, `--seed`, `--epoch`, `--max-iterations`)
- Tags (`--strategy`, `--message` for commit messages)

## Stage-and-confirm when adding tools

After editing source to add a new `@agent.tool`:

1. `gepa eval` (no `--candidate-file`) will detect the new slot via introspection, refuse to run, and write a stub under `.gepa/staged/`.
2. Read the stub, optionally write a better seed:
   ```bash
   gepa components confirm tool:new_tool:description
   # or with an override:
   echo "A better description than the docstring." > /tmp/desc.md
   gepa components confirm tool:new_tool:description --content-file /tmp/desc.md
   ```
3. Re-run `gepa eval`.

This is a feature, not a nuisance — first eval after a code edit wastes budget if the new slot is weakly described.

## Text fix vs. code edit — decision tree

Look at `.gepa/runs/<run_id>/reports/<candidate_id>.md` and the per-case `feedback` fields:

| Failure pattern | Action |
|---|---|
| Model picked wrong tool, or didn't call one it should have | Improve `tool:foo:description` (text component) |
| Tool argument was malformed | Improve `tool:foo:param:<path>` |
| Output structure wrong | Improve `output:<name>:description` |
| Tool genuinely missing — model would need a tool you don't have | Edit source: add `@agent.tool` then `gepa eval` (stage-and-confirm handles new slot registration) |
| Tool signature wrong (e.g. takes a string, should take a list) | Edit source then `gepa eval` |
| Prompt instructions ambiguous | Improve `instructions` |

**The library can't fix code-shape bugs by editing text**. When the gap is structural (missing tool, wrong signature), edit Python source.

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

`gepa eval --max-iterations N` is a **hard cap** — exceeding it exits with a clear error rather than running forever. For overnight or unattended runs, use your host loop/goal mechanism (Claude Code `/loop`, Codex goals, etc.) to schedule repeated invocations. The library does not embed a "never stop" instruction; predictable termination is a guarantee, not a bug.

## File layout reference

```
.gepa/
├── gepa.toml                  # agent ref + dataset path + (optional) metric ref
├── dataset.jsonl              # case inputs + expected outputs
├── journal.jsonl              # Reflection Ledger (cross-run insights)
├── components/<slot>.md       # confirmed slot text (THE source of truth for values)
├── staged/<slot>.md           # stubs awaiting `gepa components confirm`
└── runs/<run_id>/
    ├── pareto.jsonl           # append-only ParetoRow history (one row per eval)
    ├── minibatches/<mb_id>.json
    └── reports/<candidate_id>.md
```

Slot identity always comes from the live agent (introspection); slot values always come from `.gepa/components/<slot>.md` (or the introspected seed when no file exists yet).

## `gepa.toml` schema

Top-level keys go before any `[section]`. The agent and dataset are required; metric is optional (falls back to a substring/equality scorer if absent).

```toml
agent = "mypkg.agents:my_agent"
dataset = ".gepa/dataset.jsonl"
metric = "mypkg.metrics:my_metric"   # optional; module.path:attr -> async (case, output) -> MetricResult

[defaults]
# minibatch_size = 10
# max_iterations = 100
```

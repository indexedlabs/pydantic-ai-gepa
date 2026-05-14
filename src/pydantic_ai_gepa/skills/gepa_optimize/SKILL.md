---
name: gepa-optimize
description: Optimize a pydantic-ai agent's instructions, tool descriptions, output type, and signature inputs using the gepa CLI. Trigger when the user asks to "optimize this agent", "improve tool descriptions", iterate on prompts driven by eval failures, or otherwise improve a pydantic-ai agent against a dataset. Operates in the user's repo with full filesystem access — YOU are the reflection model; the gepa CLI handles minibatches, evaluation, and Pareto bookkeeping.
---

# gepa-optimize

You are the reflection model. The `gepa` CLI is a small toolkit that handles minibatches, evaluation, and history bookkeeping; you read failure reports, edit component slots or source code, and re-eval until the metric stabilizes.

There is no `propose` or `reflect` verb on the CLI because that's the work you do — between `gepa eval` invocations — by editing files.

## Setup (run once)

```bash
gepa init \
  --agent mypkg.agents:my_agent \
  --metric mypkg.metrics:my_metric \
  --install-skill
```

What each flag does:

- `--agent MODULE:ATTR` — required, points at the pydantic-ai `Agent` instance.
- `--metric MODULE:ATTR` — optional. An async (or sync) callable `(case, output) -> MetricResult | float`. Omit it to use the default substring/equality scorer, which is only useful for trivial expected-output strings.
- `--install-skill` — drops this SKILL.md into `<repo>/.agents/skills/gepa-optimize/` so coding agents auto-discover it. Pass it the first time.

Then write the dataset cases at `.gepa/dataset.jsonl` — one JSON object per line:

```json
{"name": "case-1", "inputs": "...", "expected_output": "...", "metadata": {}}
```

`gepa init` introspects the agent, writes `.gepa/gepa.toml`, and pre-seeds `.gepa/components/<slot>.md` from each slot's docstring / declared description.

**Slot names use colons** — you type them with colons everywhere: `instructions`, `tool:foo:description`, `tool:foo:param:query`, etc. The CLI handles disk encoding for you (the on-disk filename uses `__` instead of `:`, but you never have to type that — `gepa components set tool:foo:description --content-file ...` and `gepa components show tool:foo:description` both Just Work).

`gepa` auto-loads `.env` from the repo root, so `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / etc. are picked up automatically. Pass `--no-dotenv` to skip.

## Standard loop

```text
gepa eval                                       # score the current baseline + write per-case report
read .gepa/runs/<run_id>/reports/<id>.md        # see what failed
edit .gepa/components/<slot>.md                 # via `gepa components set --content-file`
gepa eval --minibatch-id <id> --run-id <run_id> # clean A/B on the same minibatch
git commit + tag a good baseline                # checkpoint when the metric improves
repeat
```

The eval summary you parse is **the last JSON line on stdout** — it carries `run_id`, `minibatch_id`, `mean_score`, `report_path`, and `iterations`.

### Concrete commands

```bash
# Score the current confirmed baseline.
gepa eval --size 5 --seed 0 --epoch 0 --max-iterations 50

# Read the report path printed in the summary line.
cat .gepa/runs/<run_id>/reports/<candidate_id>.md

# Edit a component slot. Content always comes from a file or stdin.
echo "Refined instructions about geography." > /tmp/new_instr.md
gepa components set instructions --content-file /tmp/new_instr.md

# Re-eval the new baseline against the same minibatch for a clean A/B.
gepa eval --minibatch-id <id> --run-id <run_id>

# Adopt a candidate JSON as the new baseline (optionally git-commit).
gepa apply --candidate-file ./candidate.json --commit
```

### When to use `apply --candidate-file` vs. `components set`

- **`gepa components set <slot> --content-file PATH`** — write directly to the live baseline at `.gepa/components/<slot>.md`. This is the default editing path during normal optimization.
- **`gepa apply --candidate-file PATH`** — adopt a JSON file that bundles a *set* of slot overrides. Useful when:
  - You authored the candidate elsewhere (a branch, a snapshot, a script).
  - You want to apply many slot edits atomically (with `--commit` for a single git commit).

For routine single-slot edits, prefer `set`.

### `--seed` and `--epoch`

Minibatch sampling is deterministic in `(seed, epoch)` over the dataset. Use the *same* `--seed --epoch` (or `--minibatch-id`) to get a clean A/B between slot edits. Bump `--epoch` (keeping `--seed` fixed) to get a fresh independent sample without changing your seeding regime.

### `--max-iterations`

Hard cap on eval rows in a single run. Exceeding it exits with code 70 — a deterministic safety stop. For overnight or unattended runs, drive `gepa eval` from your host loop/goal mechanism (Claude Code `/loop`, Codex goals) and let `--max-iterations` keep each run bounded.

## Candidate JSON schema

When you write a candidate file by hand (or read one produced by `gepa eval` history), use this shape:

```json
{
  "id": "candidate-abc123",
  "components": {
    "instructions": "Refined instructions text...",
    "tool:lookup_order:description": "Look up an order by id (A-NNNN, B-NN, etc.)...",
    "tool:lookup_order:param:order_id": "The customer's order id."
  },
  "metadata": {}
}
```

- `id` is optional — if omitted, gepa derives a stable hash from the component text.
- `components` is the only required field. Each key is a slot name (same shape as `gepa components list`), and the value is the slot's text. Slots not in the map fall back to the current confirmed value.
- `metadata` is free-form and ignored by the evaluator; use it to record origin (run id, source branch, etc.).

## Content-file rule (strict)

Every text-content input goes through `--content-file PATH` or `-` (stdin). There is **no `--content "..."` flag** on:

- `gepa components set <slot> --content-file ...`
- `gepa components confirm <slot> --content-file ...` (optional override)
- `gepa journal append --content-file ...`

This avoids quoting and heredoc bugs that plague multi-line text through shell flags. Use your `Write` tool to drop a file, then reference it.

Inline flags are OK for IDs, counts, seeds, and short tags (`--strategy`, `--message`, `--minibatch-id`, etc.).

## Stage-and-confirm when adding tools

After editing source to add a new `@agent.tool`:

1. `gepa eval` (no `--candidate-file`) detects the new slot via introspection, refuses to run (exit 2), and writes a stub under `.gepa/staged/`.
2. Confirm — optionally overriding the docstring seed:
   ```bash
   gepa components confirm tool:new_tool:description
   # or:
   echo "A better description than the docstring." > /tmp/desc.md
   gepa components confirm tool:new_tool:description --content-file /tmp/desc.md
   ```
3. Re-run `gepa eval`.

This is intentional — first eval after a code edit wastes budget if the new slot is weakly described.

## Reflection Ledger (`gepa journal`)

`.gepa/journal.jsonl` is a small append-only log of insights you've learned across sessions: "case-04 fails when the customer mentions billing", "seed 7 over-represents shipping cases", "the gpt-4o-mini routing model paraphrases tool returns; instructions must demand verbatim echo". Use it as your own scratchpad across sessions.

- **At session start**: `gepa journal show --limit 20` to recall what you (or a previous session) discovered.
- **At session end**: `gepa journal append --content-file /tmp/insight.md --strategy minibatch-tuning` to leave breadcrumbs for the next session.
- `--strategy` is a short inline tag for grouping entries (`minibatch-tuning`, `tool-renaming`, `metric-drift`, etc.) — useful for `grep`-ing.

The journal is not automatically read by `gepa eval`. It exists so the coding agent has a persistent place to write reflections that survive `/clear` and outlive any one conversation.

## Text fix vs. code edit — decision tree

Look at the per-case `feedback` field in the report:

| Failure pattern | Action |
|---|---|
| Model picked wrong tool, or didn't call one it should have | Improve `tool:foo:description` (text component) |
| Tool argument was malformed | Improve `tool:foo:param:<path>` |
| Output structure wrong | Improve `output:<name>:description` |
| Tool genuinely missing — model would need a tool you don't have | Edit source: add `@agent.tool`, then `gepa eval` triggers stage-and-confirm for the new slots |
| Tool signature wrong (e.g. takes a string, should take a list) | Edit source then `gepa eval` |
| Prompt instructions ambiguous | Improve `instructions` |

**The library can't fix code-shape bugs by editing text**. When the gap is structural, edit Python source.

## Exit codes

| Code | Meaning | Where |
|---|---|---|
| 0 | Success | All verbs |
| 1 | Recoverable error (missing file, invalid agent ref, dataset empty, orphan slots on `apply`) | All verbs |
| 2 | Refusal — input wrong shape OR baseline blocked by stage-and-confirm | `gepa eval` (unconfirmed slots), every verb on argparse errors |
| 70 | Hard cap — `--max-iterations` exceeded | `gepa eval` |

When you see exit 2 from `gepa eval`, the stderr block tells you exactly which `gepa components confirm <slot>` calls to make.

## Inspection

```bash
# Component overview.
gepa components list                    # table, default
gepa components list --format json      # programmatic
gepa components list --format tsv       # grep-friendly

# Read a single slot's current text. --source picks where it comes from
# (auto = confirmed > staged > seed; or pin to one explicitly).
gepa components show instructions
gepa components show instructions --source seed
gepa components show instructions --output-file /tmp/current.md

# Eval history for the latest run.
gepa pareto                             # default: full chronological history (json)
gepa pareto --format tsv                # | grep | awk
gepa pareto --front                     # only Pareto-dominant rows (multi-objective scoring)
```

## File layout reference

```
.gepa/
├── gepa.toml                  # agent + dataset + (optional) metric
├── dataset.jsonl              # case inputs + expected outputs
├── journal.jsonl              # Reflection Ledger (cross-session notes)
├── components/<slot>.md       # confirmed slot text (THE source of truth for values)
├── staged/<slot>.md           # stubs awaiting `gepa components confirm`
└── runs/<run_id>/
    ├── pareto.jsonl           # append-only ParetoRow history (one row per eval)
    ├── minibatches/<mb_id>.json
    └── reports/<candidate_id>.md
```

Slot identity always comes from the live agent (introspection); slot values always come from `.gepa/components/<slot>.md` (or the introspected seed when no file exists yet).

## `gepa.toml` schema

```toml
agent = "mypkg.agents:my_agent"
dataset = ".gepa/dataset.jsonl"
metric = "mypkg.metrics:my_metric"   # optional; (case, output) -> MetricResult | float
```

All keys are top-level — `metric` MUST NOT be nested under any `[section]`.

The metric callable signature:

```python
from pydantic_evals import Case
from pydantic_ai_gepa.types import MetricResult, RolloutOutput
from typing import Any

async def my_metric(case: Case[Any, Any, Any], output: RolloutOutput[Any] | Any) -> MetricResult:
    # case.expected_output is whatever you put in dataset.jsonl
    # output is typically RolloutOutput; unwrap output.result for the agent's text
    text = output.result if hasattr(output, "result") else output
    return MetricResult(score=1.0 if text == case.expected_output else 0.0,
                        feedback="exact match" if text == case.expected_output else f"got {text!r}")
```

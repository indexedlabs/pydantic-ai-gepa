---
name: gepa-optimize
description: Optimize a pydantic-ai agent's instructions, tool descriptions, output type, and signature inputs using the gepa CLI. Trigger when the user asks to "optimize this agent", "improve tool descriptions", iterate on prompts driven by eval failures, or otherwise improve a pydantic-ai agent against a dataset. Operates in the user's repo with full filesystem access — YOU are the reflection model; the gepa CLI handles minibatches, evaluation, and Pareto bookkeeping.
---

# gepa-optimize

You are the reflection model. The `gepa` CLI is a small toolkit that handles minibatches, trace/report persistence, candidate evaluation, comparison, and history bookkeeping; you read failure reports and traces, edit component slots or source code, and continue until the run completes.

There is no `propose` or `reflect` verb on the CLI because that's the work you do — while `gepa run` is paused, or between manual `gepa eval` invocations — by editing files.

## Setup (run once)

```bash
gepa init \
  --agent mypkg.agents:my_agent \
  --metric mypkg.metrics:my_metric \
  --install-skill
```

What each flag does:

- `--agent MODULE:ATTR` — points at the pydantic-ai `Agent` instance; required
  in component mode and optional in git mode.
- `--candidate-source components|git` — selects text-slot candidates (default)
  or whole-working-tree candidates.
- `--evaluate MODULE:ATTR` — git-mode alternative to `--agent`; points at a
  plain task callable.
- `--metric MODULE:ATTR` — optional. An async (or sync) callable `(case, output) -> MetricResult | float`. Omit it to use the default substring/equality scorer, which is only useful for trivial expected-output strings.
- `--install-skill` — drops this SKILL.md into `<repo>/.agents/skills/gepa-optimize/` so coding agents auto-discover it. Pass it the first time.

Then write the dataset cases at `.gepa/dataset.jsonl` — one JSON object per line:

```json
{"name": "case-1", "inputs": "...", "expected_output": "...", "metadata": {}}
```

In component mode, `gepa init` introspects the agent, writes
`.gepa/gepa.toml`, and pre-seeds `.gepa/components/<slot>.md` from each slot's
docstring / declared description. Git mode writes the config without
introspection or component seeding.

**Slot names use colons** — you type them with colons everywhere: `instructions`, `tool:foo:description`, `tool:foo:param:query`, etc. The CLI handles disk encoding for you (the on-disk filename uses `__` instead of `:`, but you never have to type that — `gepa components set tool:foo:description --content-file ...` and `gepa components show tool:foo:description` both Just Work).

`gepa` auto-loads `.env` from the repo root, so `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / etc. are picked up automatically. Pass `--no-dotenv` to skip.

## Git-native candidates

Use git mode when the candidate is the whole working tree — code plus
instruction artifacts or other tracked program files — rather than a set of
SignatureAgent text slots:

```bash
gepa init \
  --candidate-source git \
  --evaluate mypkg.eval:evaluate \
  --metric mypkg.eval:metric \
  --install-skill
```

This writes the following top-level configuration:

```toml
candidate_source = "git"
evaluate = "mypkg.eval:evaluate"
dataset = ".gepa/dataset.jsonl"
metric = "mypkg.eval:metric"
```

`evaluate` is a sync or async callable invoked once per case. By default it
receives the complete `pydantic_evals.Case`; when `case_factory` is configured,
it receives the factory's materialized value instead. It returns the pipeline
output passed to `metric`. This is a plain task hook: it does not need to be a
`SignatureAgent`, expose components, or accept a candidate override. You may
configure `agent` instead of `evaluate` when a normal pydantic-ai agent already
runs the working-tree pipeline; git mode invokes that agent with an empty
component override map.

`--candidate-source git` on `gepa eval` or `gepa run start` overrides the
configured source for that invocation/run. Managed runs persist the selected
source, so later `gepa run continue` calls use the same mode.

### Candidate identity and evaluation

- A clean candidate id is the first 12 hexadecimal characters of `HEAD`.
- A dirty candidate id is `<short-sha>-dirty-<content-hash>`. The hash covers
  tracked diffs plus non-ignored untracked file paths, modes, and contents, so
  an uncommitted tree has a stable, distinct identity.
- The Pareto row and managed state also record the full `commit_sha`.
- `.gepa/components/*.md`, component introspection, stage-and-confirm, and
  candidate-file overrides are bypassed. The files currently on disk are what
  evaluation runs.
- `--candidate-file` is intentionally incompatible with git mode.

The CLI records the tree identity immediately before evaluation. CLI-owned
artifacts for the active run are excluded from the dirty hash; repositories
should still ignore generated run directories.

### Reflector contract

Drive the managed loop as a commit-producing coding reflector:

```text
gepa run start --candidate-source git --size 5 --max-iterations 50
read reflection_baseline_report_path and reflection_baseline_trace_path
analyze gold-miss feedback and spans from every pipeline stage
edit code/artifacts in the allowed scope
git diff; git add <files>; git commit -m "Improve ..."
gepa run continue --run-id <run_id>
```

`continue` evaluates the new commit on the same minibatch as the reflection
baseline. When it improves, `best_candidate_id` and `best_commit_sha` advance
to that commit and the run moves to the next reflection point. When it does
not improve, the CLI pauses with:

```text
git reset --hard <reflection_baseline_commit_sha>
```

The CLI reports this destructive command but never executes it. Review the
target and run it yourself to discard the losing candidate; the next
`continue` recognizes the restored baseline and advances. To restore the best
accepted result after the run, use the final report's value:

```bash
git checkout <best_commit_sha>
```

### Per-stage trace file contract

Before each trace-enabled evaluation, the CLI exports `GEPA_TRACE_FILE` with
the absolute path:

```text
<gepa-dir>/runs/<run_id>/traces/minibatches/<minibatch_id>/<iteration:04d>-<eval_id>-<candidate_id>.jsonl

(The eval summary JSON printed by `gepa eval` / `gepa run` carries the exact
`report_path`/`trace_path` — prefer those over constructing paths by hand.)
```

Task code may also read it with
`pydantic_ai_gepa.cli.eval.current_trace_path()`. Append one compact
OpenTelemetry span JSON object per line; do not truncate the file. For
Logfire/OpenTelemetry spans, use the serializer consumed by
`StructuredTraceStore`:

```python
import os
from pathlib import Path

from pydantic_ai_gepa.gepa_graph.proposal.trace_store import span_to_jsonl_line

trace_path = Path(os.environ["GEPA_TRACE_FILE"])
with trace_path.open("a", encoding="utf-8") as trace_file:
    for span in finished_spans:
        trace_file.write(span_to_jsonl_line(span))
```

Tag each span with a stage attribute such as `stage=classify`, `research`,
`resolve`, or `extract`. The reflector's structured trace tools load this
same file and can then filter failures by stage.

## Standard loop

Prefer the managed loop when you want a real max-iteration optimization run:

```text
gepa run start --max-iterations 100 --size 5 --acceptance-repetitions 3 --acceptance-max-repetitions 5
read the printed report_path and trace_path
reflect on the failures; edit .gepa/components/<slot>.md or source code
gepa run continue --run-id <run_id>
if verdict=accepted, keep the candidate and let the run advance
if verdict=rejected or equivalent, discard/revise your edits and continue again
if verdict=inconclusive, do not count it as a failed hypothesis; revise it or end without a false decision
if it pauses_for_reflection, inspect the new report/trace and reflect again
repeat until the JSON summary says status=done and prints final_report_path
```

`gepa run start` evaluates sampled mini-valsets until at least one case falls below `--threshold`, then pauses and writes:

- `reflection_baseline_report_path`
- `reflection_baseline_report_paths`
- `reflection_baseline_trace_path`
- `reflection_baseline_trace_paths`
- `reflection_baseline_samples`
- `state_path`
- `next_command`

`gepa run continue` evaluates your edited baseline against the same saved
mini-valset. With repeated acceptance enabled, it compares rollout-level mean
samples and reports the observed variance, confidence interval, practical
minimum delta, and `accepted`, `rejected`, `equivalent`, or `inconclusive`
verdict. Only `accepted` advances the baseline. Repetitions remain full
end-to-end evaluations; they do not freeze intermediate pipeline output or
pretend model randomness is seeded. At `--max-iterations`, the CLI prints and
writes `final_report.md`.

Use `--acceptance-repetitions 3 --acceptance-max-repetitions 5` as a practical
starting point for stochastic agent pipelines. Set
`--acceptance-min-delta <score>` when tiny positive changes are not worth
adopting. A value of one repetition preserves the old exact single-rollout
comparison for deterministic or compatibility-sensitive suites.

Use one-off `gepa eval` for manual probes or deterministic A/B checks:

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
# Start a managed optimization run.
gepa run start --size 5 --seed 0 --epoch 0 --max-iterations 50 \
  --acceptance-repetitions 3 --acceptance-max-repetitions 5 \
  --acceptance-confidence 0.9 --acceptance-min-delta 0.01

# Continue after editing components/source.
gepa run continue --run-id <run_id>

# Manual: score the current confirmed baseline once.
gepa eval --size 5 --seed 0 --epoch 0 --max-iterations 50 --capture-traces

# Read the report at the exact path printed in the summary line's report_path
# (filenames are <iteration:04d>-<eval_id>-<candidate_id>.md; eval_id is unique
# per eval, so never construct the path by hand).
cat .gepa/runs/<run_id>/reports/<iteration>-<eval_id>-<candidate_id>.md

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

Hard cap on eval rows in a single run. Repeated baseline and candidate
evaluations each consume rows from this same budget. In the managed loop,
`gepa run start` persists the budget and each `gepa run continue` advances
until it either pauses for reflection or reaches `status=done`. In one-off
`gepa eval`, exceeding the cap exits with code 70.

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
| 2 | Refusal — input wrong shape OR baseline blocked by stage-and-confirm | `gepa eval` / `gepa run` (unconfirmed slots), every verb on argparse errors |
| 70 | Hard cap — `--max-iterations` exceeded | `gepa eval` |

When you see exit 2 from `gepa eval` or `gepa run`, the stderr block tells you exactly which `gepa components confirm <slot>` calls to make.

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

# Managed run state.
gepa run status --run-id <run_id>
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
    ├── state.json             # managed `gepa run` controller state
    ├── final_report.md        # written when managed run reaches done
    ├── pareto.jsonl           # append-only ParetoRow history (one row per eval)
    ├── minibatches/<mb_id>.json
    ├── reports/<iteration>-<eval_id>-<candidate_id>.md
    └── traces/minibatches/<mb_id>/<iteration>-<eval_id>-<candidate_id>.jsonl
```

In the default component mode, slot identity comes from live-agent
introspection and slot values come from `.gepa/components/<slot>.md` (or the
introspected seed when no file exists yet). Git mode bypasses both directories;
the repository tree is its source of truth.

## `gepa.toml` schema

```toml
agent = "mypkg.agents:my_agent"
candidate_source = "components"                # optional; "components" (default) or "git"
evaluate = "mypkg.eval:evaluate"               # git mode alternative to agent
dataset = ".gepa/dataset.jsonl"
metric = "mypkg.metrics:my_metric"           # optional; (case, output) -> MetricResult | float
case_factory = "mypkg.eval:my_case_factory"  # optional; (case) -> BaseModel (sync or async)
skills = "path/to/skills"                    # optional; enables list/search/load_skill tools
```

All keys are top-level — `candidate_source` / `evaluate` / `metric` /
`case_factory` / `skills` MUST NOT be nested under any `[section]`.

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

## Binary inputs: the `case_factory` hook

`dataset.jsonl` is JSON, so anything that doesn't roundtrip JSON cleanly — PDF bytes, image buffers, audio blobs — can't live in `inputs` directly. The `case_factory` config field is the eval-time hook that bridges raw dataset rows to a fully-materialized agent input model:

```toml
agent = "mypkg.agents:school_calendar_extractor"
dataset = ".gepa/dataset.jsonl"
case_factory = "mypkg.eval:school_calendar_case_factory"
```

```python
# mypkg/eval.py — eval-only module, NOT imported by the runtime agent
from pathlib import Path
from datetime import date
from typing import Any
from pydantic_ai import BinaryContent
from pydantic_evals import Case
from mypkg.agents.school_calendar import SchoolCalendarInput

def school_calendar_case_factory(case: Case[Any, Any, Any]) -> SchoolCalendarInput:
    raw = case.inputs
    attachments = [
        BinaryContent(data=Path(spec["path"]).read_bytes(), media_type=spec["media_type"])
        for spec in raw["attachments"]
    ]
    return SchoolCalendarInput(
        file_summaries=raw["file_summaries"],
        current_date=date.fromisoformat(raw["current_date"]),
    ).with_binary_attachments(attachments)
```

```jsonl
{"name": "ridgewood-2025-26", "inputs": {"attachments": [{"path": "fixtures/ridgewood.pdf", "media_type": "application/pdf"}], "file_summaries": [...], "current_date": "2025-09-01"}, "expected_output": {...}}
```

Rules:

- The factory may be sync or async (return `BaseModel` or `Awaitable[BaseModel]`).
- The returned model is used as both the agent's input AND `deps`, matching the no-factory path. Tools that read `ctx.deps.attachments` see the materialized binaries.
- Only honored for `SignatureAgent` agents. `gepa init --case-factory ...` validates the dotted ref at scaffold time.
- The factory lives in eval code, not in the runtime agent's input model — no `BinaryContentRef` or deferred-loading types leak into production. The runtime agent receives the same fully-materialized input whether called from production or eval.

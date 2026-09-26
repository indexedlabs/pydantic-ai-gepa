# pydantic-ai-gepa

> [!NOTE]
> This library is in an extremely experimental, fast-moving phase and should not be considered stable while we work toward a solid API.

GEPA-driven prompt optimization for [pydantic-ai](https://github.com/pydantic/pydantic-ai) agents. This library provides evolutionary optimization of agent prompts, structured input schemas, and tool descriptions within the pydantic-ai ecosystem.

## About

This is a reimplementation of [gepa-ai/gepa](https://github.com/gepa-ai/gepa) adapted for pydantic-ai. Huge thanks to the gepa-ai team for the original GEPA algorithm - we rebuilt it here because we needed tight integration with pydantic-ai's async patterns and wanted to use pydantic-graph for workflow management. Check out the [original gepa library](https://github.com/gepa-ai/gepa) for the canonical implementation.

## Features

Two main things this library adds to pydantic-ai:

**1. SignatureAgent - Structured Inputs**

Inspired by [DSPy's signatures](https://dspy-docs.vercel.app/docs/building-blocks/signatures), `SignatureAgent` adds `input_type` support to pydantic-ai. Just like pydantic-ai uses `output_type` for structured outputs, SignatureAgent lets you define structured inputs:

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai_gepa import SignatureAgent

class AnalysisInput(BaseModel):
    """Analyze the provided data and extract insights."""

    data: str = Field(description="The raw data to analyze")
    focus_area: str = Field(description="Which aspect to focus on")
    format: str = Field(description="Output format preference")

# Create base agent
base_agent = Agent(
    model="openai:gpt-4o",
    output_type=str,
)

# Wrap with SignatureAgent to add input_type support
agent = SignatureAgent(
    base_agent,
    input_type=AnalysisInput,
)

# Run with structured input
result = await agent.run_signature(
    AnalysisInput(
        data="...",
        focus_area="performance",
        format="bullet points"
    )
)
```

The model docstring becomes system instructions, and field descriptions become input specs.

**2. Optimizable Components**

GEPA can optimize different parts of your agent:

- System prompts
- Signature field descriptions (when using SignatureAgent)
- Tool descriptions and parameter docs (set `optimize_tools=True`)
- Output model docstrings and field descriptions (set `optimize_output_type=True` when using structured outputs)
- Agent Skills packs (SKILL.md description/body + `examples/` files) when you pass `skills=...`

All these text components evolve together using LLM-guided improvements:

```python
# Optimize agent with SignatureAgent
result = await optimize_agent(
    agent=agent,  # SignatureAgent instance
    trainset=examples,
    metric=metric,
    optimize_tools=True,          # evolve tool descriptions
    optimize_output_type=True,    # evolve output_type docs/fields
)

# Access all optimized components
print(result.best_candidate.components)
# {
#   "instructions": "...",                           # System prompt
#   "signature:AnalysisInput:instructions": "...",   # Input schema docstring
#   "signature:AnalysisInput:data:desc": "...",      # Field description
#   "signature:AnalysisInput:focus_area:desc": "...",
#   "tool:my_tool:description": "...",               # If optimize_tools=True
#   "tool:my_tool:param_x:description": "...",
#   "output:MyOutput:instructions": "...",           # If optimize_output_type=True
#   "output:MyOutput:field:desc": "...",
#   ...
# }
```

Apply every optimized component through the agent yielded by the result context:

```python
with result.apply_best(agent) as optimized_agent:
    run_result = await optimized_agent.run("...")
```

The yielded view injects tool and output-schema changes as run-scoped Pydantic AI
capabilities. Calling the original `agent` inside the context is not equivalent and
does not apply those capability-backed components.

GEPA evaluation and reflection runs use a run-owned OpenTelemetry provider for
correlated trace capture. This intentionally isolates optimization traces from the
application's global Logfire/OpenTelemetry provider; ordinary runs of the original
agent retain their configured instrumentation.

Reflection receives training examples, feedback, and traces. Validation
rollouts retain only scores for selection: their feedback and side information
are discarded, and no validation trace file is written. Reflectors may see
aggregate validation scores and whether a candidate improved or was adopted.

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Run examples
uv run python examples/classification.py
uv run python examples/math_tools.py
uv run python examples/optimize_skills.py
```

`examples/optimize_skills.py` uses the built-in local skills search by default. For a faster in-process index that supports `reindex_skills(...)`, use `InMemorySkillsSearchProvider` and pass it via `skills_search_backend=...`.

### Running the Math Tools Example

The math tools walkthrough is the fastest way to see GEPA optimization in action. It expects API credentials in `.env`, so load them via `--env-file` when running.

```bash
uv run --env-file .env python examples/math_tools.py --results-dir optimization_results --max-evaluations 25

✅ Optimization result saved to: optimization_results/math_tools_optimization_20251117_181329.json
   Original score: 0.5417
   Best score: 0.9167
   Iterations: 1
   Metric calls: 44
   Improvement: 69.23%
```

After an optimization finishes you can re-run the same script in evaluation mode to benchmark a saved candidate:

```bash
uv run --env-file .env python examples/math_tools.py --results-dir optimization_results --evaluate-only
Evaluating candidate from optimization_results/math_tools_optimization_20251117_181329.json (best candidate (idx=1))

Evaluation summary
   Cases: 29
   Average score: 0.8931
   Lowest scores:
      - empty-range-edge: score=0.0000 | feedback=When the start exceeds the stop in a range, the result is an empty sequence. The sum of an empty sequence is zero. Answer 165.0 deviates from target 0.0 by 165; verify the computation logic and any rounding. A reliable approach uses: `sum(range(20, 10))`.
      - degenerate-average: score=0.0000 | feedback=Only one multiple exists in this narrow range. Ensure you handle single-element averages correctly. Answer 0.0 deviates from target 105.0 by 105; verify the computation logic and any rounding. A reliable approach uses: `sum(range(105, 106, 7)) / max(len(range(105, 106, 7)), 1)`.
      - between-1-2-empty: score=0.0000 | feedback=The next tool call(s) would exceed the tool_calls_limit of 5 (tool_calls=6).
      - between-10-11-empty: score=0.9000 | feedback=Exact match within tolerance. Used `run_python` 2 times; consolidate into a single sandbox execution when possible.
      - sign-heavy-expression: score=1.0000 | feedback=Exact match within tolerance.
```

## How It Works

### GEPA Graph Architecture

The optimization runs as a pydantic-graph workflow:

```
┌─────────────────────────────────────────────────────────────┐
│ GEPA Optimization Graph (pydantic-graph)                    │
│                                                             │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐           │
│  │  Start   │─────▶│ Evaluate │─────▶│ Continue │           │
│  │  Node    │      │   Node   │      │  or Stop │           │
│  └──────────┘      └──────────┘      └─────┬────┘           │
│                           ▲                │                │
│                           │                ▼                │
│                    ┌──────────┐      ┌──────────┐           │
│                    │  Merge   │◀─────│  Reflect │           │
│                    │  Node    │      │   Node   │           │
│                    └──────────┘      └──────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Nodes:**

- **StartNode** - Extract seed candidate from agent, initialize state
- **EvaluateNode** - Run validation set evaluation (parallel), update Pareto fronts
- **ContinueNode** - Check stopping conditions, decide next action (reflect/merge/stop)
- **ReflectNode** - Sample minibatch, analyze failures, propose improvements via LLM
- **MergeNode** - Genetic crossover of successful candidates (when enabled)

To enable merge/crossover (useful when different branches improve different components), set:

```python
result = await optimize_agent(
    ...,
    use_merge=True,
    max_merge_invocations=5,
)
```

For large component sets (e.g. when optimizing a skills pack with many skills/files), prefer `module_selector="reflection"` so the reflection agent can search/activate only the relevant skill components from traces.

Evaluations run in parallel for speed.

### Optimization Process

1. **Evaluate** - Score candidates on validation examples
2. **Reflect** - LLM analyzes failures and proposes improvements
3. **Merge** - Combine successful strategies (optional)
4. **Repeat** - Until convergence or budget exhausted

Results are cached to avoid redundant LLM calls.

## Engine Composition (omni)

Inspired by [GEPA's omni composition model](https://gepa-ai.github.io/gepa/blog/2026/07/22/optimize-anything-omni/), one `OptimizationTask` bundles an agent, dataset, and metric. Select any registered optimizer with `engine=` and compose engines under one shared metric-call budget.

| Engine | Role |
| --- | --- |
| `gepa` | Reflective pydantic-graph optimization loop. |
| `coding_agent` | A caller-supplied proposer owns reflection; the library owns the loop and selection. |
| `best_of_n` | A small reference engine that evaluates scripted or sampled variants. |

```python
from pydantic_ai_gepa import EngineConfig, OptimizationTask, optimize_best_of

task = OptimizationTask(agent=agent, trainset=trainset, valset=valset, metric=metric)

async def propose(seed):
    return improved_candidate(seed)

result = await optimize_best_of(
    task,
    [
        EngineConfig(engine="best_of_n", engine_config={"n": 2, "propose": propose}),
        EngineConfig(engine="gepa", max_metric_calls=20),
    ],
    max_metric_calls=40,
)
print(result.best.engine, result.fair_scores)
```

Use `optimize_parallel(...)` to keep every result, `optimize_sequential(...)` to chain engines without accepting a regression, or `optimize_vote(...)` to select by a fair evaluation vote.

Custom engines are one class implementing the `OptimizationEngine` protocol plus a registration: `register_engine("my_engine", MyEngine)`. Every engine shares the same budget; `optimize_best_of` and `optimize_vote` re-evaluate finalists on the valset without charging that budget, so winner selection is comparable across engines.

## Example

### Basic Optimization

```python
from pydantic_ai_gepa import optimize_agent
from pydantic_ai import Agent

# Define your agent
agent = Agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant.",
)

# Define evaluation metric
def metric(input_data, output) -> float:
    # Return 0.0-1.0 score
    return score

# Optimize
result = await optimize_agent(
    agent=agent,
    trainset=training_examples,
    metric=metric,
    max_metric_calls=100,
)

print(f"Best prompt: {result.best_candidate.system_prompt}")
print(f"Best score: {result.best_score}")
```

`best_score` is `None` when no candidate finished full validation, for example when the budget cannot cover the validation set; a measured zero remains `0.0`.

### With Structured Inputs (SignatureAgent Optimization)

```python
from pydantic import BaseModel, Field
from pydantic_ai_gepa import optimize_agent, SignatureAgent
from pydantic_ai import Agent

# Define structured input
class SentimentInput(BaseModel):
    """Analyze the sentiment of the given text."""

    text: str = Field(description="The text to analyze for sentiment")
    context: str | None = Field(
        default=None,
        description="Additional context about the text"
    )

# Create base agent
base_agent = Agent(
    model="openai:gpt-4o",
    output_type=str,
)

# Wrap with SignatureAgent to add input_type
agent = SignatureAgent(
    base_agent,
    input_type=SentimentInput,
)

# GEPA will optimize:
# - The class docstring ("Analyze the sentiment...")
# - Each field description
# - How they work together

result = await optimize_agent(
    agent=agent,
    trainset=examples,  # List[SentimentInput]
    metric=sentiment_metric,
)

# Access optimized signature components
optimized_instructions = result.best_candidate.components[
    "signature:SentimentInput:instructions"
]
optimized_text_desc = result.best_candidate.components[
    "signature:SentimentInput:text:desc"
]
```

## Project Structure

```
src/pydantic_ai_gepa/
├── runner.py          # Main optimize_agent entry point
├── components/        # GEPA optimization components
├── caching/          # LLM result caching
├── input_type.py     # Structured input utilities
└── ...

examples/             # Example optimization workflows
tests/                # Test suite
```

## External-reflection CLI

In addition to the Python `optimize_agent()` entry point, `pydantic-ai-gepa`
ships a `gepa` CLI for external reflection. Coding agents inspect training
reports and edit candidates; an orchestrator-owned harness scores held-out
runs. A training improvement must also improve validation before promotion.
Stochastic pipelines can configure acceptance repetitions, confidence and
minimum delta. Git runs can use `run start --lanes N` for parallel reflection;
see the [bundled skill](src/pydantic_ai_gepa/skills/gepa_optimize/SKILL.md).

Every single-path pause writes `runs/<id>/reflector_packet.json`: the current
candidate, scored baseline, training reports and traces, last comparison,
journal tail, budget, and an exact continuation command. If the reflector loses
its session or quota, run `gepa run resume --run-id <id> --reason "session lost"
--reflector <label>` and give the packet to another local agent or Mighty thread.
Resume records the old reflector epoch as `lost` and issues an `active` epoch,
without evaluating or changing the tree. Use the packet's `--reflector-epoch`
command to fence off stale sessions. Committed proposals are preserved,
completed comparisons are replayed without scoring again, and interrupted
comparisons reuse durable samples. Lane runs use `gepa lane reset` / `lane lease`.

Scalar acceptance uses Welch's Student-t confidence interval with a Bonferroni
alpha split across the configured maximum number of looks. Use
`--acceptance-repetitions 3 --acceptance-max-repetitions 5` for three repetitions,
up to five while inconclusive, at the default confidence 0.9. When omitted, the
maximum equals the configured initial repetitions. Small sets
require at least three repetitions per side. The failure-selected training
sample is excluded from the baseline; fresh samples supply the comparison.
Validation compares fresh candidate samples against repeated incumbent evidence,
and lane finalists receive a fresh confirmation before promotion. Each additional
evaluation consumes one budget row; insufficient budget cannot promote a candidate.

For larger sets, `gepa run start --acceptance-paired-min-cases 100` switches to one
repetition per side when there are at least 100 cases, using a paired Student-t
interval over matching per-case score differences. The threshold is configurable
(integer ≥2), and paired mode is disabled by default. It can also be set in
`.gepa/gepa.toml`; an explicit CLI value takes precedence:

```toml
[acceptance]
paired_min_cases = 100
```

Omit `paired_min_cases` to retain repeated evaluation for every dataset size.
The incumbent's aggregate validation samples are retained in `state.json` and
replaced only after an accepted promotion. Paired per-case evidence stays in the
harness-owned `.gepa-validation-evidence` directory beside the external dataset,
never in public run artifacts. Missing incumbent evidence prevents promotion until
it can be collected from the incumbent tree.

For held-out runs, the **harness process started by the orchestrator** holds
`GEPA_HELDOUT_DATASET` (an absolute path). Never give that variable, its value,
or the held-out directory to the reflector or any process it launches.
Scoring commands (`run start`, `harness serve`, `run select`, and `eval`) remove
the variable from `os.environ` before importing or running candidate code and
keep the path in controller memory during the command.
`gepa.toml` must contain only the training `dataset`; the old
`validation_dataset` setting and `init --validation-dataset` are refused with
exit 2. Held-out entries in the repository's `.env` are also refused. Legacy
state carrying a validation path is refused; start a new run in a clean workspace.

```bash
# Orchestrator / harness shell only:
gepa init --agent mypkg.agents:my_agent
export GEPA_HELDOUT_DATASET=/srv/gepa-heldout/my-project/validation.jsonl
gepa run start --heldout-required --max-iterations 50
gepa harness serve --run-id RUN_ID

# Separate reflector shell, launched without the private environment:
gepa run continue --run-id RUN_ID --reflector-epoch 1
```

`continue` atomically nominates the candidate and waits for a harness result.
It never evaluates a held-out run, even if the variable was accidentally set
in its environment. The harness takes the run lock, checks the epoch and clean
candidate identity, evaluates the training gate, then confirms training winners
on validation. It rechecks the tree before saving state and after scoring.
Held-out runs require a **committed, clean git candidate with scalar, unpinned
acceptance**. Component, vector and pinned-scorer held-out modes currently fail
closed before candidate imports. A dirty-tree refusal asks you to commit
before nominating. Keep the nominated tree unchanged until the result arrives.
Git-mode identity ignores global/user excludes (`core.excludesFile` and
`~/.config/git/ignore`); move those patterns to `.gitignore` or a regular
`.git/info/exclude` file. Git LFS and other filter-based repositories are not
supported: filters never run, so materialized filter-based files look dirty
and cannot be nominated.
The harness checks the shared checkout for stale nominations, but scores a
private checkout of the nominated commit's raw objects. No worktree registration,
index update, or ref write exposes that checkout in the shared repository.

The reflector receives training feedback reports, aggregate validation verdicts
and exit codes. Arbitrary child-written traces are kept private and discarded. `--wait-secs` defaults to 300; `0` enqueues
and returns immediately. Nomination waits up to five seconds for a busy run
lock; either a lock timeout or a result timeout exits **75**. Run the same command
again to reattach without duplicating the nomination. A different pending candidate or
gate selection is refused. `harness serve --once` processes one current
nomination (and retires old epochs), then returns. Interrupted scoring resumes
from the existing paid evaluation ledger. Normal `run resume` still issues a
new epoch and packet without reading validation.

Results contain explicit controller messages only; candidate/evaluator stdout,
stderr, and logging streams are never relayed to the reflector. Unexpected
scoring exceptions produce a retryable exit **1** result naming only the exception
type and the training/held-out phase. The harness keeps serving. Unpaid or
changed-candidate continuations are retired while paid evaluations remain charged,
so a fixed commit can be nominated without operator cleanup. After a run finishes,
`continue` returns its public final status; any undelivered terminal result keeps
its original output and exit code. Lane runs immediately direct callers to
`lane continue` and `run select` without queuing a nomination.

Harness commands are held-out `run start`, `harness serve`, lane `run select`,
`eval --dataset-role validation`, and held-out `run resume --abandon-continuation`
when private recovery evidence must be retired. They fail closed without the
harness environment. Reflector commands are `run continue`, `run status`, normal
`run resume`, and reading `reflector_packet.json`. `lane continue` evaluates only
training; the orchestrator's `run select` performs held-out selection. Runs
started without the variable or `--heldout-required` retain in-process,
training-only continuation.

The harness stores the dataset path and digest in a private `.gepa-heldout/`
record beside the dataset. Paired per-case evidence stays in the neighboring
`.gepa-validation-evidence/`, with detailed spend checkpoints in
`.gepa-validation-spend/`; all must remain inaccessible to the reflector.
Public state records `heldout_required` and aggregates only. Dataset identity
changes, paths inside any checkout or `GEPA_DIR`, and data recoverable
from Git objects are refused. If Git ever held the data, use a fresh repository
without those objects. Validation produces no public reports or traces.

The reflector's sandbox must restrict data reads to its worktree and
`GEPA_DIR`, plus the runtime paths needed to execute commands. **Keep the entire
held-out directory outside every path the sandbox can read**, including
system temporary directories: Codex's `:minimal` grant includes `/tmp`,
`/private/tmp`, and system temporary locations such as `/var/folders` and
`$TMPDIR`. Putting validation outside the worktree and `GEPA_DIR` alone does
not establish isolation. A home-directory sibling such as
`/Users/operator/.gepa-heldout/run-1/` works when no readable grant covers it.

This named profile was verified locally with Codex CLI 0.157.0. Write it to the
launcher's `CODEX_HOME/config.toml` (a scratch `CODEX_HOME` works for probes):

```toml
[permissions.gepa-reflector.filesystem]
":minimal" = "read"
"/absolute/candidate-worktree" = "write"
"/absolute/training-gepa-dir" = "write"

[permissions.gepa-reflector.network]
enabled = false
```

Use `codex sandbox -P gepa-reflector -C /absolute/candidate-worktree -- COMMAND`
with that `CODEX_HOME`. Define the profile in the config file rather than `-c`
overrides: dotted path keys (for example `~/.gepa/...`) can be split on their
dots by override processing. Provision Python and its dependencies under the
worktree's interpreter directory, or grant narrowly scoped runtime reads.
Do not inherit `:workspace`/`:read-only` or grant `"/"` reads. See the
[Codex permissions reference](https://learn.chatgpt.com/docs/config-file/config-reference).

The no-model probe verified that sandboxed environment/config/state/packet
output contained no held-out path, a direct read of the home-directory held-out
file failed with `Operation not permitted`, sandboxed `continue` enqueued,
the external harness scored, and sandboxed `continue` returned the accepted
verdict. Repeat this read-denial and scoring probe for the actual deployment's
paths and grants; system-temp placement does not protect held-out data.

Candidate code is scored in a **sandboxed child from a harness-owned detached
checkout**. The parent reads the held-out dataset and sends one case at a time
over a pipe; the child returns bounded JSON results. Candidate agent, evaluator,
metric, case-factory and pricing imports happen only in that child. Training
scoring inside a held-out harness also uses the child; training-only runs retain
the in-process evaluator. The parent never adds candidate import paths while
running held-out harness commands. After each evaluation it kills the worker's
process group, then sweeps same-user processes for the child's inherited
Seatbelt permissions to catch descendants that called `setsid()`. Finding a
survivor or failing process inspection refuses the evaluation instead of
returning a quality score. This repeated sweep is not an atomic process
container: enumeration and PID reuse races remain, and a descendant that changes
UID is outside the sweep. Private checkout/scratch directories are removed on
exit. Checkouts are built from raw Git tree/blob objects without worktree
conversion; symlinks, gitlinks and special files are refused.

The current backend is macOS Seatbelt (`/usr/bin/sandbox-exec`). It denies writes
outside private scratch, reads of the held-out directory outside the child's own
checkout/scratch, and network connections except to the harness's loopback CONNECT
proxy. The proxy listens on `127.0.0.1`; Seatbelt's `localhost:<port>` rule covers
loopback addresses on that TCP port. Fork/exec is allowed for per-case evaluator
subprocesses; descendants
inherit the same Seatbelt restrictions. Linux and hosts with an unusable backend
**refuse held-out scoring**; there is no unsandboxed fallback or disable flag. macOS CI requires real OS adversarial
tests. Seatbelt is deprecated; repeat those tests on each deployment OS. Local
Codex may prohibit nested Seatbelt, so a local skip is not enforcement evidence.

Set `GEPA_HARNESS_ALLOWED_HOSTS=api.openai.com:443,api.anthropic.com:443` in the
**harness environment** to allow exact CONNECT destinations. The default is empty
(no external network). Clients must honor `HTTPS_PROXY`/`HTTP_PROXY`; ordinary
forward-proxy HTTP requests and direct connections are refused. The child inherits
the fixed provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY`, `GEMINI_API_KEY`) and sandbox settings. To pass another evaluator
setting, set `GEPA_HARNESS_PASS_ENV=TYPESAFE_API_KEY,OTHER_NAME` in the harness
environment. Names are never read from a nomination, `gepa.toml` or `.env`.
Reserved harness/runtime names (including `GEPA_HELDOUT_DATASET` and
`GEPA_HARNESS_ALLOWED_HOSTS`) and values containing the held-out path are refused.
Held-out harness commands skip candidate-owned `.env` files. Install the harness and its Python dependencies
in storage the reflector cannot modify, and keep provider credentials out of the
reflector environment.

Harness Git commands read reflector objects through a harness-owned repository,
without loading reflector repository, global or system configuration, executable
attribute drivers, hooks, fsmonitor, filters, textconv, external diff or lazy fetch.
The harness pins an absolute Git executable once per process, requiring root
ownership and no group/other write permissions on the executable and every
ancestor directory. On macOS it appends `usr/bin/git` to the developer directory
named by the root-owned `/var/db/xcode_select_link`, or falls back to
`/Library/Developer/CommandLineTools/usr/bin/git`;
it never executes the `/usr/bin/git` xcrun shim or honors `DEVELOPER_DIR`.
On other systems it checks `/usr/bin/git`, then `/bin/git`. No qualifying binary
means refusal. Git retains its compiled helper location; `GIT_EXEC_PATH` is not set.
Reflector metadata reads refuse symlinks, special files and oversized files;
symlink or special-file `info/exclude` entries are treated as empty. Worktree
attributes can still apply Git's built-in data conversions. Harness Git storage
lives beside the held-out set and must be outside the reflector's writable roots,
just like the private scoring checkout.
Public candidate retention refs are written as data through directory handles
that refuse symlinks, without invoking source Git hooks; ordinary branch resets
and garbage collection preserve the retained commits.

Held-out lane start/select and all lane Git mutations (including worktree
creation/removal, reset, checkout and branch updates) currently fail closed when
`GEPA_HELDOUT_DATASET` is configured. Use a single-checkout run (`--lanes 0`) for
held-out scoring; safe lane mutations require a separate design.

Sandbox rollouts run serially; the parent retains cost admission and response-level
accounting. Public spend uses a fixed `sandbox` model bucket to prevent model names
from carrying held-out text. Arbitrary output/side-info/trace artifacts are not
copied back; training feedback remains available.

The orchestrator must enforce the reflector's sandbox: the harness cannot
identify the reflector's profile or refuse an unverified one. Require denial of
SysV IPC listing and use. Preventing concurrent process-argument disclosure also
requires that the reflector cannot read other processes' arguments; the two
Codex profiles below **do not meet that requirement** for same-user scoring.
A separate scoring UID would close that argv channel; this library does not
provision one. Unsandboxed reflectors (including Pi used without a sandbox)
must not be used for code-engine held-out runs. Claude Code's sandbox and
Pi/Kimi reflectors have not been verified and are not on the verified list.

Host probes on **2026-09-25**, using **macOS 26.6.2** and
**codex-cli 0.157.0**, produced the following results. “Passed” means the probe
was denied; “failed” means another same-user process's data was readable.
These results verify only the listed operations, not complete isolation.

| Reflector profile | `ipcs` listing | SysV use | `/bin/ps` | Raw `KERN_PROCARGS2` |
| --- | --- | --- | --- | --- |
| README `gepa-reflector` | Passed | Passed | Passed | **Failed** |
| Built-in `:workspace` | Passed | Passed | Passed | **Failed** |

SysV use probes covered `shmat`, `msgsnd`, `msgrcv`, and `semctl GETVAL`; all
were denied under both profiles. Those two profiles close the tested SysV
channel on the reader side. The `:workspace` results do not replace the
held-out file-read restrictions described above. Denial of `/bin/ps` does not
prevent raw `KERN_PROCARGS2` from returning another same-user process's
arguments and environment.

What this does not cover:

- Scores and numeric usage feed public aggregates, creating low-bandwidth covert
  channels across repeated evaluations.
- Traffic to an allowlisted model endpoint may be readable through its provider
  account. Keep harness credentials out of the reflector, but this is not enough:
  candidate code can also supply its own credentials to an allowed host. CONNECT
  restricts destinations, not accounts, HTTP methods or request contents.
- SysV shared memory, message queues and semaphore sets created by the scoring
  child can outlive it: Seatbelt does not mediate their creation. The library
  does not remove them. macOS records no creator PID for queues or semaphore
  sets, so it cannot distinguish them from objects created by other programs.
  The project handles these objects through the reflector-profile requirement;
  reader-side denial is verified only for the two Codex profiles above. A
  separate scoring UID or VM is a hardening follow-up.
- Process arguments and environment visible to another same-user process during
  scoring. Raw `KERN_PROCARGS2` works under both tested reflector profiles, and
  a process can rewrite its own argument strings to expose case data. No child
  Seatbelt profile prevents this reader-side channel. Survivor cleanup does not
  close the during-scoring window; a different scoring UID would.
- Timing, CPU, memory, `flock` signalling on readable files, and denial-of-service
  channels. The sandbox is not a resource limiter (the result reader has a
  timeout and message-size bound).
- OS/kernel exploits, changes to the trusted harness/runtime, or other private
  datasets outside the configured held-out directory.
- Evaluators that need a local database or another local service. The child has
  no direct local-service ports; only the allowlisted model proxy is reachable.
- Parallel rollouts within a scoring child; scoring is serial.
- Authentication of shared results/state and other `GEPA_DIR` coordination files;
  a reflector with write access can still forge them. This needs a separate
  ownership/authentication change.

Run files are shared coordination state, not authenticated messages; protection
against a reflector forging state/results in `GEPA_DIR` requires an additional
ownership or IPC boundary.

### CLI rollout spend caps

`gepa run start --max-token-cost 5.0` caps the managed run's metered rollout
spend in US dollars; the value must be finite and positive. `gepa eval --run-id`
obeys that run's cap, and a one-off `gepa eval --max-token-cost 5.0` can impose a
cap too. Status, start/continue summaries, and the final report include aggregate
training-side and validation dollars, per-model tokens/requests/dollars, and
unpriced usage, even without a cap. Reflector subscription/spend is excluded.
An incompatible cap (prior unmetered/unpriced work or a one-off limit below
recorded spend) exits 2 without changing the ledger or managed run state. A
tighter one-off cap stops that eval without finalizing the managed run.
Unpriced or unmetered work under a one-off cap also invalidates a managed run's
own cap and finalizes it with the fail-closed reason.

The run's locked `spend.jsonl` ledger checkpoints training-side deltas after each
response. Validation response checkpoints stay beside the private validation
evidence outside checkouts; the workspace receives one aggregate row when a
validation eval ends. Budget checks include all private checkpoints. Public
reports withhold live validation spend and tokens with `validation_in_progress: true`,
then include the published aggregate when the eval ends. The harness can also
recover paid spend from private checkpoints after an owner dies. Reflector status
without the harness environment reads published aggregates only; unfinished
validation remains flagged and blocks admission until the harness recovers it.
If a checkpoint is missing, harness status warns with
`validation_checkpoint_missing: true`; admission refuses when any registered eval
lacks a published aggregate. Validation's highest rollout cost stays private for
harness admission checks. Public files contain no private checkpoint paths.
Harness spend-cap stops pass their output and exit code 70 through the nomination
result, preserving the incumbent when validation is incomplete.
For a one-case validation set, the aggregate total inherently reveals that case's cost.
Status reports a torn final ledger line with `ledger_torn_tail: true`; evaluation
refuses a malformed ledger rather than assuming the missing spend is zero.
Batch projections use each evaluation kind's own observed mean. Short file locks
reserve projected batch costs against other processes' outstanding reservations;
finished evals settle to actual spend, and reservations whose owning PID is gone
are reclaimed on the next admission/check. Capped rollouts keep configured
concurrency while remaining headroom covers concurrency × the kind's highest
observed rollout cost. An unobserved kind, or a batch near the cap, starts one
rollout at a time and rechecks before each start. Uncapped concurrency is unchanged.

The margin is **one rollout per concurrent eval process**, provided no rollout
costs more than the highest observed for its kind. A new price high can add the
cost of rollouts already in flight at that moment. Response prices are known only
after the call; this is not a prepaid guarantee. All received responses, including
in-flight overshoot, are charged. A cost stop exits 70, preserves the incumbent,
writes the final report, and never makes a partial eval selectable.

For custom models, set `price_fn = "my_suite:price"` in `gepa.toml`, where
`price(response: ModelResponse) -> float | None` returns dollars (`None` uses
the bundled offline price catalog). Pricing is loaded from the primary scorer
workspace. Unknown prices stop capped work rather than counting as free.

Agent rollouts are metered automatically. Plain `evaluate` callables and metrics
must attach the context-scoped capability to **every** student, judge, or nested
agent they run:

```python
from pydantic_ai_gepa.spend import current_rollout_capability

capability = current_rollout_capability()  # None outside a CLI evaluation
result = await judge.run(prompt, capabilities=[capability] if capability else [])
```

For a suite-owned cache hit, call `pydantic_ai_gepa.spend.report_cached_rollout()`
from `evaluate` or its metric. Declared hits without model responses count as
`cached_rollouts` at $0 and do not lower cost projections; fresh calls still need metering.

Judge usage is included in rollout spend. A successful capped callable rollout
with neither metered responses nor a cache declaration stops with
`Evaluate callable reported no spend`;
suites that make no model calls should omit the cap. Ordinary failed rollouts
retain the existing failure/retry behavior. Workspace validation ledger entries
contain only eval aggregates, with no case identifiers, scores, or per-case costs.

### Pinned-scorer component IDs

Assertion-vector runs can set `acceptance.pinned_scorer = true` to load the
metric, comparator, and reviewer from the incumbent workspace instead of the
candidate worktree. In this mode, every entry in
`acceptance.component_files` serves two roles: it is an allowed relative file
path and the exact component ID supplied in the candidate component map. For
example, `component_files = ["prompts/planner.md"]` supplies the component as
`{"prompts/planner.md": <text>}`. Downstream harnesses must accept these
file-path component IDs. Non-component run metadata belongs in
`acceptance.meta_files`; `prediction.json` is always treated as metadata.

## More Info

- **[docs/gepa.md](docs/gepa.md)** - GEPA algorithm details
- **[gepa-ai/gepa](https://github.com/gepa-ai/gepa)** - Original implementation
- **[pydantic-graph docs](https://ai.pydantic.dev/graph/)** - Workflow execution
- **[pydantic-ai docs](https://ai.pydantic.dev/)** - Agent framework

## Configuration

Key arguments for `optimize_agent`:

`max_token_cost` uses observed mean costs to guard the next step. The first step
and in-flight requests can overshoot; `result.spend_report` reports actual spend.
The cap applies per engine run; composition helpers' comparison evaluations are not metered.
Unknown model prices stop capped runs; pass `price_fn(response) -> float | None`
to supply custom prices in dollars (`None` falls back to the bundled catalog).

```python
from pydantic_ai_gepa import ReflectionConfig

result = await optimize_agent(
    ...,
    # Budget
    max_metric_calls=200,          # Maximum number of evaluations
    max_token_cost=5.0,            # Optional US dollar cap across reflection + rollouts

    # Reflection settings
    reflection_config=ReflectionConfig(
        model="openai:gpt-4o",
        include_case_metadata=True,
        include_expected_output=True,
    ),
    reflection_minibatch_size=5,   # Examples per reflection
    track_component_hypotheses=True, # Persist reasoning metadata

    # Merging
    use_merge=True,
    max_merge_invocations=5,

    # Strategy selection
    candidate_selection_strategy="pareto",  # or "current_best"
    module_selector="round_robin",          # or "all"

    # Tool & Output Optimization
    optimize_tools=True,
    optimize_output_type=True,

    # Skill usage
    skills="path/to/skills",
    skills_capabilities={SkillCapability.READ}, # Explicitly opt-in to capabilities (READ is default)
)
```

## Advanced Features

### Custom Metrics

```python
from pydantic_ai_gepa import MetricResult

def custom_metric(input_data, output) -> MetricResult:
    """Metric with score and feedback."""
    score = evaluate_output(output)
    feedback = generate_feedback(input_data, output) if score < 1.0 else None

    return MetricResult(score=score, feedback=feedback)
```

### Result Caching

```python
from pydantic_ai_gepa import metric_code_identity

result = await optimize_agent(
    agent=agent,
    trainset=trainset,
    metric=metric,
    enable_cache=True,
    cache_dir=".gepa_cache",
    # Caching fails closed: nothing is stored unless you opt in explicitly.
    cache_metric_results=True,  # cache metric scores (requires an identity)
    cache_metric_identity=metric_code_identity(metric),  # or a version string
    cache_rollouts=True,  # cache agent runs (freezes each rollout's first sample)
)
# Second run reuses cached metric results and agent runs
```

Rollouts and judge-based metrics call sampled models, so caching freezes the
first sample. The cache therefore fails closed: `enable_cache=True` without an
explicit opt-in raises `ValueError` before any model call, and
`cache_metric_results=True` requires `cache_metric_identity` — a string that
must change whenever the metric, its grader, or judge prompts change (a version
string, a freeze-manifest hash, or `metric_code_identity(metric)`; the helper
covers only the function's own source, not helpers it calls or prompt files it
reads).

A cached entry is invalidated (a cache miss) whenever any of these change:
the metric identity, the case's expected output (gold) or case-level
evaluators, the candidate text, the case inputs/name/metadata, the rollout
output (for metric results), or the model identifier.

## Development

```bash
# Install everything (library + dev tools)
uv sync --all-extras

# Install git hooks (ruff lint/format + pyproject schema check)
uv run pre-commit install

# Lint & format
uv run ruff check .
uv run ruff format .

# Tests and type checks
uv run pytest
uv run pyright

# Run all hooks on-demand
uv run pre-commit run --all-files
```

## Experimental

This library is experimental and currently depends on pydantic-ai PR #5143 until those changes are released. Expect API changes.

## Contributing

See `AGENTS.md` for coding standards and contribution guidelines.

## License

MIT License - see LICENSE file for details.

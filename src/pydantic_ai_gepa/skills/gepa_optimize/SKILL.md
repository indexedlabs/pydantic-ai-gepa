---
name: gepa-optimize
description: Optimize a pydantic-ai agent's instructions, tool descriptions, output type, and signature inputs using the gepa CLI. Trigger when the user asks to "optimize this agent", "improve tool descriptions", iterate on prompts driven by eval failures, or otherwise improve a pydantic-ai agent against a dataset. Operates in the user's repo with full filesystem access — YOU are the reflection model; the gepa CLI handles minibatches, evaluation, and Pareto bookkeeping.
---

# gepa-optimize

You are the reflection model. The `gepa` CLI is a small toolkit that handles minibatches, trace/report persistence, candidate evaluation, comparison, and history bookkeeping; you read failure reports and traces, edit component slots or source code, and continue until the run completes.

There is no `propose` or `reflect` verb on the CLI because that's the work you do — while `gepa run` is paused, or between manual `gepa eval` invocations — by editing files.

## Driving an already-started run

`gepa --gepa-dir /absolute/workspace run drive --run-id RUN --reflectors reflectors.json`
automates lane runs and single-checkout held-out runs with one short-lived
reflector per step. Run it in the harness environment. It leases/reset lanes,
dispatches the fixed training-reflection prompt, guards process exit, and calls
`run select` or `harness serve --once`. **No separate harness service, selector,
or reflector may run for a driven run: the driver owns scoring timing.** The
existing held-out lane refusal remains until standalone lane repositories land.

The JSON configuration contains only allowed commands, for example:

```json
{"pass_env":[],"reflectors":[{"label":"codex","argv":["codex","exec","--profile","gepa-reflector","-C","{checkout}","Read {prompt}; follow its instructions using {packet}."],"usage_limit":{"exit_codes":[9],"output_regexes":["(?i)usage limit"]}}]}
```

Configure detection for the actual command; exit 9 above is an example, not a
Codex exit-code guarantee. Usage-limit matching applies only when no nomination
or lane result exists. A legacy bare array means empty `pass_env`. Templates accept
`{checkout}`, `{packet}`, `{prompt}`.
The default is one Codex entry using `--profile gepa-reflector` and usage-limit
output detection. In current `exec --help`, `--profile` loads a separate
`$CODEX_HOME/gepa-reflector.config.toml` overlay; provision its actual permissions.
It is not the `codex sandbox -P` permissions selector. Follow the README's dated
verified-profile table and held-out read restrictions. Verification is the caller's
responsibility; never list unverified Claude Code or Pi profiles for held-out runs.

`--step-timeout` defaults to 900 seconds; `--max-attempts` to 2; `--max-steps`
optionally bounds sessions. Lane nomination adds `--foreground`; single-checkout
nomination adds `--wait-secs 0`. Keep harness credentials out of the reflector.
Provider keys are scrubbed by default; `pass_env` accepts explicit **training-only**
variable names such as `OPENAI_API_KEY`, never patterns. Do not pass scoring keys.
`GEPA_HELDOUT_*`, `GEPA_HARNESS_*`, names in `GEPA_HARNESS_PASS_ENV`, candidate
components and trace variables cannot be allowlisted. Subscription configuration
is retained; keep other harness secrets out of the launch environment. Git config
appends `gc.autoDetach=false` and `maintenance.autoDetach=false` to existing entries.
Public step directories hold training-only prompts and logs. Held-out state lives
beside the pin. Training-only state lives under
`${XDG_STATE_HOME:-~/.local/state}/pydantic-ai-gepa/drive/<key>/drive.json`, keyed by
resolved GEPA_DIR and run ID. State/lock files are 0600 in a current-user-owned
0700 directory outside GEPA_DIR, project and lane worktrees. The reflector sandbox
must not grant that directory. Legacy public `run_dir/drive.json` is refused,
never loaded or migrated. Forging private state is outside the pin's threat model.

Restart the same drive command after a kill: unfinished reflectors are guarded
first, interrupted lanes reset, and scoring reuses existing durable checkpoints.
Events are acknowledged after recording the action; merge opportunities are never
auto-merged. A changed list requires `--accept-reflector-change`. Exhausted fallback
lists restart from the first entry. Exit 0 prints the final report; 2 is input/list
refusal; 76 is an attempt/step/scoring pause; 77 is usage-limit exhaustion; 78 is
process-guard refusal. Scoring errors do not retry: recover the service, explicitly
resume the managed run, then drive again.

### What the process guard does not detect

The macOS guard uses libproc's immutable process and original-parent unique IDs,
polls at 50 ms, and retains observed ancestry across restarts. After group cleanup
and a final snapshot it persists a step-end ceiling. Only IDs between the starting
watermark and that ceiling, or descendants linked through those IDs, count. A
restart of an interrupted step without a ceiling freezes it at the first restart
snapshot; unrelated later births cannot keep extending the window.

The guard kills attributed reflector descendants and ignores processes provably
rooted in another pre-existing process. An otherwise unknown Apple job is unrelated
**only when all three checks pass**: its original-parent unique ID is launchd's
(not merely its current PPID); `proc_pidpath` is under `/System/` or `/usr/libexec/`
and `csops` reports `CS_PLATFORM_BINARY`; and its responsible PID is available and
differs from both the driver's and the reflector root's responsible PIDs. A missing
or failed check leaves it unknown. Unknown processes are never killed: after
`--survivor-grace` (default 5 seconds) they pause scoring with PID/command diagnostics.
Unsupported platforms refuse to drive.

- Launchd/XPC/`launchctl`/`open` work with unknown ancestry pauses the drive. The
  Apple job exemption above is a gap if a reflector induces a job satisfying all
  its checks to do work on its behalf. Orphaning a platform binary alone does not
  qualify; the original-parent identity and responsibility checks still apply.
- Work done by an already-running same-UID process (tmux server, another shell,
  a loaded LaunchAgent, an ssh ControlMaster) escapes attribution.
- Processes that change UID/GID through setuid/setgid execution can leave the
  same-UID snapshot and escape inspection.
- An ordinary double-fork can lose its intermediate before the 50 ms poll. That
  reader remains unknown and alive, so ordinary daemonizing tools can stall the
  drive. Git's no-detach settings reduce one benign source; no fork-event tracking
  is installed. Unknown survivors can deny service by keeping the drive paused.
- Start the driver with no reflector running. Pre-existing reflectors and
  nominations are not guarded; a pending single-checkout nomination at startup is
  scored without a step guard.
- Same-UID processes can hide or act before inspection. Snapshots and signals are
  not atomic; identity rechecks reduce but cannot eliminate PID reuse races. A
  future Linux PID-plus-start-time backend would also need to address reuse.
- Reads during the reflector step remain possible. The guard separates reflection
  from scoring in time; it does not deny reads of other processes' argv while the
  reflector runs. The dated table shows raw `KERN_PROCARGS2` reads remain possible;
  this is why process lifetime separation is needed.

A separate scoring UID would close these same-UID disclosure and interference
channels; the driver does not provision one.

## Outer Omni protocol for code candidates

`gepa run --lanes` is intentionally a single-parent managed run. Do not try to
turn its lane count into a multi-parent Omni plan. For independently explored
git/code candidates, a root orchestrator drives the separate durable outer
controller and routes packets without reading reflection text:

```bash
gepa omni start --plan ./omni-plan.json
gepa omni next <omni-id> --json
# create only the workspace named in child_ready/phase2_ready, then launch your worker
gepa omni child-dispatched <omni-id> --receipt ./dispatch.json
gepa omni child-submit <omni-id> --receipt ./child-result.json
gepa omni compare-submit <omni-id> --receipt ./comparison.json
gepa omni reporting-submit <omni-id> --receipt ./report.json  # only if reporting was planned
gepa omni ack <omni-id> <event-id>
```

The plan uses SHA-256-pinned seed/minibatch/test artifacts, evaluator identity
and digest, equal phase-one metric-call slices, an explicit repeated comparison
budget, and a fresh phase-two workspace. Workspace paths may be intended paths
at `start`; create or verify the exact isolated directory before submitting the
dispatch receipt. A `child_ready` packet is all the
worker needs: its child/engine ID, isolated workspace, immutable seed and
minibatch paths/hashes, opaque SHA-pinned driver manifest, and reserved metric
calls. The controller never executes the manifest; the orchestrator reads it
to choose its engine adapter. Submit only immutable
receipts with those same identities. The controller, not child self-reported
scores, chooses a Pareto provisional winner, then accepts it only when the
shared confidence-interval/practical-delta comparison clears the frozen
threshold against seed/incumbent. Rejected, equivalent, or inconclusive votes
retain the baseline; an optional bounded `max_repetitions` can collect more
matched samples only for an inconclusive vote. Events redeliver unacked work
after restart.

Keep the emitted packet untouched: its original SHA is recorded in the durable
outbox, and dispatch/semantic receipts are rejected if the packet, plan, or
frozen input bytes changed after emission.

Child optimization usage is an explicit `child_receipt` attestation. The outer
controller enforces its reserved ceiling but does not independently meter an
arbitrary worker; production adapters should bind manifests to evaluator-owned
usage-ledger receipts.

Use `error-submit` only for an explicit, evidence-backed frozen evaluator target
that is inconsistent or unattainable. Do not emit `ERROR.md` for ordinary code,
test, provider, credential, or worker failures.

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

Write reflection-training cases at `.gepa/dataset.jsonl`, one JSON object per
line. The harness provisions held-out selection cases in the same format at
the harness-only `GEPA_HELDOUT_DATASET` path:

```json
{"name": "case-1", "inputs": "...", "expected_output": "...", "metadata": {}}
```

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
Held-out runs require a **committed, clean git candidate with scalar acceptance**.
They support unpinned code scoring or the trusted-scorer mode described below.
Component candidates, vector comparators and pinned scoring without
`trusted_scorer` fail closed before candidate imports. A dirty-tree refusal asks you to commit
before nominating. Keep the nominated tree unchanged until the result arrives.
The harness checks the shared checkout for stale nominations, but scores a
private checkout of raw Git objects (the nominated commit for unpinned scoring,
or the harness-pinned scorer revision for trusted scoring). No worktree registration,
index update, or ref write exposes that checkout in the shared repository.

The reflector receives training feedback reports, aggregate validation verdicts
and exit codes. Arbitrary child-written traces are kept private and discarded. `--wait-secs` defaults to 300; `0` enqueues
and returns immediately. Nomination waits up to five seconds for a busy run
lock; either a lock timeout or a result timeout exits **75**. Run the same command
again to reattach without duplicating the nomination. A different pending candidate or
gate selection is refused. `harness serve --once` processes one current
nomination (and retires old epochs), then returns. Interrupted scoring resumes
from the existing paid evaluation ledger. Harness-side `run resume` issues a
new epoch and packet without evaluating validation.

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
`eval --dataset-role validation`, and held-out `run resume` (including
`--abandon-continuation`). They fail closed without the harness environment.
Reflector commands are `run continue`, `run status`, and reading
`reflector_packet.json`. `lane continue` evaluates only
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

For **trusted-scorer held-out scoring**, freeze these settings in `gepa.toml`:

```toml
[acceptance]
mode = "scalar"
pinned_scorer = true
trusted_scorer = true
component_files = ["prompts/planner.md"]
```

The orchestrator must set `GEPA_HARNESS_SCORER_REVISION` to the trusted scorer's
full commit SHA (40 or 64 hexadecimal characters) in the harness environment,
including `run start`, `harness serve` and `run select`. The revision must exist
in the same repository. Missing, malformed or non-commit revisions are refused.
The private checkout contains only that revision. The parent reads the nominated
commit's component files as raw UTF-8 Git blobs, with no filters or candidate
imports, and sends their text in the initialization message. The child sets
`GEPA_CANDIDATE_COMPONENTS_JSON` before importing the scorer. Keys are the exact
`component_files` paths, relative to the Python project root. Missing files,
symlinks, non-blobs and invalid UTF-8 are refused. The scorer must consume this
text as data. The candidate's tree is never written into the child's checkout.

For either held-out mode, the harness can set `GEPA_HARNESS_FROZEN_FILES` to a
JSON object mapping **Git repository root-relative paths** to SHA-256 hashes,
for example `{"api/evals/scorer.py": "<64 hex characters>"}`. The trusted parent
verifies the exact private-checkout bytes before starting the child: candidate
revision files in unpinned mode, trusted revision files in trusted mode. A bad
manifest, unsafe path, missing/non-regular file or hash mismatch is a static
infrastructure refusal, never a quality score. Omission means no frozen-file
checks; launchers requiring a freeze must supply the complete mapping. Both
variables come only from the orchestrator's environment, never nominations,
reflector config or run state, and cannot be forwarded with
`GEPA_HARNESS_PASS_ENV`. Training-only `gepa eval` behavior is unchanged.

The parent verifies the commit, every tree and every consumed blob against its
Git object ID (SHA-1 or SHA-256) before launching the child. Shared object-store
substitutions are refused. Frozen manifest paths must match verified tree names
exactly, including case. When `GEPA_HARNESS_FROZEN_FILES` is set (even to `{}`),
checkouts cannot contain `__pycache__/`, `.pyc`, `.so` or `.pyd` files, or package
directories shadowing a frozen `.py` module. In unpinned code-engine mode,
frozen hashes detect edits; they do not isolate the scorer from candidate code
running in the same interpreter. **Trusted mode is the scorer isolation boundary.**

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
`GEPA_HARNESS_ALLOWED_HOSTS`, `GEPA_HARNESS_SCORER_REVISION`,
`GEPA_HARNESS_FROZEN_FILES` and `GEPA_CANDIDATE_COMPONENTS_JSON`) and values
containing the held-out path are refused.
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
Xcode under `/Applications` does not qualify while that directory is
admin-group-writable (the default); such hosts need the Command Line Tools installed.
On other systems it checks `/usr/bin/git`, then `/bin/git`. No qualifying binary
means refusal. Git retains its compiled helper location; `GIT_EXEC_PATH` is not set.
Reflector metadata reads refuse symlinks, hard links and special files, except
that these `info/exclude` entries are treated as empty. Oversized metadata,
including `info/exclude`, fails closed. Worktree
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
- Separate OS ownership of the harness and reflector. A reflector that can write
  beside the held-out dataset can defeat the private record; filesystem isolation
  of that directory remains required.

For held-out runs, harness-written files in `GEPA_DIR` are views: the harness acts
only on its private record beside the held-out dataset, refusing and restoring
changed views. A missing private record fails closed; legacy held-out runs must
be restarted. Runs without held-out validation keep their existing behavior.
Held-out epoch changes and `run resume` require the harness or orchestrator with
`GEPA_HELDOUT_DATASET`; reflector-side resume is refused. Harness `run status` on
a finished held-out run rewrites `final_report.md` from the private record.

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

Git-mode identity ignores global/user excludes (`core.excludesFile` and
`~/.config/git/ignore`); move those patterns to `.gitignore` or a regular
`.git/info/exclude` file. Git LFS and other filter-based repositories are not
supported: filters never run, so materialized filter-based files look dirty
and cannot be nominated.

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

`continue` first evaluates the new commit on the same training minibatch as
the reflection baseline. Only a training improvement spends a full held-out
validation evaluation. Validation—not the training minibatch—decides whether
`best_candidate_id` and `best_commit_sha` advance. The CLI persists no
validation report, trace, per-case score, or minibatch manifest for reflection.
Status/start/continue summaries, lane packets and events, Pareto history, and
the final report expose only aggregate validation scores and outcomes. Training
reports and traces retain their full feedback. Validation evaluations write no
report or trace file.
When either gate rejects the candidate, the CLI pauses with:

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

Before each trace-enabled training evaluation, the CLI exports `GEPA_TRACE_FILE` with
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
if verdict=accepted, the candidate improved training and held-out validation; keep it and advance
if verdict=rejected or equivalent, discard/revise your edits and continue again
if verdict=inconclusive, do not count it as a failed hypothesis; revise it or end without a false decision
if it pauses_for_reflection, inspect the new report/trace and reflect again
repeat until the JSON summary says status=done and prints final_report_path
```

`gepa run start` scores the seed on held-out validation, then evaluates sampled
reflection-training minibatches until at least one case falls below
`--threshold`. It pauses and writes:

- `reflection_baseline_report_path`
- `reflection_baseline_report_paths`
- `reflection_baseline_trace_path`
- `reflection_baseline_trace_paths`
- `reflection_baseline_samples`
- `state_path`
- `next_command`

`gepa run continue` nominates your edited baseline. The orchestrator's harness
evaluates it against the same saved training minibatch. With repeated acceptance enabled, it compares rollout-level mean
samples and reports the observed variance, confidence interval, practical
minimum delta, and `accepted`, `rejected`, `equivalent`, or `inconclusive`
training verdict. The harness then evaluates a training-accepted proposal on the
complete held-out validation dataset, and only a validation improvement
advances the baseline. Repetitions remain full
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

### Resuming with another reflector

When the coding agent loses its session, hits quota, or must hand off, issue:

```bash
gepa run resume --run-id <run_id> --reason "session lost" --reflector <label>
```

Omit `--run-id` to resume the latest managed run. The command never evaluates,
changes the candidate tree, or consumes budget. It works at every single-path
pause and re-issues the final packet for a completed run. Its last JSON line is
`{"packet": ...}`; `start`, `continue`, and `status` also expose
`reflector_packet_path`. Hand the new agent that file at
`runs/<run_id>/reflector_packet.json`. It contains the scored baseline, current
tree identity, training reports and traces, last comparison, journal tail,
notes index, best candidate, budget, and instructions. Validation remains
aggregate-only. Local agents and Mighty threads receive the same packet.

Run `next_command.argv` in `next_command.cwd`, or use `next_command.shell` from
any directory. Both select the absolute workspace and carry
`--reflector-epoch N`. `continue` refuses a stale epoch with exit 2 and names the
current epoch. Omitting the flag preserves existing CLI behavior.

The loop phase remains in `status`. The separate `reflector` block has an epoch,
label, issue timestamp, and `active` or `lost` state. Resume records the previous
epoch as `lost`, with its reason and timestamp in a bounded history, then issues
an `active` epoch and appends a `reflector_lost` journal row. Repeating resume
simply issues another packet and epoch. There is no heartbeat or automatic
expiry; the operator decides when a reflector has been lost.

- **Death after commit, before continue:** the packet identifies the committed,
  unscored proposal. Review it and run `next_command`, or explicitly restore the
  reported baseline. The CLI never discards the commit.
- **Death after continue completed:** the packet retains the recorded verdict,
  including an accepted proposal after the loop advances. Repeating continue on
  the recorded candidate returns the result with exit 0 and no evaluations.
  In particular, after acceptance advances to `paused_for_reflection`, calling
  `continue` on the unchanged tree returns "already scored" without evaluating;
  edit the tree to move on.
  For a rejected proposal, restore the baseline to advance or revise the tree
  to compare a new proposal.
- **Death during continue:** durable training samples and any completed
  validation evaluation are recovered before evaluating remaining samples, so
  paid evaluations are not duplicated. Gate samples use a separate checkpoint
  because they do not write Pareto rows. Keep the interrupted candidate unchanged
  until its comparison finishes; the packet preserves any `--gate-case` options.

If a pending candidate cannot be restored (especially overwritten components),
use `gepa run resume --run-id <run_id> --abandon-continuation`. This journals and
drops the checkpoint, preserving all paid budget charges, so the next continue
starts a new comparison. A candidate change detected during evaluation also
drops the checkpoint. Paired validation replay snapshots are private and removed
when the continuation completes or is abandoned.

Continuation and resume share a run lock. A live holder causes exit 1; a process
death releases the lock automatically. Lane runs refuse this command with exit
2: use `gepa lane reset` and `gepa lane lease` instead.

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

Hard cap on eval rows in a single run. Repeated training baseline/candidate
evaluations and held-out validation evaluations all consume rows from this
same budget. One validation row evaluates the complete validation dataset, so
row count is a lifecycle bound rather than a direct model-call or cost bound.
In the managed loop,
`gepa run start` persists the budget and each `gepa run continue` advances
until it either pauses for reflection or reaches `status=done`. In one-off
`gepa eval`, exceeding the cap exits with code 70.

## Parallel reflection lanes

Use lanes when you want to explore several independent reflection directions
at once in git candidate mode. `gepa run start --lanes N` fans the run out
into N git worktrees — one per lane, each on branch
`gepa/lane/<run_id>/<lane>/<iteration>` cut from the current best — evaluates every
lane candidate in the background against one frozen shared baseline, and
coordinates everything through an event stream. You play two roles: a thin
**orchestrator** that consumes events and dispatches work, and short-lived
**reflector subagents** (and occasionally a merge subagent) that do the actual
reflection in isolated context.

Two operational notes from dogfooding:

- **Run the orchestrator where it can spawn children.** In Prime Agent /
  RLM-style runtimes the orchestrator must sit at depth 0 (or dispatch through
  the host) — a depth-capped nested orchestrator cannot spawn reflectors
  itself.
- **`.gepa/` inside the candidate repo means the committed tree snapshots
  journal state.** Restoring the best commit (`git checkout <best_commit_sha>`
  after `run_done`) leaves `.gepa/journal.jsonl` modified in the working tree
  — expected and harmless; commit or ignore it.

Requires `candidate_source = "git"` and a clean primary checkout — lane
branches are always cut from a clean commit. Component mode stays on the
single-path loop.

```bash
gepa run start --candidate-source git --lanes 3 --max-iterations 200 \
  --acceptance-repetitions 3 --acceptance-max-repetitions 5 \
  --straggler-timeout-secs 3600 --reflection-lease-secs 1800
```

Start emits one `lane_ready` event per lane. Each event payload carries
`packet_path` and `worktree_path` — everything needed to dispatch a reflector
without reading any other state.

### Orchestrator event loop

Lane verbs resolve the workspace explicitly (never from cwd) — export the
absolute workspace once and every orchestrator verb inherits it:

```bash
export GEPA_DIR="$(pwd)/.gepa"   # absolute; lane lease/continue/reset and run select require this
```

Drive the run by long-polling `gepa next` and dispatching on the event type:

```text
loop:
    event = gepa next --wait --timeout 300 --json --run-id <run_id>
    # exit 0: event delivered; exit 4: timeout (retry the loop)
    # (without --wait, exit 3 means none pending)
    match event.type:
        lane_ready:
            gepa lane lease <lane> --run-id <run_id>
            # lease-refused (exit 1) means already dispatched — wait for the
            # lease to expire (lane_stalled) or `gepa lane reset` to reclaim
            dispatch reflector subagent with the packet/worktree paths
        verdict:
            record it (verdict, delta, comparison_path); nothing to dispatch
        selection_due:
            gepa run select --run-id <run_id>
        merge_opportunity:
            dispatch a merge subagent for the two named branches (select keeps
            merge-pair branches until the next select; candidate SHAs are in
            the journal's lane_outcome entries if you need them)
        lane_stalled:
            gepa lane reset <lane> --run-id <run_id>
            re-dispatch the reflector with the same packet path
        budget_low:
            note remaining_evals; steer new dispatches toward cheap edits
        run_done:
            read final_report_path; stop the loop
            # a selection_due for a later iteration may arrive AFTER run_done
            # (the reaper synthesizes it before noticing done) — just ack it;
            # `gepa run select` on a done run refuses cleanly.
    gepa ack <event.id> --run-id <run_id>
```

Rules that keep the loop correct:

- **Route, never read.** Reflection content — traces, reports, diffs — never
  enters orchestrator context. Event payloads carry paths and scalars only.
  Dispatch subagents with those paths; do not open the packet, reports, or
  traces yourself.
- **Ack discipline.** `gepa ack <event_id>` only AFTER you have durably
  recorded your dispatch or decision (e.g. noted it in a scratch file or the
  subagent is launched). Events must be acked in delivery order — acking
  anything but the oldest unacked event is rejected (exit 5). If you crash,
  compact, or restart, `gepa next` redelivers unacked events verbatim; replay
  them to reconstruct exactly the pending work.
- **Lease before dispatch.** `gepa lane lease <lane>` records the dispatch in
  lane state; a leased lane rejects re-dispatch and (for a fresh spawn) a
  second `gepa lane continue` until the lease is consumed by the reflector's
  continue, reclaimed with `gepa lane reset`, or `--reflection-lease-secs`
  expires. There is no release verb — the lease ends via continue, reset, or
  expiry.
- **One orchestrator per run.** The event bus has a single consumer cursor;
  never run two orchestrator loops against the same run.
- **Stop on operator-required provider failures.** If an evaluating verb
  reports expired credentials or a provider billing/quota code
  (`insufficient_quota` or `credit_balance_exhausted`), record the reason and
  stop the loop. Do not retry until a human restores credentials or credit.
- Use `gepa run status --run-id <run_id>` for the lane board (status,
  candidate, verdict, progress per lane — always printed as JSON) whenever
  you need a ground-truth snapshot.

### Supervision protocol

When a run-verb JSON summary contains a `stall` block, fork a separate
supervisor session. The fork reviews the inherited trajectory, appends concise
redirect advice with `gepa journal append --strategy redirect`, and exits. The
primary loop continues normally; the next reflection receives that entry via
its journal tail. The supervisor fork's only write surface is the journal:
never edit components, commit candidates, mutate run state, or run `gepa
next` / `gepa ack`.

### Reflector notes

At session start, read the frontmatter index in `.gepa/notes/`; each
`<slug>.md` note provides a `name` and `description`. Load bodies only when
they are relevant. Promote proven journal insights into notes only BETWEEN
runs: reflectors do not create, edit, or delete notes during an active managed
run. Notes are reference material, never candidate components.

### Reflector subagent dispatch template

The reflector gets exactly three inputs — the packet path, the worktree path,
and an optional one-line steer. Never paste conversation history, other
lanes' work, or your own analysis into the dispatch:

```text
You are a reflection subagent for GEPA lane <lane>.

Inputs:
- Reflection packet: <packet_path> (read this first)
- Lane worktree: <worktree_path> (all edits happen here, on the lane branch)
- Steer: <one line, e.g. "focus on tool:lookup_order argument formatting">

The packet carries the baseline candidate's samples with report/trace paths,
metric side info, a journal tail of prior reflections, and the exact
`gepa lane continue` invocation. Work only from the packet plus the repo.

Do:
1. Read the packet, then the baseline reports/traces it points at.
2. Form one hypothesis and edit code/artifacts inside <worktree_path> ONLY.
3. Run the packet's continue_invocation verbatim as your terminal act:
   `gepa --gepa-dir <abs> lane continue <lane> --run-id <id>`
   It auto-commits the worktree onto the lane branch and starts the
   background acceptance eval. Your job ends there — do not report results
   back; the verdict arrives as an event.

Never:
- Touch the primary checkout or any other lane's worktree.
- Edit files under .gepa/ (run state, events, journal) by hand.
- Run `gepa run continue` / `gepa run select` — those belong to the
  orchestrator (and `run continue` errors out in a lane run by design).
```

The subagent's terminal act is `gepa lane continue` — it never relays
completion by hand. The background eval streams progress into lane state and
emits the `verdict` event; you will see it from `gepa next`.

### Selection, merges, and stalls

- **`selection_due` → `gepa run select --run-id <run_id>`.** Select is the
  single sequential authority: it validation-scores every training-accepted
  lane, promotes the strongest held-out improvement to the run's best,
  journals every loser (diff summary, verdict, delta) before
  deleting its branch, invalidates stragglers, enforces the budget, and
  re-fans every lane onto the new best with a fresh shared baseline and new
  `lane_ready` events. Never run `gepa run continue` in a lane run — it
  errors and points you at `lane continue` / `run select`.
- **`merge_opportunity` → dispatch a merge subagent.** When two accepted
  lanes' diffs touch disjoint file sets, select names the pair
  (`branch_a`, `branch_b`, plus a `diff_stat_path`). Dispatch a subagent to
  resolve a real `git merge` of the two branches — merging needs code
  understanding, so the CLI never auto-merges. The merged tree enters the
  NEXT iteration as a lane candidate, never as an auto-accepted best: land
  the merge result in one lane's worktree after the re-fan (a normal
  reflector dispatch can start from it), so it goes through the same
  `lane continue` acceptance eval as any other candidate.
- **`lane_stalled` → reset and re-dispatch.** A lane stalls when its
  reflection lease expires (reflector never ran `lane continue`) or its
  background eval died (stale heartbeat, dead pid). Run `gepa lane reset
  <lane> --run-id <run_id>` — it terminates a live recorded eval pid and
  returns the lane to paused; uncommitted worktree content is never
  auto-deleted. Then re-dispatch the reflector with the same packet path.
- **`budget_low` → tighten dispatches.** Emitted when remaining evals fall
  below lanes × (`--acceptance-max-repetitions` + one validation eval). Prefer cheap, high-confidence
  edits from here and expect `run_done` soon.
- **`run_done` → stop.** Read `final_report_path` for the outcome (including
  any budget overshoot) and restore the best commit with `git checkout
  <best_commit_sha>`.

### Lanes vs the single-path loop

Prefer lanes when ALL of these hold:

- The candidate source is git (component mode has one process-global agent
  and stays single-path).
- You have several independent reflection directions worth trying against the
  same baseline — lanes pay the baseline evals once per iteration and overlap
  reflection with evaluation, so wall-clock per accepted candidate approaches
  `max(reflection time, eval time / N)` instead of their sum.
- You can afford the coordination overhead: one orchestrator loop, N
  subagent dispatches per iteration, and a straggler timeout.

Stay on the single-path loop (`gepa run start` / `gepa run continue` without
`--lanes`) when the budget is tight (lanes spend up to N ×
(`--acceptance-max-repetitions` + one validation eval) in flight per iteration), when
reflections are sequential by nature (each depends on the last verdict), or
when you are in component mode.

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
| 1 | Recoverable error (missing file, invalid agent ref, dataset empty, orphan slots on `apply`, live run lock holder) | All verbs |
| 2 | Refusal — wrong input shape, unconfirmed slots, stale reflector epoch, or single-path resume of a lane run | `gepa eval` / `gepa run`, every verb on argparse errors |
| 70 | Hard cap — `--max-iterations` exceeded | `gepa eval` |

When you see exit 2, read stderr for the recovery command: component confirmation,
a fresh reflector packet, or the lane workflow.

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
    ├── reflector_packet.json  # versioned single-path handoff; training + validation aggregates
    ├── run.lock               # continuation/resume flock and holder PID
    ├── final_report.md        # written when managed run reaches done
    ├── pareto.jsonl           # append-only ParetoRow history (one row per eval)
    ├── minibatches/<mb_id>.json
    ├── reports/<iteration>-<eval_id>-<candidate_id>.md  # training only
    └── traces/minibatches/<mb_id>/<iteration>-<eval_id>-<candidate_id>.jsonl  # training only
```

The held-out validation dataset lives outside this repository; no validation
case files, reports, or traces belong in this layout. Validation rows in
`pareto.jsonl` contain only aggregate scores and outcomes.

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

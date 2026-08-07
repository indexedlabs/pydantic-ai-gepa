"""Reflection lanes: worktree-backed candidate slots for managed runs.

Implements pydanticaigepa-spec-1do. A lane is a git worktree plus a per-lane
state slice under ``runs/<run_id>/lanes/<lane>/`` — the container for exactly
one in-flight candidate, from branch-off through reflection, background
evaluation, and verdict.

Key contracts:

- Lane processes locate the workspace explicitly — absolute ``--gepa-dir`` /
  run id / lane id from packet fields or flags, ``GEPA_DIR`` env fallback; no
  workspace path is derived from cwd (pydanticaigepa-dec-780).
- ``gepa lane continue`` auto-commits all worktree changes onto
  ``gepa/lane/<lane>/<iteration>`` before evaluating, so lane candidates are
  always clean commits (pydanticaigepa-dec-tlz).
- Lane evals run the repeated-evaluation acceptance policy as a detached
  background process that records its pid and heartbeat in lane state and
  emits a ``verdict`` event on resolution; budget checks are advisory per-eval
  and enforced at select (pydanticaigepa-dec-msy).
- Per-lane state is written only by that lane's own eval process or lifecycle
  verbs — never by other lanes. Verdicts travel via the event stream and lane
  state only; background evals never write shared run state.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import typer

from ..acceptance import compare_candidate_samples
from .candidates import GitCandidateError, git_candidate_state
from .eval import run_eval_once
from .events import EventDraft, LaneScan, LaneScanResult, emit, run_reaper_pass
from .layout import (
    current_gepa_dirname,
    gepa_dir,
    repo_root,
    run_dir,
)
from .runs import utc_now_iso

LaneStatus = Literal[
    "created",
    "paused_for_reflection",
    "leased",
    "evaluating",
    "awaiting_selection",
    "stalled",
]

PACKET_VERSION = 1
WORKTREES_DIRNAME = "worktrees"
LANE_BRANCH_PREFIX = "gepa/lane"
LANE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def lane_ids(count: int) -> list[str]:
    """Canonical lane ids for a run started with ``--lanes count``."""
    return [f"lane-{i}" for i in range(1, count + 1)]


def lane_branch(run_id: str, lane: str, iteration: int) -> str:
    """Run-scoped lane branch (dec-jh6): lane refs from different runs never
    collide, and an abandoned run can never poison a future one."""
    return f"{LANE_BRANCH_PREFIX}/{run_id}/{lane}/{iteration}"


# ----------------------------- workspace resolution ---------------------


def _resolve_workspace_root() -> Path:
    """Resolve the primary workspace root from the explicit gepa dir.

    Lane processes are invoked with an absolute ``--gepa-dir`` (or the
    ``GEPA_DIR`` env fallback), so the workspace root is the absolute
    workspace's parent — never a directory walked up from cwd (dec-780).
    """
    dirname = current_gepa_dirname()
    path = Path(dirname)
    if not path.is_absolute():
        raise typer.BadParameter(
            "Lane verbs require an explicit absolute workspace: pass "
            "`gepa --gepa-dir /abs/path/to/.gepa ...` or export GEPA_DIR. "
            "No workspace path is derived from the current directory."
        )
    return path.resolve().parent


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    # Reap first when the process is our own child: a terminated child stays
    # visible to kill(pid, 0) as a zombie until someone wait()s for it.
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    except ChildProcessError:
        pass  # not our child — fall through to the kill probe
    except OSError:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


# ----------------------------- per-lane state ---------------------------


@dataclass(frozen=True)
class LaneState:
    """Per-lane state slice persisted at runs/<run_id>/lanes/<lane>/state.json.

    Written only by the lane's own eval process or lifecycle verbs. Lanes have
    no terminal state — a lane is re-fanned at select or removed when the run
    completes.
    """

    lane: str
    status: LaneStatus
    iteration: int
    branch: str | None = None
    worktree_path: str | None = None
    packet_path: str | None = None
    lease_epoch: int = 0
    lease_expires_at: str | None = None
    # "dispatch" (orchestrator leased for a reflector) vs "handoff" (continue
    # spawned the detached eval): a handoff lease rejects parent-mode
    # continues; a dispatch lease is CONSUMED by the reflector's continue.
    lease_purpose: str | None = None
    candidate_sha: str | None = None
    eval_samples: tuple[float, ...] = ()
    verdict: str | None = None
    verdict_delta: float | None = None
    comparison_path: str | None = None
    eval_pid: int | None = None
    heartbeat_at: str | None = None
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "status": self.status,
            "iteration": self.iteration,
            "branch": self.branch,
            "worktree_path": self.worktree_path,
            "packet_path": self.packet_path,
            "lease_epoch": self.lease_epoch,
            "lease_expires_at": self.lease_expires_at,
            "lease_purpose": self.lease_purpose,
            "candidate_sha": self.candidate_sha,
            "eval_samples": list(self.eval_samples),
            "verdict": self.verdict,
            "verdict_delta": self.verdict_delta,
            "comparison_path": self.comparison_path,
            "eval_pid": self.eval_pid,
            "heartbeat_at": self.heartbeat_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> LaneState:
        return LaneState(
            lane=str(data["lane"]),
            status=str(data["status"]),  # type: ignore[arg-type]
            iteration=int(data["iteration"]),
            branch=data.get("branch"),
            worktree_path=data.get("worktree_path"),
            packet_path=data.get("packet_path"),
            lease_epoch=int(data.get("lease_epoch", 0)),
            lease_expires_at=data.get("lease_expires_at"),
            lease_purpose=data.get("lease_purpose"),
            candidate_sha=data.get("candidate_sha"),
            eval_samples=tuple(float(v) for v in data.get("eval_samples", [])),
            verdict=data.get("verdict"),
            verdict_delta=(
                float(data["verdict_delta"])
                if data.get("verdict_delta") is not None
                else None
            ),
            comparison_path=data.get("comparison_path"),
            eval_pid=(
                int(data["eval_pid"]) if data.get("eval_pid") is not None else None
            ),
            heartbeat_at=data.get("heartbeat_at"),
            updated_at=str(data.get("updated_at", "")),
        )

    def save(self, workspace_root: Path, run_id: str) -> Path:
        path = lane_state_path(workspace_root, run_id, self.lane)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, self.to_dict())
        return path


@contextmanager
def _lane_lock(workspace_root: Path, run_id: str, lane: str) -> Any:
    """Exclusive flock on the lane's state dir (dec-l95).

    Lease claims, the leased -> evaluating handoff, resets, and select's
    re-fan all hold this lock, so concurrent verbs serialize on the
    filesystem instead of racing read-then-write on state.json. The lock
    auto-releases if the holder dies.
    """
    path = lane_state_path(workspace_root, run_id, lane)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path.parent / ".lane.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def lanes_dir(workspace_root: Path, run_id: str) -> Path:
    return run_dir(run_id, workspace_root) / "lanes"


def lane_state_path(workspace_root: Path, run_id: str, lane: str) -> Path:
    return lanes_dir(workspace_root, run_id) / lane / "state.json"


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON via sibling tmpfile + os.replace — a torn state file can
    never wedge the verbs that read it."""
    import tempfile

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise


def load_lane_state(workspace_root: Path, run_id: str, lane: str) -> LaneState:
    path = lane_state_path(workspace_root, run_id, lane)
    if not path.exists():
        raise typer.BadParameter(
            f"No lane state at {path}. Was this run started with --lanes?"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"Lane state at {path} is corrupt ({exc.msg}); recover with "
            f"`gepa lane reset {lane}` after verifying no eval is running."
        ) from exc
    return LaneState.from_dict(data)


def load_all_lane_states(workspace_root: Path, run_id: str) -> list[LaneState]:
    base = lanes_dir(workspace_root, run_id)
    if not base.is_dir():
        return []
    states: list[LaneState] = []
    for entry in sorted(base.iterdir()):
        state_path = entry / "state.json"
        if state_path.exists():
            states.append(
                LaneState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
            )
    return states


# ----------------------------- worktrees --------------------------------


def worktrees_root(workspace_root: Path) -> Path:
    return workspace_root / WORKTREES_DIRNAME


def lane_worktree_path(workspace_root: Path, run_id: str, lane: str) -> Path:
    """Run-scoped worktree path (dec-jh6)."""
    return worktrees_root(workspace_root) / run_id / lane


def ensure_worktrees_ignored(workspace_root: Path) -> None:
    """Gitignore the worktrees dir via .git/info/exclude (never tracked files).

    Adding to .gitignore would dirty the primary tree; info/exclude is local
    and untracked, so `git ls-files --others --exclude-standard` (used by git
    candidate identity) skips lane worktrees automatically.
    """
    git_path = workspace_root / ".git"
    if git_path.is_file():
        # Linked worktree as primary: .git is a gitdir pointer file; resolve
        # the real git dir before writing info/exclude.
        content = git_path.read_text(encoding="utf-8").strip()
        if content.startswith("gitdir:"):
            git_path = Path(content.split(":", 1)[1].strip())
            if not git_path.is_absolute():
                git_path = (workspace_root / git_path).resolve()
    info_dir = git_path / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    exclude = info_dir / "exclude"
    entry = f"/{WORKTREES_DIRNAME}/"
    existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    if entry not in existing.splitlines():
        with exclude.open("a", encoding="utf-8") as fh:
            fh.write(f"{entry}\n")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def create_lane_worktree(
    workspace_root: Path, run_id: str, lane: str, iteration: int, base_sha: str
) -> tuple[Path, str]:
    """Create the lane worktree + branch cut from ``base_sha`` (the run's best)."""
    branch = lane_branch(run_id, lane, iteration)
    path = lane_worktree_path(workspace_root, run_id, lane)
    path.parent.mkdir(parents=True, exist_ok=True)
    _git(workspace_root, "worktree", "add", str(path), "-b", branch, base_sha)
    return path, branch


# ----------------------------- packet -----------------------------------


def _journal_tail(workspace_root: Path, limit: int) -> list[dict[str, Any]]:
    """Bounded journal tail for the reflection packet (workspace-explicit)."""
    from .layout import journal_path

    path = journal_path(workspace_root)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows[-limit:] if limit > 0 else rows


def _collect_metric_side_info(trace_paths: list[str]) -> dict[str, Any]:
    """Collect per-case metric side info from baseline trace files."""
    side_info: dict[str, Any] = {}
    for raw in trace_paths:
        path = Path(raw)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            info = record.get("metric_side_info")
            if info and record.get("case_id"):
                side_info[str(record["case_id"])] = info
    return side_info


def write_packet(
    workspace_root: Path,
    run_state: Any,  # RunState — imported lazily to avoid a module cycle
    lane: str,
    iteration: int,
    worktree_path: Path,
    branch: str,
) -> Path:
    """Write the versioned reflection packet for a paused lane.

    A subagent needs only the packet path to work: baseline samples with
    report/trace paths, metric side info, a bounded journal tail, the worktree
    path, and the exact `gepa lane continue` invocation.
    """
    gepa_abs = str(gepa_dir(workspace_root).resolve())
    invocation = (
        f"gepa --gepa-dir {shlex.quote(gepa_abs)} "
        f"lane continue {shlex.quote(lane)} --run-id {shlex.quote(run_state.run_id)}"
    )
    packet = {
        "packet_version": PACKET_VERSION,
        "run_id": run_state.run_id,
        "lane": lane,
        "iteration": iteration,
        "worktree_path": str(worktree_path),
        "branch": branch,
        "baseline": {
            "candidate_id": run_state.reflection_baseline_candidate_id,
            "commit_sha": run_state.reflection_baseline_commit_sha,
            "mean_score": run_state.reflection_baseline_mean_score,
            "samples": list(run_state.reflection_baseline_samples),
            "minibatch_id": run_state.reflection_minibatch_id,
            "report_paths": list(run_state.reflection_baseline_report_paths),
            "trace_paths": list(run_state.reflection_baseline_trace_paths),
        },
        "metric_side_info": _collect_metric_side_info(
            list(run_state.reflection_baseline_trace_paths)
        ),
        "journal_tail": _journal_tail(workspace_root, run_state.journal_tail_lines),
        "continue_invocation": invocation,
    }
    path = lanes_dir(workspace_root, run_state.run_id) / lane / "packet.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet, indent=2, default=str), encoding="utf-8")
    return path


# ----------------------------- lane eval --------------------------------


def _auto_commit_worktree(worktree: Path, branch: str, lane: str) -> str:
    """Commit all worktree changes onto the lane branch; return the HEAD sha.

    Lane candidates are always clean commits because continue auto-commits
    (dec-tlz).
    """
    current_branch = _git(worktree, "rev-parse", "--abbrev-ref", "HEAD")
    if current_branch != branch:
        # Check BEFORE committing: a commit on a rogue branch would silently
        # lose the candidate from the lane's lineage.
        raise typer.BadParameter(
            f"Lane worktree {worktree} is on branch {current_branch!r}, "
            f"expected {branch!r}; run `gepa lane reset {lane}` to recover."
        )
    _git(worktree, "add", "-A")
    status = _git(worktree, "status", "--porcelain")
    if status:
        _git(
            worktree,
            "-c",
            "user.name=gepa-lane",
            "-c",
            "user.email=gepa-lane@localhost",
            "commit",
            "-m",
            f"gepa lane continue: {lane} candidate",
        )
    return _git(worktree, "rev-parse", "HEAD")


def _write_comparison(
    workspace_root: Path,
    run_id: str,
    lane: str,
    iteration: int,
    comparison: dict[str, Any],
) -> Path:
    path = lanes_dir(workspace_root, run_id) / lane / f"comparison-{iteration:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    return path


def _touch_heartbeat(
    workspace_root: Path, run_id: str, state: LaneState, pid: int
) -> LaneState:
    fresh = LaneState(
        **{
            **state.to_dict(),
            "heartbeat_at": utc_now_iso(),
            "eval_pid": pid,
            "updated_at": utc_now_iso(),
        }
    )
    fresh.save(workspace_root, run_id)
    return fresh


def _run_lane_eval_loop(
    *,
    workspace_root: Path,
    run_state: Any,  # RunState
    lane_state: LaneState,
) -> LaneState:
    """Run the escalating acceptance eval for the lane's committed candidate.

    Mirrors the single-path ``_evaluate_reflected_candidate`` escalation
    (repeated evals against the frozen reflection minibatch, early stop once
    the comparison resolves), writing progress into lane state and emitting a
    ``verdict`` event on resolution. Never writes shared run state.
    """
    run_id = run_state.run_id
    lane = lane_state.lane
    if run_state.reflection_minibatch_id is None:
        raise typer.BadParameter(
            f"Run {run_id} has no reflection minibatch; cannot evaluate lane."
        )
    if not run_state.reflection_baseline_samples:
        raise typer.BadParameter(
            f"Run {run_id} is missing reflection baseline samples."
        )

    worktree = Path(str(lane_state.worktree_path))
    branch = str(lane_state.branch)
    commit_sha = _auto_commit_worktree(worktree, branch, lane)

    try:
        git_state = git_candidate_state(worktree)
    except GitCandidateError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if git_state.dirty:
        raise typer.BadParameter(
            f"Lane worktree {worktree} is still dirty after auto-commit; "
            "refusing to evaluate an unstable candidate."
        )

    pid = os.getpid()
    lane_state = LaneState(
        **{
            **lane_state.to_dict(),
            "status": "evaluating",
            "candidate_sha": commit_sha,
            "eval_samples": (),
            "verdict": None,
            "verdict_delta": None,
            "comparison_path": None,
            "lease_expires_at": None,  # the handoff lease is consumed
            "lease_purpose": None,
        }
    )
    lane_state = _touch_heartbeat(workspace_root, run_id, lane_state, pid)

    baseline_samples = tuple(float(v) for v in run_state.reflection_baseline_samples)
    max_candidate_samples = len(baseline_samples)
    initial_samples = min(run_state.acceptance_repetitions, max_candidate_samples)

    # The candidate tree is the cwd for evaluation: `evaluate` callables and
    # user metric code routinely read files by relative path, and those reads
    # must observe the lane worktree, not the primary checkout (detached
    # children are spawned with cwd=worktree; foreground callers are switched
    # here so both paths evaluate the same tree).
    previous_cwd = Path.cwd()
    os.chdir(worktree)
    samples: list[float] = []
    outcomes: list[Any] = []
    comparison_result = None
    try:
        for _ in range(max_candidate_samples):
            outcome = run_eval_once(
                candidate_file=None,
                minibatch_id=run_state.reflection_minibatch_id,
                size=run_state.size,
                seed=run_state.seed,
                epoch=run_state.next_epoch,
                run_id=run_id,
                concurrency=run_state.concurrency,
                max_iterations=run_state.max_iterations,
                threshold=run_state.threshold,
                capture_traces=True,
                candidate_source="git",
                lane=lane,
                candidate_root=worktree,
                workspace_root=workspace_root,
            )
            outcomes.append(outcome)
            current_id = str(outcome.summary["candidate_id"])
            if current_id != git_state.candidate_id:
                raise typer.BadParameter(
                    "The lane candidate changed mid-evaluation "
                    f"({git_state.candidate_id} -> {current_id}); refusing to "
                    "compare mixed candidates."
                )
            samples.append(float(outcome.summary["mean_score"]))
            lane_state = LaneState(
                **{**lane_state.to_dict(), "eval_samples": tuple(samples)}
            )
            lane_state = _touch_heartbeat(workspace_root, run_id, lane_state, pid)

            if len(samples) < initial_samples:
                continue
            comparison_result = compare_candidate_samples(
                baseline_samples[: len(samples)],
                tuple(samples),
                confidence=run_state.acceptance_confidence,
                min_delta=run_state.acceptance_min_delta,
            )
            if comparison_result.verdict != "inconclusive":
                break
    finally:
        os.chdir(previous_cwd)

    assert comparison_result is not None
    comparison = {
        "run_id": run_id,
        "lane": lane,
        "iteration": lane_state.iteration,
        "minibatch_id": run_state.reflection_minibatch_id,
        "baseline_candidate_id": run_state.reflection_baseline_candidate_id,
        "baseline_commit_sha": run_state.reflection_baseline_commit_sha,
        "candidate_id": git_state.candidate_id,
        "candidate_commit_sha": commit_sha,
        "candidate_report_paths": [
            outcome.summary["report_path"] for outcome in outcomes
        ],
        "candidate_trace_paths": [
            outcome.summary["trace_path"]
            for outcome in outcomes
            if outcome.summary.get("trace_path")
        ],
        **comparison_result.to_dict(),
    }
    comparison_path = _write_comparison(
        workspace_root, run_id, lane, lane_state.iteration, comparison
    )
    # State lands before the event: a crash between them must leave the
    # verdict in lane state (select reads verdicts from state, not the bus),
    # with the event redeliverable rather than a verdict event pointing at a
    # lane that looks mid-eval.
    lane_state = LaneState(
        **{
            **lane_state.to_dict(),
            "status": "awaiting_selection",
            "verdict": comparison_result.verdict,
            "verdict_delta": comparison_result.delta,
            "comparison_path": str(comparison_path),
            "eval_pid": None,
            "updated_at": utc_now_iso(),
        }
    )
    lane_state.save(workspace_root, run_id)
    verdict_id = emit(
        run_id,
        lane,
        EventDraft(
            type="verdict",
            lane=lane,
            payload={
                "verdict": comparison_result.verdict,
                "delta": comparison_result.delta,
                "comparison_path": str(comparison_path),
            },
        ),
        root=workspace_root,
    )
    typer.echo(
        f"Lane {lane} verdict: {comparison_result.verdict} "
        f"(delta={comparison_result.delta:+.4f}); event {verdict_id}."
    )
    return lane_state


# ----------------------------- verbs ------------------------------------

app = typer.Typer(
    no_args_is_help=True,
    help="Drive reflection lanes: lease, continue (background eval), reset.",
)


def _load_run_state(workspace_root: Path, run_id: str) -> Any:
    from .run import RunState  # local import: run.py imports lanes lazily

    path = run_dir(run_id, workspace_root) / "state.json"
    if not path.exists():
        typer.echo(
            f"No managed run state at {path}. Start one with `gepa run start`.",
            err=True,
        )
        raise typer.Exit(code=1)
    return RunState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _resolve_lane_run(run_id: str | None) -> tuple[Path, Any]:
    """Resolve (workspace_root, RunState) explicitly; reject non-lane runs."""
    workspace_root = _resolve_workspace_root()
    if run_id is None:
        from .layout import latest_run_id

        run_id = latest_run_id(workspace_root)
        if run_id is None:
            typer.echo("No runs found. Start one with `gepa run start`.", err=True)
            raise typer.Exit(code=1)
    run_state = _load_run_state(workspace_root, run_id)
    if run_state.status == "done":
        typer.echo(
            f"Run {run_state.run_id} is done; lane verbs no longer apply "
            "(a lane is re-fanned at select or removed when the run completes).",
            err=True,
        )
        raise typer.Exit(code=1)
    if run_state.lanes < 1:
        typer.echo(
            f"Run {run_state.run_id} is not a lane run (started without --lanes); "
            "lane verbs do not apply.",
            err=True,
        )
        raise typer.Exit(code=1)
    return workspace_root, run_state


def _validate_lane_id(lane: str) -> str:
    if not LANE_ID_RE.match(lane):
        raise typer.BadParameter(f"Invalid lane id {lane!r}.")
    return lane


@app.command("lease")
def lane_lease(
    lane: str = typer.Argument(..., help="Lane id (e.g. lane-1)."),
    run_id: str | None = typer.Option(None, "--run-id", help="Defaults to latest run."),
) -> None:
    """Record a dispatch lease on a paused lane.

    The orchestrator leases a lane before dispatching a reflector subagent.
    A leased lane rejects re-dispatch and a second concurrent
    `gepa lane continue` until the lease is released or times out.
    """
    _validate_lane_id(lane)
    workspace_root, run_state = _resolve_lane_run(run_id)
    with _lane_lock(workspace_root, run_state.run_id, lane):
        state = load_lane_state(workspace_root, run_state.run_id, lane)
        now = datetime.now(timezone.utc)
        if state.status == "leased" and state.lease_expires_at:
            if _parse_iso(state.lease_expires_at) > now:
                typer.echo(
                    f"Lane {lane} is already leased until {state.lease_expires_at} "
                    f"(lease epoch {state.lease_epoch}); refusing re-dispatch.",
                    err=True,
                )
                raise typer.Exit(code=1)
        if state.status == "evaluating":
            typer.echo(
                f"Lane {lane} is evaluating (pid {state.eval_pid}); it cannot "
                "be leased for dispatch.",
                err=True,
            )
            raise typer.Exit(code=1)
        if state.status not in {"paused_for_reflection", "leased", "stalled"}:
            typer.echo(
                f"Lane {lane} is {state.status}; only paused_for_reflection "
                "lanes can be leased for dispatch.",
                err=True,
            )
            raise typer.Exit(code=1)
        expires = now.timestamp() + run_state.reflection_lease_secs
        expires_at = datetime.fromtimestamp(expires, timezone.utc).isoformat()
        state = LaneState(
            **{
                **state.to_dict(),
                "status": "leased",
                "lease_epoch": state.lease_epoch + 1,
                "lease_expires_at": expires_at,
                "lease_purpose": "dispatch",
                "updated_at": utc_now_iso(),
            }
        )
        state.save(workspace_root, run_state.run_id)
    typer.echo(
        f"Lane {lane} leased (epoch {state.lease_epoch}, expires {expires_at}). "
        f"Dispatch a reflector with the packet at {state.packet_path}."
    )


@app.command("continue")
def lane_continue(
    lane: str = typer.Argument(..., help="Lane id (e.g. lane-1)."),
    run_id: str | None = typer.Option(None, "--run-id", help="Defaults to latest run."),
    foreground: bool = typer.Option(
        False,
        "--foreground",
        help="Run the acceptance eval in-process instead of detaching (used by the detached wrapper and by tests).",
    ),
    handoff_lease_epoch: int | None = typer.Option(
        None,
        "--handoff-lease-epoch",
        hidden=True,
    ),
) -> None:
    """Auto-commit the lane worktree and evaluate the candidate in the background.

    Resolves the workspace explicitly from --gepa-dir / GEPA_DIR (never cwd),
    commits all worktree changes onto the lane branch, then runs the
    repeated-evaluation acceptance policy as a detached process that streams
    progress + heartbeat into lane state and emits a `verdict` event on
    resolution.

    Lease discipline (dec-l95): continue requires a lease — an unexpired one
    from `gepa lane lease`, or it auto-claims one for a paused/stalled lane.
    A lane already evaluating rejects a second continue; a leased lane rejects
    any continue that does not consume its lease. The claim and the handoff
    are serialized by an flock on the lane state dir, and the handoff lease
    expires after --eval-stall-timeout-secs so a child that dies before its
    first heartbeat is reaped. The detached child carries the handoff lease
    epoch and must claim that exact lease under the lock; a late child cannot
    revive a lane after the reaper/reset path has replaced it.
    """
    _validate_lane_id(lane)
    workspace_root, run_state = _resolve_lane_run(run_id)
    now = datetime.now(timezone.utc)

    with _lane_lock(workspace_root, run_state.run_id, lane):
        state = load_lane_state(workspace_root, run_state.run_id, lane)
        claimed_handoff = False

        if foreground and handoff_lease_epoch is not None:
            # A detached child is fenced by the epoch its parent created.  It
            # must claim that exact handoff while holding the lane lock: after
            # expiry/reset/re-dispatch an old child is not allowed to revive a
            # lane or overwrite the replacement eval's state.
            if (
                state.status != "leased"
                or state.lease_purpose != "handoff"
                or state.lease_epoch != handoff_lease_epoch
                or state.lease_expires_at is None
                or _parse_iso(state.lease_expires_at) <= now
            ):
                typer.echo(
                    f"Lane {lane}'s handoff lease epoch {handoff_lease_epoch} "
                    "is no longer current; refusing stale detached eval.",
                    err=True,
                )
                raise typer.Exit(code=1)
            state = LaneState(
                **{
                    **state.to_dict(),
                    "status": "evaluating",
                    "eval_pid": os.getpid(),
                    "heartbeat_at": utc_now_iso(),
                    "lease_expires_at": None,
                    "lease_purpose": None,
                    "updated_at": utc_now_iso(),
                }
            )
            state.save(workspace_root, run_state.run_id)
            claimed_handoff = True
        if state.status == "evaluating":
            if claimed_handoff:
                # The fenced child has already claimed the handoff above.
                pass
            else:
                typer.echo(
                    f"Lane {lane} is already evaluating (pid {state.eval_pid}); a "
                    "second concurrent `gepa lane continue` is rejected. If the "
                    f"eval is dead, run `gepa lane reset {lane}`.",
                    err=True,
                )
                raise typer.Exit(code=1)
        if state.status == "leased" and state.lease_purpose == "handoff":
            # A detached eval child was already spawned for this lease. Only
            # that child, carrying the matching handoff epoch, may consume it.
            typer.echo(
                f"Lane {lane} is leased (epoch {state.lease_epoch}) with "
                "an eval handoff in flight; a second `gepa lane continue` "
                "is rejected until the child starts (evaluating), the lease "
                "expires, or you run `gepa lane reset`.",
                err=True,
            )
            raise typer.Exit(code=1)
        # A dispatch lease (from `gepa lane lease`) is CONSUMED by this
        # continue — the reflector's terminal act is the whole point of the
        # lease. The flock + the evaluating check above serialize any true
        # double-continue race.
        allowed_states = {"paused_for_reflection", "leased", "stalled"}
        if claimed_handoff:
            allowed_states.add("evaluating")
        if state.status not in allowed_states:
            typer.echo(
                f"Lane {lane} is {state.status}; `gepa lane continue` expects a "
                "paused_for_reflection (or leased) lane.",
                err=True,
            )
            raise typer.Exit(code=1)
        if not state.worktree_path or not state.branch:
            typer.echo(
                f"Lane {lane} has no worktree/branch recorded; run "
                f"`gepa lane reset {lane}` to re-fan it.",
                err=True,
            )
            raise typer.Exit(code=1)

        if not foreground:
            # Handoff lease with a REAL expiry: if the detached child dies
            # before its first heartbeat, the reaper can stall the lane.
            handoff_expiry = now.timestamp() + run_state.eval_stall_timeout_secs
            state = LaneState(
                **{
                    **state.to_dict(),
                    "status": "leased",
                    "lease_epoch": state.lease_epoch + 1,
                    "lease_expires_at": datetime.fromtimestamp(
                        handoff_expiry, timezone.utc
                    ).isoformat(),
                    "lease_purpose": "handoff",
                    "updated_at": utc_now_iso(),
                }
            )
            state.save(workspace_root, run_state.run_id)

    if foreground:
        _run_lane_eval_loop(
            workspace_root=workspace_root, run_state=run_state, lane_state=state
        )
        return

    # Detached background eval: the child re-invokes this verb with
    # --foreground, consumes the handoff lease, records its own pid, and
    # refreshes the lane heartbeat.
    gepa_abs = str(gepa_dir(workspace_root).resolve())
    lane_dir = lanes_dir(workspace_root, run_state.run_id) / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    log_path = lane_dir / "eval.log"
    argv = [
        sys.executable,
        "-c",
        "from pydantic_ai_gepa.cli import app; app()",
        "--gepa-dir",
        gepa_abs,
        "lane",
        "continue",
        lane,
        "--run-id",
        run_state.run_id,
        "--foreground",
        "--handoff-lease-epoch",
        str(state.lease_epoch),
    ]
    worktree_path = str(state.worktree_path)
    try:
        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(  # noqa: S603
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=worktree_path,
            )
    except OSError as exc:
        # Spawn failed: release the handoff lease so the lane stays dispatchable.
        with _lane_lock(workspace_root, run_state.run_id, lane):
            current = load_lane_state(workspace_root, run_state.run_id, lane)
            if current.status == "leased":
                LaneState(
                    **{
                        **current.to_dict(),
                        "status": "paused_for_reflection",
                        "lease_expires_at": None,
                        "updated_at": utc_now_iso(),
                    }
                ).save(workspace_root, run_state.run_id)
        typer.echo(f"Failed to spawn lane eval: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Lane {lane} eval detached (pid {process.pid}); log: {log_path}. "
        f"Watch `gepa next` for the verdict event."
    )


@app.command("reset")
def lane_reset(
    lane: str = typer.Argument(..., help="Lane id (e.g. lane-1)."),
    run_id: str | None = typer.Option(None, "--run-id", help="Defaults to latest run."),
) -> None:
    """Recover a stalled lane: terminate a dead/alive eval, reset to paused.

    A live recorded eval pid is terminated first (SIGTERM); uncommitted
    worktree content is never auto-deleted. Lanes holding a resolved verdict
    (awaiting_selection) cannot be reset — only `gepa run select` consumes
    them, so a completed candidate is never destroyed unrecorded.
    """
    _validate_lane_id(lane)
    workspace_root, run_state = _resolve_lane_run(run_id)
    with _lane_lock(workspace_root, run_state.run_id, lane):
        state = load_lane_state(workspace_root, run_state.run_id, lane)
        if state.status == "awaiting_selection":
            typer.echo(
                f"Lane {lane} is awaiting_selection with verdict "
                f"{state.verdict!r}; only `gepa run select` consumes it. "
                "Refusing to reset (its outcome would be lost unrecorded).",
                err=True,
            )
            raise typer.Exit(code=1)
        if state.eval_pid is not None and _pid_alive(state.eval_pid):
            typer.echo(
                f"Terminating lane {lane} eval process (pid {state.eval_pid}).",
                err=True,
            )
            os.kill(state.eval_pid, signal.SIGTERM)
            deadline = time.monotonic() + 5.0
            while _pid_alive(state.eval_pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if _pid_alive(state.eval_pid):
                typer.echo(
                    f"Lane {lane} eval pid {state.eval_pid} did not exit after "
                    "SIGTERM; refusing to reset a live lane.",
                    err=True,
                )
                raise typer.Exit(code=1)
        worktree = Path(state.worktree_path) if state.worktree_path else None
        if worktree is not None and worktree.exists():
            dirty = _git(worktree, "status", "--porcelain")
            if dirty:
                typer.echo(
                    f"Note: lane worktree {worktree} has uncommitted content; "
                    "it is preserved (never auto-deleted).",
                    err=True,
                )
        state = LaneState(
            **{
                **state.to_dict(),
                "status": "paused_for_reflection",
                "lease_epoch": state.lease_epoch + 1,
                "lease_expires_at": None,
                "lease_purpose": None,
                "eval_pid": None,
                "heartbeat_at": None,
                "verdict": None,
                "verdict_delta": None,
                "eval_samples": (),
                "updated_at": utc_now_iso(),
            }
        )
        state.save(workspace_root, run_state.run_id)
    typer.echo(
        f"Lane {lane} reset to paused_for_reflection (lease epoch "
        f"{state.lease_epoch}). Re-dispatch with the packet at "
        f"{state.packet_path}."
    )


# ----------------------------- reaper scanner ---------------------------


def scan_lane_states(workspace_root: Path, run_id: str, run_state: Any) -> LaneScan:
    """Scan lane leases, heartbeat freshness, and recorded pids (dec-pm3).

    This is the real scanner behind the event bus's lazy reaper: a lane is
    stalled when its eval pid is dead, its heartbeat is stale, or its
    reflection lease expired. ``selection_due`` signals when every lane is
    resolved (awaiting_selection) or the straggler timeout has elapsed with
    at least one resolved lane.
    """
    now = datetime.now(timezone.utc)
    results: list[LaneScanResult] = []
    resolved: list[str] = []
    stragglers: list[str] = []
    for state in load_all_lane_states(workspace_root, run_id):
        stalled_reason: str | None = None
        if state.status == "evaluating":
            pid_dead = state.eval_pid is not None and not _pid_alive(state.eval_pid)
            heartbeat_stale = False
            if state.heartbeat_at:
                age = now - _parse_iso(state.heartbeat_at)
                heartbeat_stale = (
                    age.total_seconds() > run_state.eval_stall_timeout_secs
                )
            else:
                heartbeat_stale = True
            # spec-1do: eval death is stale heartbeat PLUS dead pid — a stale
            # heartbeat with a live pid is a slow eval, not a dead one (the
            # documented response to lane_stalled is `lane reset`, which
            # SIGTERMs the pid; false positives would kill healthy evals).
            if pid_dead:
                stalled_reason = f"eval process pid {state.eval_pid} is dead"
                if heartbeat_stale:
                    stalled_reason += " with a stale heartbeat"
        elif state.status == "leased":
            if state.lease_expires_at and _parse_iso(state.lease_expires_at) <= now:
                stalled_reason = "reflection lease expired"
        if stalled_reason is not None:
            results.append(
                LaneScanResult(
                    lane=state.lane,
                    lease_epoch=state.lease_epoch,
                    stalled_reason=stalled_reason,
                )
            )
            stragglers.append(state.lane)
        elif state.status == "awaiting_selection":
            resolved.append(state.lane)
        else:
            stragglers.append(state.lane)

    from .events import SelectionDueSignal

    selection_due: SelectionDueSignal | None = None
    if resolved:
        # The straggler clock starts at fan-out/re-fan, not from
        # run_state.updated_at — unrelated verb saves refresh that field and
        # would otherwise starve the timeout indefinitely (spec-er3).
        started_raw = run_state.iteration_started_at or run_state.updated_at
        started = _parse_iso(started_raw)
        elapsed = (now - started).total_seconds()
        all_resolved = not stragglers
        if all_resolved or elapsed > run_state.straggler_timeout_secs:
            selection_due = SelectionDueSignal(
                iteration=run_state.iterations,
                resolved_lanes=tuple(resolved),
                straggler_lanes=tuple(stragglers),
            )
    return LaneScan(lanes=tuple(results), selection_due=selection_due)


def reaper_pass_for_run(workspace_root: Path, run_state: Any) -> list[str]:
    """Run the lazy reaper with the real lane scanner for a lane run.

    Stalled lanes are transitioned in state (dec-l95), not just reported on
    the bus: the lane board shows `stalled`, and the status machine rows
    leased/evaluating → stalled become reachable outside select.
    """
    scan = scan_lane_states(workspace_root, run_state.run_id, run_state)
    emitted = run_reaper_pass(run_state.run_id, scan, root=workspace_root)
    for lane_scan in scan.lanes:
        if lane_scan.stalled_reason is None:
            continue
        with _lane_lock(workspace_root, run_state.run_id, lane_scan.lane):
            state = load_lane_state(workspace_root, run_state.run_id, lane_scan.lane)
            if (
                state.status in {"leased", "evaluating"}
                and state.lease_epoch == lane_scan.lease_epoch
            ):
                LaneState(
                    **{
                        **state.to_dict(),
                        "status": "stalled",
                        "eval_pid": None,
                        "lease_expires_at": None,
                        "lease_purpose": None,
                        "updated_at": utc_now_iso(),
                    }
                ).save(workspace_root, run_state.run_id)
    return emitted


# ----------------------------- fan-out (run start) ----------------------


def fan_out_lanes(run_state: Any, workspace_root: Path) -> None:
    """Create worktrees, write packets, and emit lane_ready for every lane.

    Called by `gepa run start --lanes N` after the shared baseline pause: all
    lanes branch from the frozen baseline commit (the run's best), so every
    lane in an iteration shares one branch point and one frozen baseline.

    Fan-out failure is rolled back — worktrees, branches, lane states, and
    lane_ready events created so far are removed — so a crash mid-fan-out
    never leaves a zombie run (dec-jh6).
    """
    base_sha = run_state.reflection_baseline_commit_sha
    if not base_sha:
        typer.echo(
            "Cannot fan out lanes: the run has no reflection baseline commit.",
            err=True,
        )
        raise typer.Exit(code=1)
    created: list[tuple[Path, str, str]] = []  # (worktree path, branch, lane)
    try:
        for lane in lane_ids(run_state.lanes):
            path, branch = create_lane_worktree(
                workspace_root, run_state.run_id, lane, run_state.iterations, base_sha
            )
            created.append((path, branch, lane))
            packet_path = write_packet(
                workspace_root, run_state, lane, run_state.iterations, path, branch
            )
            LaneState(
                lane=lane,
                status="paused_for_reflection",
                iteration=run_state.iterations,
                branch=branch,
                worktree_path=str(path),
                packet_path=str(packet_path),
                updated_at=utc_now_iso(),
            ).save(workspace_root, run_state.run_id)
            emit(
                run_state.run_id,
                "run",
                EventDraft(
                    type="lane_ready",
                    lane=lane,
                    payload={
                        "packet_path": str(packet_path),
                        "worktree_path": str(path),
                    },
                ),
                root=workspace_root,
            )
            typer.echo(f"Lane {lane} ready: worktree {path}, packet {packet_path}.")
    except Exception:
        for path, branch, lane in reversed(created):
            try:
                _git(workspace_root, "worktree", "remove", "--force", str(path))
            except Exception:
                pass
            try:
                _git(workspace_root, "branch", "-D", branch)
            except Exception:
                pass
            lane_dir = lanes_dir(workspace_root, run_state.run_id) / lane
            if lane_dir.exists():
                import shutil

                shutil.rmtree(lane_dir, ignore_errors=True)
        typer.echo(
            "Lane fan-out failed; rolled back created worktrees/branches.",
            err=True,
        )
        raise


def scan_run_lanes(run_id: str, root: Path | None = None) -> LaneScan | None:
    """Real scanner behind events.scan_lanes: None when this is not a lane run."""
    workspace_root = (root or repo_root()).resolve()
    state_path = run_dir(run_id, workspace_root) / "state.json"
    if not state_path.exists():
        return None
    from .run import RunState  # lazy: run.py imports lanes lazily

    run_state = RunState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
    if run_state.lanes < 1 or run_state.status == "done":
        return None
    return scan_lane_states(workspace_root, run_id, run_state)

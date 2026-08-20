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
from ..vector_acceptance import (
    VectorComparison,
    VectorComparisonRequest,
    VectorRecord,
    VectorRecordStore,
    compare_vectors,
    resolve_vector_comparator,
)
from ..evaluation_health import (
    EvaluationInfrastructureFailure,
    evaluation_infrastructure_failures,
)
from .candidates import GitCandidateError, git_candidate_state
from .eval import run_eval_once
from .events import EventDraft, LaneScan, LaneScanResult, emit, run_reaper_pass
from .layout import (
    GepaConfig,
    candidate_project_root,
    config_path,
    current_gepa_dirname,
    gepa_dir,
    git_root,
    project_prefix,
    project_root_for_workspace,
    repo_root,
    run_dir,
    vector_records_path,
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

PACKET_VERSION = 2
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
    return project_root_for_workspace(path)


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
    candidate_project_path: str | None = None
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
    review_failures: int = 0
    review_findings: tuple[dict[str, Any], ...] = ()
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "status": self.status,
            "iteration": self.iteration,
            "branch": self.branch,
            "worktree_path": self.worktree_path,
            "candidate_project_path": self.candidate_project_path,
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
            "review_failures": self.review_failures,
            "review_findings": list(self.review_findings),
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
            candidate_project_path=data.get("candidate_project_path"),
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
            review_failures=int(data.get("review_failures", 0)),
            review_findings=tuple(
                item
                for item in data.get("review_findings", [])
                if isinstance(item, dict)
            ),
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
    return git_root(workspace_root) / WORKTREES_DIRNAME


def lane_worktree_path(workspace_root: Path, run_id: str, lane: str) -> Path:
    """Run-scoped worktree path (dec-jh6)."""
    return worktrees_root(workspace_root) / run_id / lane


def ensure_worktrees_ignored(workspace_root: Path) -> None:
    """Gitignore the worktrees dir via the common Git dir's info/exclude.

    Adding to .gitignore would dirty the primary tree; info/exclude is local
    and untracked, so `git ls-files --others --exclude-standard` (used by git
    candidate identity) skips lane worktrees automatically.
    """
    repository = git_root(workspace_root)
    common_raw = _git(repository, "rev-parse", "--git-common-dir")
    common_dir = Path(common_raw)
    if not common_dir.is_absolute():
        common_dir = (repository / common_dir).resolve()
    info_dir = common_dir / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    exclude = info_dir / "exclude"
    entry = f"/{WORKTREES_DIRNAME}/"
    existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    if entry not in existing.splitlines():
        with exclude.open("a", encoding="utf-8") as fh:
            if existing and not existing.endswith("\n"):
                fh.write("\n")
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
    _git(git_root(workspace_root), "worktree", "add", str(path), "-b", branch, base_sha)
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


def _collect_metric_side_info_by_rep(trace_paths: list[str]) -> list[dict[str, Any]]:
    """Keep repetition boundaries intact; scalar packets retain the legacy view."""
    result: list[dict[str, Any]] = []
    for raw in trace_paths:
        result.append({"trace_path": raw, "cases": _collect_metric_side_info([raw])})
    return result


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
    repository = git_root(workspace_root)
    candidate_project = candidate_project_root(workspace_root, worktree_path)
    gepa_abs = str(gepa_dir(workspace_root).resolve())
    continue_argv = _lane_continue_argv(
        gepa_abs=gepa_abs,
        lane=lane,
        run_id=run_state.run_id,
    )
    invocation = (
        f"cd {shlex.quote(str(candidate_project))} && {shlex.join(continue_argv)}"
    )
    cfg = GepaConfig.load(config_path(workspace_root))
    metric_info: Any = (
        _collect_metric_side_info_by_rep(
            list(run_state.reflection_baseline_trace_paths)
        )
        if cfg.acceptance.mode == "vector"
        else _collect_metric_side_info(list(run_state.reflection_baseline_trace_paths))
    )
    packet = {
        "packet_version": PACKET_VERSION,
        "run_id": run_state.run_id,
        "lane": lane,
        "iteration": iteration,
        "worktree_path": str(worktree_path),
        "git_root": str(repository),
        "project_root": str(workspace_root.resolve()),
        "project_prefix": project_prefix(workspace_root, repository).as_posix(),
        "candidate_project_root": str(candidate_project),
        "gepa_workspace": gepa_abs,
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
        "metric_side_info": metric_info,
        "journal_tail": _journal_tail(workspace_root, run_state.journal_tail_lines),
        "continue_cwd": str(candidate_project),
        "continue_argv": continue_argv,
        "continue_invocation": invocation,
    }
    path = lanes_dir(workspace_root, run_state.run_id) / lane / "packet.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet, indent=2, default=str), encoding="utf-8")
    return path


def _lane_continue_argv(
    *,
    gepa_abs: str,
    lane: str,
    run_id: str,
    foreground: bool = False,
    handoff_lease_epoch: int | None = None,
) -> list[str]:
    """Build a launcher independent of the caller's PATH or active venv."""

    argv = [
        sys.executable,
        "-I",
        "-c",
        "from pydantic_ai_gepa.cli import app; app()",
        "--gepa-dir",
        gepa_abs,
        "lane",
        "continue",
        lane,
        "--run-id",
        run_id,
    ]
    if foreground:
        argv.append("--foreground")
    if handoff_lease_epoch is not None:
        argv.extend(["--handoff-lease-epoch", str(handoff_lease_epoch)])
    return argv


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


def _candidate_gate(
    *,
    workspace_root: Path,
    run_state: Any,
    state: LaneState,
) -> LaneState | None:
    """Reject out-of-scope diffs and run the optional pre-evaluation reviewer."""
    cfg = GepaConfig.load(config_path(workspace_root))
    if cfg.acceptance.mode != "vector":
        return None
    worktree = Path(str(state.worktree_path))
    baseline = str(run_state.reflection_baseline_commit_sha)
    changed = _candidate_changed_paths(worktree, baseline)
    allowed = set(cfg.acceptance.component_files) | set(cfg.acceptance.meta_files)
    unexpected = sorted(item for item in changed if item and item not in allowed)
    if unexpected:
        findings = (
            {
                "component": None,
                "excerpt": ", ".join(unexpected),
                "explanation": "Candidate changed files outside acceptance.component_files.",
                "severity": "error",
            },
        )
        rejected = LaneState(
            **{
                **state.to_dict(),
                "status": "stalled",
                "review_failures": state.review_failures + 1,
                "review_findings": findings,
                "lease_expires_at": None,
                "lease_purpose": None,
                "updated_at": utc_now_iso(),
            }
        )
        rejected.save(workspace_root, run_state.run_id)
        return rejected
    if cfg.acceptance.reviewer:
        from ..candidate_review import (
            AgentCandidateReviewer,
            CandidateReviewRequest,
            CommandCandidateReviewer,
            resolve_candidate_reviewer,
        )
        from .layout import resolve_module_attr

        components = {
            relative: (worktree / relative).read_text(encoding="utf-8")
            for relative in cfg.acceptance.component_files
            if (worktree / relative).is_file()
        }
        request = CandidateReviewRequest(
            components=components,
            diff=_git(
                worktree, "diff", baseline, "--", *cfg.acceptance.component_files
            ),
            workspace_path=str(worktree),
            opaque_context=cfg.acceptance.review_context,
            attempt=state.review_failures + 1,
            prior_findings=(),
        )
        reviewer_root = workspace_root if cfg.acceptance.pinned_scorer else worktree
        if cfg.acceptance.reviewer_kind == "module":
            reviewer = resolve_candidate_reviewer(
                cfg.acceptance.reviewer, expected_root=reviewer_root
            )
        elif cfg.acceptance.reviewer_kind == "agent":
            agent = resolve_module_attr(
                str(cfg.acceptance.reviewer),
                kind="reviewer agent",
                expected_root=reviewer_root,
            )
            reviewer = AgentCandidateReviewer(
                agent, samples=int(cfg.acceptance.reviewer_options.get("samples", 3))
            )
        else:
            reviewer = CommandCandidateReviewer(
                str(cfg.acceptance.reviewer),
                output_path=str(
                    cfg.acceptance.reviewer_options.get("output_path", "verdict.json")
                ),
                timeout_secs=float(
                    cfg.acceptance.reviewer_options.get("timeout_secs", 120)
                ),
                retries=int(cfg.acceptance.reviewer_options.get("retries", 1)),
            )
        verdict = reviewer.review(request)
        if verdict.disposition == "fail":
            findings = tuple(
                {
                    "component": item.component,
                    "excerpt": item.excerpt,
                    "explanation": item.explanation,
                    "severity": item.severity,
                }
                for item in verdict.findings
            )
            return _review_rejection(workspace_root, run_state, state, findings)
    if not cfg.acceptance.require_probe_receipt:
        return None
    from .probe import component_hash, is_fixed_status_change
    from .layout import probe_receipts_dir

    prediction_path = worktree / "prediction.json"
    try:
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        raw_predictions = prediction.get("predictions")
        if not isinstance(raw_predictions, list) or not raw_predictions:
            raise ValueError("predictions must be a non-empty list")
        predictions: list[tuple[str, str, str]] = []
        for item in raw_predictions:
            if not isinstance(item, dict):
                raise ValueError("each prediction must name key, case, and direction")
            key = item.get("key")
            case_id = item.get("case")
            direction = item.get("direction")
            if not isinstance(key, str) or not isinstance(case_id, str):
                raise ValueError("each prediction must name string key and case values")
            if direction != "fail_to_pass":
                raise ValueError("prediction direction must be 'fail_to_pass'")
            predictions.append((key, case_id, direction))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _review_rejection(
            workspace_root,
            run_state,
            state,
            (
                {
                    "component": "prediction.json",
                    "excerpt": None,
                    "explanation": f"A valid prediction.json is required: {exc}",
                    "severity": "error",
                },
            ),
        )
    component_digest = component_hash(worktree, cfg.acceptance.component_files)
    for receipt_path in probe_receipts_dir(run_state.run_id, workspace_root).glob(
        "*.json"
    ):
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if receipt.get("candidate_component_hash") != component_digest:
            continue
        if (
            receipt.get("lane") != state.lane
            or receipt.get("iteration") != state.iteration
        ):
            continue
        proof = receipt.get("proof")
        changes = receipt.get("changes")
        if not isinstance(proof, dict) or not isinstance(changes, dict):
            continue
        proof_tuple = (proof.get("key"), proof.get("case"), proof.get("direction"))
        if proof_tuple not in predictions:
            continue
        change = changes.get(proof.get("key"))
        if not isinstance(change, dict):
            continue
        if is_fixed_status_change(change.get("before"), change.get("after")):
            return None
    names = [f"{case_id}:{key}:{direction}" for key, case_id, direction in predictions]
    return _review_rejection(
        workspace_root,
        run_state,
        state,
        (
            {
                "component": "prediction.json",
                "excerpt": ", ".join(sorted(names)),
                "explanation": "No matching probe receipt proves a predicted assertion flip for this candidate.",
                "severity": "error",
            },
        ),
    )


def _review_rejection(
    workspace_root: Path,
    run_state: Any,
    state: LaneState,
    findings: tuple[dict[str, Any], ...],
) -> LaneState:
    failures = state.review_failures + 1
    rejected = LaneState(
        **{
            **state.to_dict(),
            "status": "stalled" if failures >= 3 else "paused_for_reflection",
            "review_failures": failures,
            "review_findings": findings,
            "lease_expires_at": None,
            "lease_purpose": None,
            "updated_at": utc_now_iso(),
        }
    )
    rejected.save(workspace_root, run_state.run_id)
    return rejected


def _candidate_changed_paths(worktree: Path, baseline: str) -> set[str]:
    """Return changed paths, including both sides of renames and untracked files."""

    def raw_git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(worktree), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    changed = {
        item
        for item in raw_git("diff", "--name-only", "-z", baseline).split("\0")
        if item
    }
    entries = raw_git("status", "--porcelain=v1", "-z").split("\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4:
            continue
        status = entry[:2]
        changed.add(entry[3:])
        if "R" in status or "C" in status:
            if index < len(entries) and entries[index]:
                changed.add(entries[index])
            index += 1
    return changed


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


def _stall_for_infrastructure_failure(
    *,
    workspace_root: Path,
    run_state: Any,
    lane_state: LaneState,
    git_state: Any,
    outcomes: list[Any],
    failures: tuple[EvaluationInfrastructureFailure, ...],
    valid_samples: list[float],
) -> LaneState:
    """Persist a non-selectable failed evaluation and preserve the incumbent."""

    run_id = run_state.run_id
    lane = lane_state.lane
    reports = [str(outcome.summary["report_path"]) for outcome in outcomes]
    traces = [
        str(outcome.summary["trace_path"])
        for outcome in outcomes
        if outcome.summary.get("trace_path")
    ]
    comparison = {
        "run_id": run_id,
        "lane": lane,
        "iteration": lane_state.iteration,
        "outcome": "infrastructure_failure",
        "selectable": False,
        "verdict": None,
        "retryable": True,
        "reason_code": "required_rollout_failed",
        "minibatch_id": run_state.reflection_minibatch_id,
        "baseline_candidate_id": run_state.reflection_baseline_candidate_id,
        "baseline_commit_sha": run_state.reflection_baseline_commit_sha,
        "candidate_id": git_state.candidate_id,
        "candidate_commit_sha": lane_state.candidate_sha,
        "candidate_report_paths": reports,
        "candidate_trace_paths": traces,
        "valid_samples_before_failure": list(valid_samples),
        "evaluation_error_count": len(failures),
        "evaluation_errors": [failure.to_dict() for failure in failures],
    }
    comparison_path = _write_comparison(
        workspace_root, run_id, lane, lane_state.iteration, comparison
    )
    stalled = LaneState(
        **{
            **lane_state.to_dict(),
            "status": "stalled",
            "eval_samples": (),
            "verdict": None,
            "verdict_delta": None,
            "comparison_path": str(comparison_path),
            "eval_pid": None,
            "heartbeat_at": None,
            "updated_at": utc_now_iso(),
        }
    )
    stalled.save(workspace_root, run_id)
    reason = (
        f"required rollout failed ({len(failures)} evaluation error(s)); "
        f"inspect {comparison_path} and reset the lane after infrastructure recovery"
    )
    emit(
        run_id,
        lane,
        EventDraft(
            type="lane_stalled",
            lane=lane,
            payload={"reason": reason, "lease_epoch": stalled.lease_epoch},
        ),
        root=workspace_root,
    )
    typer.echo(f"Lane {lane} stalled: {reason}.", err=True)
    return stalled


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
    candidate_project = (
        Path(lane_state.candidate_project_path)
        if lane_state.candidate_project_path
        else candidate_project_root(workspace_root, worktree)
    )
    branch = str(lane_state.branch)
    commit_sha = _auto_commit_worktree(worktree, branch, lane)

    try:
        git_state = git_candidate_state(candidate_project)
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

    cfg = GepaConfig.load(config_path(workspace_root))
    vector_mode = cfg.acceptance.mode == "vector"
    baseline_samples = tuple(float(v) for v in run_state.reflection_baseline_samples)
    max_candidate_samples = (
        run_state.acceptance_repetitions + 1 if vector_mode else len(baseline_samples)
    )
    initial_samples = min(run_state.acceptance_repetitions, max_candidate_samples)

    # The candidate tree is the cwd for evaluation: `evaluate` callables and
    # user metric code routinely read files by relative path, and those reads
    # must observe the lane worktree, not the primary checkout (detached
    # children are spawned with cwd=worktree; foreground callers are switched
    # here so both paths evaluate the same tree).
    previous_cwd = Path.cwd()
    os.chdir(candidate_project)
    samples: list[float] = []
    outcomes: list[Any] = []
    comparison_result = None
    vector_comparison: VectorComparison | None = None
    infra_retries = 0
    attempts = 0
    try:
        while len(samples) < max_candidate_samples:
            attempts += 1
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
                candidate_root=candidate_project,
                workspace_root=workspace_root,
                vector_incumbent_hash=run_state.reflection_baseline_candidate_id,
                vector_repetition=len(samples) + 1,
            )
            outcomes.append(outcome)
            current_id = str(outcome.summary["candidate_id"])
            if current_id != git_state.candidate_id:
                raise typer.BadParameter(
                    "The lane candidate changed mid-evaluation "
                    f"({git_state.candidate_id} -> {current_id}); refusing to "
                    "compare mixed candidates."
                )
            failures = evaluation_infrastructure_failures(outcome.records)
            if failures:
                if vector_mode and infra_retries == 0:
                    infra_retries += 1
                    continue
                return _stall_for_infrastructure_failure(
                    workspace_root=workspace_root,
                    run_state=run_state,
                    lane_state=lane_state,
                    git_state=git_state,
                    outcomes=outcomes,
                    failures=failures,
                    valid_samples=samples,
                )
            samples.append(float(outcome.summary["mean_score"]))
            lane_state = LaneState(
                **{**lane_state.to_dict(), "eval_samples": tuple(samples)}
            )
            lane_state = _touch_heartbeat(workspace_root, run_id, lane_state, pid)

            if len(samples) < initial_samples:
                continue
            if vector_mode:
                raw_record = outcome.summary.get("vector_record")
                if not isinstance(raw_record, dict):
                    raise typer.BadParameter(
                        "Vector acceptance requires a vector metric record."
                    )
                current = VectorRecord.from_dict(raw_record)
                store = VectorRecordStore(vector_records_path(run_id, workspace_root))
                candidate_records = tuple(store.matching(current.key))
                from dataclasses import replace

                incumbent_records = tuple(
                    store.matching(
                        replace(current.key, candidate_hash=current.key.incumbent_hash)
                    )
                )
                comparator = resolve_vector_comparator(
                    str(cfg.acceptance.comparator),
                    expected_root=(
                        workspace_root
                        if cfg.acceptance.pinned_scorer
                        else candidate_project
                    ),
                )
                vector_comparison = compare_vectors(
                    comparator,
                    VectorComparisonRequest(
                        incumbent=incumbent_records,
                        candidate=candidate_records,
                        attempt=attempts,
                        escalation=max(0, len(samples) - initial_samples),
                        journal_context={
                            "run_id": run_id,
                            "lane": lane,
                            "iteration": lane_state.iteration,
                            "comparison_kind": "candidate_acceptance",
                            "accepted_promotion_count": run_state.accepted_promotion_count,
                            "run_start_baseline": run_state.run_start_baseline,
                        },
                    ),
                )
                if vector_comparison.verdict != "needs_escalation":
                    break
                if len(samples) >= max_candidate_samples:
                    vector_comparison = VectorComparison(
                        verdict="equivalent",
                        ranking_key=vector_comparison.ranking_key,
                        display_score=vector_comparison.display_score,
                        detail={
                            **vector_comparison.detail,
                            "escalation_exhausted": True,
                        },
                    )
                    break
            else:
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

    if vector_comparison is not None:
        result_data = vector_comparison.to_dict()
        verdict = vector_comparison.verdict
        display = vector_comparison.display_score
    else:
        assert comparison_result is not None
        result_data = comparison_result.to_dict()
        verdict = comparison_result.verdict
        display = comparison_result.delta
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
        **result_data,
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
            "verdict": verdict,
            "verdict_delta": display,
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
                "verdict": verdict,
                "delta": display,
                "comparison_path": str(comparison_path),
            },
        ),
        root=workspace_root,
    )
    typer.echo(
        f"Lane {lane} verdict: {verdict} (display={display:+.4f}); event {verdict_id}."
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

        # Direct foreground continues and detached-parent continues both run
        # the gate. Only the fenced child skips it because its parent already
        # gated the exact unchanged worktree before spawning the child.
        if not (foreground and handoff_lease_epoch is not None):
            rejected = _candidate_gate(
                workspace_root=workspace_root, run_state=run_state, state=state
            )
            if rejected is not None:
                typer.echo(
                    f"Lane {lane} candidate review failed; no paired evaluation was spent. "
                    f"Round {rejected.review_failures}/3.",
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
    argv = _lane_continue_argv(
        gepa_abs=gepa_abs,
        lane=lane,
        run_id=run_state.run_id,
        foreground=True,
        handoff_lease_epoch=state.lease_epoch,
    )
    worktree_path = Path(str(state.worktree_path))
    candidate_project_path = (
        Path(state.candidate_project_path)
        if state.candidate_project_path
        else candidate_project_root(workspace_root, worktree_path)
    )
    try:
        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(  # noqa: S603
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=str(candidate_project_path),
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
                candidate_project_path=str(
                    candidate_project_root(workspace_root, path)
                ),
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
                _git(
                    git_root(workspace_root),
                    "worktree",
                    "remove",
                    "--force",
                    str(path),
                )
            except Exception:
                pass
            try:
                _git(git_root(workspace_root), "branch", "-D", branch)
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

"""Durable, agent-independent packets and handoff for single-path runs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import shlex
import tempfile
import time
from typing import Any, Iterator

import typer

from .candidates import (
    GitCandidateError,
    candidate_id_from_components,
    git_candidate_state,
)
from .layout import (
    GepaConfig,
    GepaConfigError,
    candidate_identity_exempt_paths,
    config_path,
    gepa_dir,
    insert_repo_root_on_path,
    notes_dir,
    repo_root,
    resolve_agent,
    resolve_skills,
    run_dir,
    run_state_path,
)
from .notes import notes_index
from .runs import ParetoLog, current_commit_sha, utc_now_iso
from .store import ComponentStore


def default_reflector() -> dict[str, Any]:
    """Defaults also used when opening a state written before handoffs existed."""
    return {
        "epoch": 1,
        "label": None,
        "issued_at": utc_now_iso(),
        "state": "active",
        "lost_at": None,
        "lost_reason": None,
        "history": [],
    }


@contextmanager
def run_lock(
    run_id: str,
    root: Path | None = None,
    *,
    wait: bool = False,
    timeout: float | None = None,
) -> Iterator[None]:
    """Serialize continuation and handoff; kernel releases the lock on death."""
    path = run_dir(run_id, root) / "run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            try:
                nonblocking = not wait or deadline is not None
                fcntl.flock(
                    handle.fileno(),
                    fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0),
                )
                break
            except BlockingIOError as exc:
                if wait and deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Run lock acquisition timed out") from exc
                    time.sleep(min(0.05, remaining))
                    continue
                handle.seek(0)
                pid = handle.read().strip() or "unknown"
                typer.echo(
                    f"Run {run_id} is locked by live process {pid}; retry after it finishes.",
                    err=True,
                )
                raise typer.Exit(code=1) from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            os.fsync(handle.fileno())
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _comparison_packet(comparison: dict[str, Any] | None) -> dict[str, Any] | None:
    if not comparison:
        return None
    # Explicitly project training fields. In particular, never copy validation
    # artifacts or opaque error/metric payloads into the reflector's packet.
    keys = (
        "outcome",
        "selectable",
        "minibatch_id",
        "baseline_candidate_id",
        "baseline_commit_sha",
        "baseline_iteration",
        "baseline_mean_score",
        "baseline_samples",
        "baseline_report_path",
        "baseline_report_paths",
        "baseline_trace_path",
        "baseline_trace_paths",
        "candidate_id",
        "candidate_commit_sha",
        "candidate_iteration",
        "candidate_mean_score",
        "candidate_samples",
        "candidate_report_path",
        "candidate_report_paths",
        "candidate_trace_path",
        "candidate_trace_paths",
        "verdict",
        "training_verdict",
        "training_mean_score",
        "delta",
        "lower_bound",
        "upper_bound",
        "confidence",
        "recommendation",
        "rejection_reason",
        "discard_command",
    )
    training = comparison.get("training_comparison", comparison)
    packet = {key: training[key] for key in keys if key in training}
    for key in (
        "outcome",
        "selectable",
        "verdict",
        "recommendation",
        "rejection_reason",
    ):
        if key in comparison:
            packet[key] = comparison[key]
    if (
        "validation_comparison" in comparison
        and "training_comparison" not in comparison
    ):
        # Older noise-acceptance states overwrote training statistics with
        # validation statistics. Do not present those as training evidence.
        for key in (
            "baseline_samples",
            "candidate_samples",
            "baseline_mean_score",
            "candidate_mean_score",
            "delta",
            "lower_bound",
            "upper_bound",
            "minibatch_id",
        ):
            packet.pop(key, None)
    if (
        comparison.get("reason_code") == "validation_rollout_failed"
        or comparison.get("rejection_reason") == "validation_infrastructure_failure"
    ):
        packet.pop("minibatch_id", None)
    if comparison.get("verdict") == "accepted":
        packet.pop("discard_command", None)
    packet["training_verdict"] = comparison.get(
        "training_verdict", comparison.get("verdict")
    )
    packet["validation"] = {
        "evaluated": bool(comparison.get("validation_evaluated", False)),
        "improved": comparison.get("validation_improved"),
        "mean": comparison.get("validation_mean_score"),
        "prior_best_mean": comparison.get("prior_best_validation_mean_score"),
    }
    gate = comparison.get("gate")
    if isinstance(gate, dict):
        packet["gate"] = {
            key: gate[key]
            for key in ("cases", "report_paths", "trace_paths")
            if key in gate
        }
    return packet


def already_scored(state: Any, candidate_id: str | None) -> bool:
    """Whether continue will reissue this candidate's terminal comparison."""
    comparison = state.last_comparison or state.last_reflector_comparison or {}
    return bool(
        state.continuation is None
        and candidate_id is not None
        and candidate_id == comparison.get("candidate_id")
        and comparison.get("verdict")
        in {"accepted", "rejected", "equivalent", "inconclusive"}
        and (
            state.status == "done"
            or (
                (
                    comparison.get("minibatch_id") == state.reflection_minibatch_id
                    or comparison.get("improved")
                )
                and (
                    candidate_id != state.reflection_baseline_candidate_id
                    or comparison.get("improved")
                )
            )
        )
    )


def _current_tree(state: Any, project: Path) -> dict[str, Any]:
    candidate_id = None
    commit_sha = current_commit_sha(project)
    dirty = False
    identity_error = None
    try:
        if state.candidate_source == "git":
            tree = git_candidate_state(
                project, exclude_paths=candidate_identity_exempt_paths(project)
            )
            candidate_id, commit_sha, dirty = (
                tree.candidate_id,
                tree.commit_sha,
                tree.dirty,
            )
        elif config_path(project).exists():
            config = GepaConfig.load(config_path(project))
            insert_repo_root_on_path(project)
            components = ComponentStore(project).effective_candidate(
                resolve_agent(config), skills_fs=resolve_skills(config, root=project)
            )
            candidate_id = candidate_id_from_components(components)
    except (GitCandidateError, GepaConfigError) as exc:
        # A packet still provides recovery instructions if the checkout/config
        # is temporarily unavailable, without claiming an invented identity.
        identity_error = type(exc).__name__
    comparison = state.last_comparison or state.last_reflector_comparison or {}
    scored = already_scored(state, candidate_id)
    result = {
        "candidate_id": candidate_id,
        "commit_sha": commit_sha,
        "dirty": dirty,
        "differs_from_baseline": candidate_id is not None
        and candidate_id != state.reflection_baseline_candidate_id,
        "already_scored": scored,
        "verdict": comparison.get("verdict") if scored else None,
    }
    if identity_error:
        result["identity_error"] = identity_error
    return result


def _instructions(state: Any, tree: dict[str, Any]) -> str:
    if state.status == "done":
        return "Run complete. Review the best candidate and final report."
    if state.continuation:
        candidate = state.continuation["candidate_id"]
        restore = (
            f"Restore candidate {candidate} before continuing. "
            if tree["candidate_id"] != candidate
            else ""
        )
        return (
            f"Continuation of candidate {candidate} was interrupted. {restore}"
            "Run next_command with its saved gate options to recover completed "
            "evaluations and finish the comparison. Preserve this candidate "
            "until its pending continuation finishes. To drop the checkpoint "
            "(for example, when overwritten components cannot be restored), run "
            f"gepa run resume --run-id {state.run_id} --abandon-continuation. "
            "Paid evaluations remain charged to the budget."
        )
    if state.status == "paused_after_infrastructure_error":
        return "A required evaluation rollout failed outside the quality comparison. The incumbent was preserved. Recover the service or configuration, then run next_command to retry."
    if tree["already_scored"]:
        followup = (
            "Inspect the current baseline training reports and traces, then edit a new proposal."
            if tree["verdict"] == "accepted"
            else "Revise the candidate or restore the baseline before continuing reflection."
        )
        return f"Candidate {tree['candidate_id']} already has a recorded {tree['verdict']} comparison. Review last_comparison; repeating continue returns the recorded result without scoring again. {followup}"
    if (
        state.candidate_source == "git"
        and tree["differs_from_baseline"]
        and not tree["dirty"]
    ):
        return f"Candidate {tree['commit_sha']} is committed but not scored; review it, then run next_command to score it, or git reset --hard {state.reflection_baseline_commit_sha} to drop it."
    if state.status == "paused_after_candidate_eval":
        comparison = state.last_comparison or {}
        if comparison.get("verdict") == "inconclusive":
            return "Candidate comparison remains inconclusive after the configured repetitions. Revise the candidate or restore the baseline, then run next_command."
        if comparison.get("verdict") == "equivalent":
            return "Candidate is equivalent within the configured practical delta. Restore the baseline or revise the candidate, then run next_command."
        if comparison.get("rejection_reason") == "validation":
            return "Candidate improved the training minibatch but did not improve held-out validation. Discard or revise the edits, then run next_command."
        return "Candidate did not beat the reflection baseline. Recommendation: discard or revise the edits, then run next_command."
    surface = (
        "source/artifacts and commit the result"
        if state.candidate_source == "git"
        else "components or source"
    )
    return f"Paused for reflection. Inspect the training reports and traces, edit {surface}, then run next_command."


def write_packet(run_id: str, root: Path | None = None) -> Path:
    """Rebuild the packet exclusively from the saved state and workspace disk."""
    from .lanes import _journal_tail
    from .run import RunState

    workspace_root = (root or repo_root()).resolve()
    state = RunState.from_dict(
        json.loads(run_state_path(run_id, workspace_root).read_text(encoding="utf-8"))
    )
    project = (
        Path(state.project_root).resolve() if state.project_root else workspace_root
    )
    workspace = gepa_dir(workspace_root).resolve()
    tree = _current_tree(state, project)
    reflector = state.reflector or default_reflector()
    comparison = _comparison_packet(
        state.last_comparison or state.last_reflector_comparison
    )
    argv = [
        "gepa",
        "--gepa-dir",
        str(workspace),
        "run",
        "continue",
        "--run-id",
        run_id,
        "--reflector-epoch",
        str(reflector["epoch"]),
    ]
    if state.continuation:
        for case in state.continuation.get("gate_case", []):
            argv.extend(["--gate-case", str(case)])
    budget_used = (
        ParetoLog(run_id, workspace_root).count_budget_rows()
        + state.gate_consumed_iterations
    )
    packet = {
        "packet_version": 1,
        "run_id": run_id,
        "status": state.status,
        "candidate_source": state.candidate_source,
        "instructions": _instructions(state, tree),
        "project_root": str(project),
        "cwd": str(project),
        "gepa_dir": str(workspace),
        "next_command": {
            "argv": argv,
            "cwd": str(project),
            "shell": f"cd {shlex.quote(str(project))} && {shlex.join(argv)}",
        }
        if state.status != "done"
        else None,
        "baseline": {
            "candidate_id": state.reflection_baseline_candidate_id,
            "commit_sha": state.reflection_baseline_commit_sha,
            "minibatch_id": state.reflection_minibatch_id,
            "mean_score": state.reflection_baseline_mean_score,
            "samples": list(state.reflection_baseline_samples),
            "report_paths": list(state.reflection_baseline_report_paths),
            "trace_paths": list(state.reflection_baseline_trace_paths),
        },
        "current_tree": tree,
        "last_comparison": comparison,
        "discard_command": comparison.get("discard_command") if comparison else None,
        "best": {
            "candidate_id": state.best_candidate_id,
            "commit_sha": state.best_commit_sha,
            "validation_mean": state.best_mean_score
            if state.validation_seeded
            else None,
        },
        "budget": {
            "used": budget_used,
            "remaining": max(0, state.max_iterations - budget_used),
            "max_iterations": state.max_iterations,
        },
        "journal_tail": _journal_tail(workspace_root, state.journal_tail_lines),
        "notes_index": [
            note.to_dict() for note in notes_index(notes_dir(workspace_root))
        ],
        "reflector": reflector,
    }
    if state.continuation:
        packet["pending_continuation"] = {
            "candidate_id": state.continuation["candidate_id"],
            "gate_case": list(state.continuation.get("gate_case", [])),
        }
    if state.iterations_since_acceptance >= state.stall_threshold:
        packet["stall"] = {
            "stalled": True,
            "iterations_since_acceptance": state.iterations_since_acceptance,
        }
    path = run_dir(run_id, workspace_root) / "reflector_packet.json"
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(packet, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise
    return path


def resume(
    run_id: str | None = None,
    reason: str | None = None,
    reflector: str | None = None,
    abandon_continuation: bool = False,
) -> None:
    """Declare the previous reflector lost and issue a fresh reflection packet."""
    from .lanes import _append_journal
    from .run import _load_state

    state = _load_state(run_id)
    if state.lanes > 0:
        typer.echo(
            "Lane runs use `gepa lane reset` / `gepa lane lease`; single-path resume is unavailable.",
            err=True,
        )
        raise typer.Exit(code=2)
    with run_lock(state.run_id):
        state = _load_state(state.run_id)
        if abandon_continuation:
            from .reflector_recovery import abandon_continuation as abandon

            state = abandon(state, reason=reason or "explicit_resume")
        now = utc_now_iso()
        old = state.reflector or default_reflector()
        lost = {key: value for key, value in old.items() if key != "history"}
        lost.update(state="lost", lost_at=now, lost_reason=reason)
        issued = {
            "epoch": int(old["epoch"]) + 1,
            "label": reflector,
            "issued_at": now,
            "state": "active",
            "lost_at": None,
            "lost_reason": None,
            "history": [*old.get("history", []), lost][-20:],
        }
        project = Path(state.project_root) if state.project_root else repo_root()
        tree = _current_tree(state, project)
        _append_journal(
            repo_root(),
            {
                "kind": "reflector_lost",
                "run_id": state.run_id,
                "timestamp": now,
                "old_epoch": old["epoch"],
                "new_epoch": issued["epoch"],
                "reason": reason,
                "unscored_candidate_commit": tree["commit_sha"]
                if tree["differs_from_baseline"] and not tree["already_scored"]
                else None,
            },
        )
        state = replace(state, reflector=issued, updated_at=now)
        state.save()
        path = write_packet(state.run_id)
        packet = json.loads(path.read_text(encoding="utf-8"))
        typer.echo(packet["instructions"])
        if packet["next_command"]:
            typer.echo(packet["next_command"]["shell"])
        typer.echo(json.dumps({"packet": packet}))

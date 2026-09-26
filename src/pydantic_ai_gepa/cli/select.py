"""`gepa run select` — lockstep multi-way selection and re-fan (spec-er3).

Select is the single sequential authority that turns N lane verdicts into at
most one promoted candidate, advances the run's best, teaches the journal what
lost and why, and re-fans lanes for the next iteration
(pydanticaigepa-task-vso).

Phase model — recorded in run state as ``select_phase`` with resumption
progress in ``select_context``; every phase checkpoints before the next
begins, so a select killed mid-way resumes idempotently from the recorded
phase on the next invocation:

1. ``promote`` — invalidate stragglers (terminate the eval pid, journal the
   partial result + diff summary, mark ``stalled``; pydanticaigepa-dec-4tw),
   invalidate lanes whose candidate no longer descends from the frozen
   baseline commit, validate every training-accepted candidate on the held-out
   dataset, use aggregate score in scalar mode or the configured comparator in
   vector mode, promote the winning validation improvement to the run's best
   in controller-owned storage (the original checkout stays unchanged),
   journal the winner, and emit
   ``merge_opportunity`` for accepted lane pairs with disjoint diffs (merging
   itself is always delegated to the coding agent — never auto-merged).
2. ``journal`` — every non-promoted resolved lane is journaled (diff summary,
   verdict, delta, confidence) *before* its branch is deleted.
3. ``run_start_rebaseline`` — when vector ``acceptance.rebaseline_interval``
   reaches an accepted-promotion multiple, compare the promoted incumbent
   with the immutable run-start vector set. Its result is journal evidence;
   it never rolls back or promotes either side.
4. ``refan`` — replace each independent lane repository with a complete
   checkout of the retained best on a fresh lane branch, and return lanes to
   ``paused_for_reflection`` with a bumped iteration and fresh lease epoch.
   Retention refs keep nominated commits available after lane deletion.
5. ``rebaseline`` — sample the next reflection minibatch and re-measure the
   shared baseline once per iteration against the new best tree (baseline
   evals are paid once per iteration, not once per lane).
6. ``emit`` — write fresh reflection packets and emit ``lane_ready`` per lane.

``finalize`` replaces re-fan when the pareto ledger reaches
``--max-iterations``: lane-run budget enforcement lives at select, not
per-eval (pydanticaigepa-dec-msy). The run is marked done, any overshoot
(in-flight lane evals beyond the cap, bounded by N x
``--acceptance-max-repetitions``) is recorded in the final report,
``run_done`` is emitted, and lane worktrees are removed — a lane is re-fanned
at select or removed when the run completes (spec-1do).

Journal writes and event emissions carry dedup keys, so re-executing a phase
after a kill is exactly-once. Lane verdicts are consumed from lane state —
they are memoized by the lane eval processes and never re-derived here.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator

import typer

from . import harness_record

from .eval import run_eval_once
from .events import EventDraft, emit, list_events
from ..evaluation_health import evaluation_infrastructure_failures
from ..vector_acceptance import (
    VectorComparisonRequest,
    VectorRecord,
    compare_vectors,
    resolve_vector_comparator,
)
from .lanes import (
    LaneState,
    _git,
    _pid_alive,
    _resolve_lane_run,
    lane_branch,
    lane_worktree_path,
    load_all_lane_states,
    load_lane_state,
    reaper_pass_for_run,
    scan_lane_states,
    write_packet,
)
from .layout import (
    GepaConfig,
    candidate_project_root,
    config_path,
    final_report_path,
    git_root,
    insert_repo_root_on_path,
    journal_path,
    run_dir,
    run_state_path,
    vector_records_path,
)
from .harness_record import VectorRecordStore
from .runs import ParetoLog, utc_now_iso

SELECT_PRODUCER_ID = "select"

# Phase ordering (spec-er3). ``finalize`` is the budget-exhausted terminal
# phase reached from ``journal`` instead of ``refan``.
SELECT_PHASES = (
    "promote",
    "journal",
    "run_start_rebaseline",
    "refan",
    "rebaseline",
    "emit",
    "finalize",
)

_PID_TERM_GRACE_SECS = 5.0
_PID_KILL_GRACE_SECS = 2.0


# ----------------------------- helpers ----------------------------------


def _save_state(state: Any, workspace_root: Path) -> None:
    """Persist run state at the explicit workspace (never derived from cwd).

    Atomic (tmpfile + os.replace): select's entire idempotent-resume contract
    rests on this file, so a kill mid-write must never leave torn JSON.
    """
    state.save(workspace_root)


def _checkpoint(
    state: Any, workspace_root: Path, phase: str | None, ctx: dict[str, Any] | None
) -> Any:
    """Record select phase progress; ``phase=None`` clears the in-flight marker."""
    fresh = replace(
        state,
        select_phase=phase,
        select_context=(dict(ctx) if ctx is not None else None),
        updated_at=utc_now_iso(),
    )
    _save_state(fresh, workspace_root)
    return fresh


@contextmanager
def _chdir(path: Path) -> Iterator[None]:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _load_comparison(lane_state: LaneState) -> dict[str, Any]:
    if lane_state.comparison_path and Path(lane_state.comparison_path).exists():
        from .validation import heldout_dataset

        path = Path(lane_state.comparison_path)
        if (
            heldout_dataset(required=False)
            and harness_record._public_workspace(path) is None
        ):
            raise harness_record._unavailable()
        data = json.loads(harness_record.read_text(path))
        if isinstance(data, dict):
            return data
    return {}


def _lane_outcome_journaled(
    workspace_root: Path, run_id: str, lane: str, iteration: int
) -> bool:
    """Dedup key for journal write-back: one lane outcome per (run, lane, iteration)."""
    path = journal_path(workspace_root)
    if not path.exists():
        return False
    for line in harness_record.read_text(path, root=workspace_root).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if (
            row.get("kind") == "lane_outcome"
            and row.get("run_id") == run_id
            and row.get("lane") == lane
            and row.get("iteration") == iteration
        ):
            return True
    return False


def _journal_lane_outcome(workspace_root: Path, entry: dict[str, Any]) -> None:
    """Append a lane-outcome journal entry unless already journaled (resume-safe)."""
    if _lane_outcome_journaled(
        workspace_root,
        str(entry["run_id"]),
        str(entry["lane"]),
        int(entry["iteration"]),
    ):
        return
    path = journal_path(workspace_root)
    from .harness_record import write_text

    if write_text(
        path, json.dumps(entry, sort_keys=True) + "\n", root=workspace_root, append=True
    ):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def _journal_rows(workspace_root: Path, run_id: str, kind: str) -> list[dict[str, Any]]:
    """Read this run's journal rows of one kind in append order."""
    from .harness_record import for_run

    record = for_run(run_id, workspace_root)
    path = journal_path(workspace_root)
    content = (
        (record.read("@select-journal") or "")
        if record
        else (
            harness_record.read_text(path, root=workspace_root) if path.exists() else ""
        )
    )
    rows: list[dict[str, Any]] = []
    for line in content.splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if (
            isinstance(raw, dict)
            and raw.get("run_id") == run_id
            and raw.get("kind") == kind
        ):
            rows.append(raw)
    return rows


def _append_journal_once(
    workspace_root: Path,
    entry: dict[str, Any],
    *,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Append one durable run-journal record, deduplicated for select resume."""
    from .harness_record import for_run

    kind = str(entry["kind"])
    run_id = str(entry["run_id"])
    for existing in _journal_rows(workspace_root, run_id, kind):
        if all(existing.get(key) == value for key, value in identity.items()):
            return existing
    record = for_run(run_id, workspace_root)
    if record:
        record.write(
            "@select-journal", json.dumps(entry, sort_keys=True) + "\n", append=True
        )
    path = journal_path(workspace_root)
    from .harness_record import write_text

    if write_text(
        path, json.dumps(entry, sort_keys=True) + "\n", root=workspace_root, append=True
    ):
        return entry
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def _record_accepted_promotion(
    workspace_root: Path,
    state: Any,
    winner: LaneState,
    *,
    candidate_id: str,
) -> int:
    """Append the authoritative promotion counter record exactly once."""
    existing = _journal_rows(workspace_root, state.run_id, "accepted_promotion")
    identity = {
        "lane": winner.lane,
        "iteration": winner.iteration,
        "candidate_sha": winner.candidate_sha,
    }
    for row in existing:
        if all(row.get(key) == value for key, value in identity.items()):
            return int(row["promotion_count"])
    count = max((int(row.get("promotion_count", 0)) for row in existing), default=0) + 1
    _append_journal_once(
        workspace_root,
        {
            "timestamp": utc_now_iso(),
            "kind": "accepted_promotion",
            "run_id": state.run_id,
            **identity,
            "candidate_id": candidate_id,
            "promotion_count": count,
            "content": f"accepted promotion {count}: {winner.lane} iteration {winner.iteration}",
        },
        identity=identity,
    )
    return count


def _lane_outcome_entry(
    run_id: str,
    lane_state: LaneState,
    *,
    outcome: str,
    diff_summary: str = "",
    confidence: float | None = None,
    reason: str | None = None,
    untracked_paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    delta = lane_state.verdict_delta
    verdict = lane_state.verdict or "none"
    content = (
        f"select {outcome}: lane {lane_state.lane} iteration "
        f"{lane_state.iteration} verdict={verdict}"
    )
    if delta is not None:
        content += f" delta={delta:+.4f}"
    if reason:
        content += f" ({reason})"
    return {
        "timestamp": utc_now_iso(),
        "kind": "lane_outcome",
        "strategy": "select",
        "run_id": run_id,
        "lane": lane_state.lane,
        "iteration": lane_state.iteration,
        "outcome": outcome,
        "branch": lane_state.branch,
        "candidate_sha": lane_state.candidate_sha,
        "verdict": lane_state.verdict,
        "delta": delta,
        "confidence": confidence,
        "eval_samples": list(lane_state.eval_samples),
        "diff_summary": diff_summary,
        "untracked_paths": list(untracked_paths),
        "reason": reason,
        "content": content,
    }


def _diff_stat(workspace_root: Path, base_sha: str, head_sha: str) -> str:
    try:
        return _git(workspace_root, "diff", "--stat", base_sha, head_sha)
    except Exception:  # noqa: BLE001 — missing refs on hand-crafted lane state
        return ""


def _changed_files(workspace_root: Path, base_sha: str, head_sha: str) -> set[str]:
    try:
        out = _git(workspace_root, "diff", "--name-only", base_sha, head_sha)
    except Exception:  # noqa: BLE001
        return set()
    return {line for line in out.splitlines() if line}


def _terminate_eval_pid(pid: int, *, lane: str) -> None:
    """Terminate a straggler's eval process: SIGTERM, then SIGKILL (dec-4tw)."""
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + _PID_TERM_GRACE_SECS
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _pid_alive(pid):
        os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + _PID_KILL_GRACE_SECS
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
    if _pid_alive(pid):
        typer.echo(
            f"Lane {lane} eval pid {pid} survived SIGKILL; refusing to "
            "continue select while a straggler eval is alive.",
            err=True,
        )
        raise typer.Exit(code=1)


def _mark_stalled(
    workspace_root: Path, run_id: str, lane_state: LaneState
) -> LaneState:
    fresh = LaneState(
        **{
            **lane_state.to_dict(),
            "status": "stalled",
            "eval_pid": None,
            "heartbeat_at": None,
            "updated_at": utc_now_iso(),
        }
    )
    fresh.save(workspace_root, run_id)
    return fresh


def _invalidate_straggler(
    workspace_root: Path, state: Any, lane_state: LaneState, *, reason: str
) -> None:
    """Invalidate a lane that was not resolved when select ran (dec-4tw).

    The eval process is terminated, the partial result and diff summary are
    journaled, and the lane is marked ``stalled``. A straggler is never
    compared against any baseline and never promoted. Uncommitted lane work is
    recorded (diff stat + untracked paths) before the re-fan resets it.
    """
    run_id = state.run_id
    pid = lane_state.eval_pid
    # Only signal when the lane claims to be mid-eval: a stale pid on a lane
    # in any other status may have been recycled by the OS onto an unrelated
    # process, and killing by bare number must never hit an innocent owner.
    if pid is not None and lane_state.status == "evaluating" and _pid_alive(pid):
        typer.echo(f"Terminating straggler lane {lane_state.lane} eval (pid {pid}).")
        _terminate_eval_pid(pid, lane=lane_state.lane)

    diff_summary = ""
    untracked: tuple[str, ...] = ()
    baseline_sha = state.reflection_baseline_commit_sha
    worktree = lane_worktree_path(workspace_root, run_id, lane_state.lane)
    if worktree is not None and worktree.exists() and baseline_sha:
        try:
            # Working-tree diff against the frozen baseline captures both the
            # committed lane work and uncommitted tracked edits.
            diff_summary = _git(worktree, "diff", "--stat", baseline_sha)
            untracked = tuple(
                line[3:]
                for line in _git(worktree, "status", "--porcelain").splitlines()
                if line.startswith("?? ")
            )
        except Exception:  # noqa: BLE001 — best-effort record before reset
            diff_summary = ""
            untracked = ()
    _journal_lane_outcome(
        workspace_root,
        _lane_outcome_entry(
            run_id,
            lane_state,
            outcome="straggler",
            diff_summary=diff_summary,
            reason=reason,
            untracked_paths=untracked,
        ),
    )
    _mark_stalled(workspace_root, run_id, lane_state)


def _invalidate_cross_baseline(
    workspace_root: Path,
    state: Any,
    lane_state: LaneState,
    *,
    reason: str | None = None,
) -> None:
    """Invalidate a lane whose branch point is not the frozen baseline commit.

    Lanes are never compared against baselines from a different branch point;
    the lane is marked ``stalled``, never silently compared (spec-er3).
    """
    run_id = state.run_id
    pid = lane_state.eval_pid
    if pid is not None and lane_state.status == "evaluating" and _pid_alive(pid):
        _terminate_eval_pid(pid, lane=lane_state.lane)
    baseline_sha = str(state.reflection_baseline_commit_sha)
    diff_summary = (
        _diff_stat(workspace_root, baseline_sha, lane_state.candidate_sha)
        if lane_state.candidate_sha
        else ""
    )
    _journal_lane_outcome(
        workspace_root,
        _lane_outcome_entry(
            run_id,
            lane_state,
            outcome="invalidated",
            diff_summary=diff_summary,
            confidence=_load_comparison(lane_state).get("confidence"),
            reason=reason
            or (
                "candidate commit does not descend from the frozen baseline "
                f"commit {baseline_sha}; never compared or promoted"
            ),
        ),
    )
    _mark_stalled(workspace_root, run_id, lane_state)


def _is_ancestor(workspace_root: Path, ancestor_sha: str, head_sha: str) -> bool:
    from .safe_git import run_git

    from .lane_repositories import candidate_root

    completed = run_git(
        candidate_root(workspace_root),
        "merge-base",
        "--is-ancestor",
        ancestor_sha,
        head_sha,
        capture_output=True,
    )
    return completed.returncode == 0


def _emit_merge_opportunities(
    workspace_root: Path,
    state: Any,
    ctx: dict[str, Any],
    accepted: list[LaneState],
) -> None:
    """Emit merge_opportunity for accepted lane pairs with disjoint diffs.

    Auto-merge is prohibited (spec-er3): the event surfaces the branch pair
    plus a diff-stat path for the orchestrator to resolve with a real merge.
    """
    run_id = state.run_id
    baseline_sha = str(state.reflection_baseline_commit_sha)
    file_sets = {
        lane_state.lane: _changed_files(
            workspace_root, baseline_sha, str(lane_state.candidate_sha)
        )
        for lane_state in accepted
        if lane_state.candidate_sha
    }
    existing = list_events(run_id, workspace_root)
    pairs: list[dict[str, list[str]]] = []
    for first, second in combinations(
        sorted(accepted, key=lambda lane_state: lane_state.lane), 2
    ):
        files_a = file_sets.get(first.lane, set())
        files_b = file_sets.get(second.lane, set())
        if not files_a or not files_b or not files_a.isdisjoint(files_b):
            continue
        merge_dir = run_dir(run_id, workspace_root) / "merge_opportunities"
        stat_path = merge_dir / f"{first.lane}--{second.lane}.diffstat"
        contents = (
            f"# {first.lane} ({first.branch}) vs baseline {baseline_sha}\n"
            + _diff_stat(workspace_root, baseline_sha, str(first.candidate_sha))
            + f"\n\n# {second.lane} ({second.branch}) vs baseline {baseline_sha}\n"
            + _diff_stat(workspace_root, baseline_sha, str(second.candidate_sha))
            + "\n"
        )
        if not harness_record.write_text(stat_path, contents, root=workspace_root):
            merge_dir.mkdir(parents=True, exist_ok=True)
            stat_path.write_text(contents, encoding="utf-8")
        duplicate = any(
            event.type == "merge_opportunity"
            and event.payload.get("lane_a") == first.lane
            and event.payload.get("lane_b") == second.lane
            and event.payload.get("branch_a") == first.branch
            and event.payload.get("branch_b") == second.branch
            for event in existing
        )
        if not duplicate:
            emit(
                run_id,
                SELECT_PRODUCER_ID,
                EventDraft(
                    type="merge_opportunity",
                    lane=None,
                    payload={
                        "lane_a": first.lane,
                        "lane_b": second.lane,
                        "branch_a": str(first.branch),
                        "branch_b": str(second.branch),
                        "commit_a": first.candidate_sha,
                        "commit_b": second.candidate_sha,
                        "diff_stat_path": str(stat_path),
                    },
                ),
                root=workspace_root,
            )
        pairs.append(
            {
                "lanes": [first.lane, second.lane],
                "branches": [str(first.branch), str(second.branch)],
                "shas": [str(first.candidate_sha), str(second.candidate_sha)],
            }
        )
        typer.echo(
            f"merge_opportunity: {first.lane} and {second.lane} touched disjoint "
            f"files; resolve with a real git merge (never auto-merged). "
            f"Diff stat: {stat_path}"
        )
    ctx["merge_pairs"] = pairs


# ----------------------------- phases -----------------------------------


@contextmanager
def _validation_incumbent_root(workspace_root: Path, commit_sha: str) -> Iterator[Path]:
    """Yield the exact incumbent tree without touching the primary checkout."""

    from .lane_repositories import candidate_root, checkout
    import tempfile

    repository = git_root(candidate_root(workspace_root))
    with tempfile.TemporaryDirectory(dir=repository.parent) as temporary:
        target = Path(temporary) / "incumbent"
        checkout(repository, commit_sha, target, "incumbent")
        yield candidate_project_root(workspace_root, target)


def _numeric_ranking_key(raw: Any, *, label: str) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)):
        raise typer.BadParameter(f"{label} ranking_key must be a numeric tuple.")
    ranking: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise typer.BadParameter(f"{label} ranking_key must be a numeric tuple.")
        try:
            number = float(item)
        except OverflowError as exc:
            raise typer.BadParameter(
                f"{label} ranking_key values must be finite."
            ) from exc
        if not math.isfinite(number):
            raise typer.BadParameter(f"{label} ranking_key values must be finite.")
        ranking.append(number)
    return tuple(ranking)


def _phase_promote(
    workspace_root: Path, state: Any, ctx: dict[str, Any]
) -> tuple[Any, dict[str, Any], str]:
    """Invalidate stragglers/cross-baseline lanes, pick + promote the winner."""
    if "accepted_promotion_count" in ctx:
        # The promotion checkpoint includes the new incumbent evidence. Do
        # not compare the winner with itself when resuming that checkpoint.
        if "merge_pairs" not in ctx:
            results = ctx.get("validation_results", {})
            accepted = [
                lane
                for lane in load_all_lane_states(workspace_root, state.run_id)
                if lane.verdict == "accepted"
                and lane.lane not in ctx.get("invalidated", [])
                and (
                    not results
                    or (
                        results.get(lane.lane, {}).get("selectable")
                        and results[lane.lane]
                        .get("comparison", {})
                        .get("verdict", "accepted")
                        == "accepted"
                    )
                )
            ]
            _emit_merge_opportunities(workspace_root, state, ctx, accepted)
        return state, ctx, "journal"
    run_id = state.run_id
    baseline_sha = state.reflection_baseline_commit_sha
    if not baseline_sha:
        typer.echo(
            f"Run {run_id} has no frozen reflection baseline commit; cannot select.",
            err=True,
        )
        raise typer.Exit(code=1)

    lane_states = load_all_lane_states(workspace_root, run_id)
    from .lane_repositories import CandidateAncestryError, LaneSourceError, load

    repositories = load(workspace_root, run_id)

    unrelated: dict[str, str | None] = {}
    candidate_roots: dict[str, Path] = {}
    for proposal in lane_states:
        if proposal.status == "awaiting_selection" and proposal.candidate_sha:
            try:
                repositories.import_lane(proposal.lane, proposal.candidate_sha)
                candidate_roots[proposal.lane] = repositories.candidate(
                    proposal.candidate_sha
                )
            except CandidateAncestryError:
                unrelated[proposal.lane] = None
            except LaneSourceError as error:
                unrelated[proposal.lane] = (
                    f"Candidate import/checkout refused ({type(error).__name__}); never compared or promoted"
                )

    # 1. Straggler invalidation (dec-4tw): anything not awaiting_selection when
    #    select runs is terminated, journaled, and stalled — never compared,
    #    never promoted. Idempotent: journaled lanes dedup, stalled markings
    #    repeat harmlessly.
    stragglers: list[str] = []
    for lane_state in lane_states:
        if lane_state.status == "awaiting_selection":
            continue
        _invalidate_straggler(
            workspace_root,
            state,
            lane_state,
            reason="lane was not resolved when select ran",
        )
        stragglers.append(lane_state.lane)

    # 2. Frozen-baseline consistency: a lane whose candidate commit no longer
    #    descends from the frozen branch point is invalidated, never compared.
    invalidated: list[str] = []
    valid: list[LaneState] = []
    for lane_state in lane_states:
        if lane_state.status != "awaiting_selection":
            continue
        candidate_sha = lane_state.candidate_sha
        if (
            candidate_sha
            and lane_state.lane not in unrelated
            and _is_ancestor(workspace_root, str(baseline_sha), candidate_sha)
        ):
            valid.append(lane_state)
        else:
            _invalidate_cross_baseline(
                workspace_root, state, lane_state, reason=unrelated.get(lane_state.lane)
            )
            invalidated.append(lane_state.lane)

    # 3. Training verdicts are consumed from lane state (memoized by the lane
    #    eval). Every training-accepted proposal is then scored on held-out
    #    validation. Vector mode scores the incumbent and every proposal for
    #    the configured maximum repetition count, then applies the configured
    #    comparator entirely in memory. Validation writes no reports, traces,
    #    or vectors, so its evidence never enters a reflection packet.
    accepted = [lane_state for lane_state in valid if lane_state.verdict == "accepted"]
    config = GepaConfig.load(config_path(workspace_root))
    from .harness_record import for_run

    validation_enabled = for_run(run_id, workspace_root) is not None
    if state.heldout_required and not validation_enabled:
        raise typer.BadParameter("Held-out selection requires harness access.")
    vector_validation = validation_enabled and config.acceptance.mode == "vector"
    validation_results = {
        lane.lane: result
        for lane in accepted
        if (result := (ctx.get("validation_results") or {}).get(lane.lane))
        and result.get("commit_sha") == lane.candidate_sha
    }
    state = replace(
        state,
        iterations=max(
            state.iterations, ParetoLog(run_id, workspace_root).count_budget_rows()
        ),
    )
    from .run import _evaluate_validation_candidate

    if validation_enabled and not vector_validation:
        from .run import _ensure_validation_seed

        with _chdir(workspace_root):
            state, _ = _ensure_validation_seed(state)
        state = _checkpoint(state, workspace_root, "promote", ctx)
        if state.status == "paused_after_infrastructure_error":
            typer.echo(
                "Incumbent validation failed; no promotion. Recover the evaluator "
                "and retry `gepa run select`.",
                err=True,
            )
            raise typer.Exit(code=1)

    incumbent_vectors: tuple[VectorRecord, ...] = ()
    missing_validation = [
        lane_state
        for lane_state in accepted
        if lane_state.lane not in validation_results
    ]
    if vector_validation and missing_validation and validation_results:
        # Validation vectors are intentionally never persisted. An interrupted
        # round must therefore restart in full so every lane is compared with
        # the same freshly scored incumbent repetitions.
        validation_results = {}
        ctx["validation_results"] = {}
        missing_validation = list(accepted)

    validation_comparator = None
    if vector_validation and missing_validation:
        ctx["validation_rounds"] = int(ctx.get("validation_rounds", 0)) + 1
        state = _checkpoint(state, workspace_root, "promote", ctx)
        # Held-out selection is run-owned: candidates may change their own
        # unpinned scorer code, but never the comparator that ranks lanes.
        insert_repo_root_on_path(workspace_root)
        validation_comparator = resolve_vector_comparator(
            str(config.acceptance.comparator), expected_root=workspace_root
        )
        incumbent_outcomes = []
        validation_repetitions = state.acceptance_max_repetitions
        with _validation_incumbent_root(
            workspace_root, str(baseline_sha)
        ) as incumbent_root:
            for repetition in range(1, validation_repetitions + 1):
                with _chdir(incumbent_root):
                    state, incumbent_outcome = _evaluate_validation_candidate(
                        state,
                        candidate_root=incumbent_root,
                        workspace_root=workspace_root,
                        lane="incumbent:validation",
                        vector_repetition=repetition,
                    )
                incumbent_outcomes.append(incumbent_outcome)
        incumbent_failures = tuple(
            failure
            for outcome in incumbent_outcomes
            for failure in evaluation_infrastructure_failures(outcome.records)
        )
        if incumbent_failures:
            failure_history = dict(ctx.get("validation_infrastructure_failures") or {})
            failure_history["incumbent"] = {
                "evaluation_error_count": len(incumbent_failures),
                "error_kinds": ["infrastructure_failure"],
            }
            ctx["validation_infrastructure_failures"] = failure_history
            state = _checkpoint(state, workspace_root, "promote", ctx)
            typer.echo(
                "Held-out validation failed for the incumbent; no selection "
                "decision was memoized. Recover the evaluator and retry "
                "`gepa run select`.",
                err=True,
            )
            raise typer.Exit(code=1)
        incumbent_vectors = tuple(
            outcome.selection_vector
            for outcome in incumbent_outcomes
            if outcome.selection_vector is not None
        )
        if len(incumbent_vectors) != validation_repetitions:
            raise typer.BadParameter(
                "Vector held-out validation requires a vector metric record."
            )
        failure_history = dict(ctx.get("validation_infrastructure_failures") or {})
        failure_history.pop("incumbent", None)
        if failure_history:
            ctx["validation_infrastructure_failures"] = failure_history
        else:
            ctx.pop("validation_infrastructure_failures", None)

    for lane_state in accepted if validation_enabled else []:
        prior = validation_results.get(lane_state.lane)
        if prior and prior.get("commit_sha") == lane_state.candidate_sha:
            continue
        validation_results.pop(lane_state.lane, None)
        validation_lane = f"{lane_state.lane}:validation"
        recovered_row = (
            next(
                (
                    row
                    for row in reversed(ParetoLog(run_id, workspace_root).iter_rows())
                    if row.extra.get("row_scope") == "validation"
                    and row.extra.get("outcome") == "valid"
                    and row.lane == validation_lane
                    and row.commit_sha == lane_state.candidate_sha
                ),
                None,
            )
            if not vector_validation
            else None
        )
        if recovered_row is not None:
            ledger = ParetoLog(run_id, workspace_root)
            validation_results[lane_state.lane] = {
                "candidate_id": recovered_row.candidate_id,
                "commit_sha": recovered_row.commit_sha,
                "mean_score": recovered_row.mean_score,
                "selectable": recovered_row.extra.get("selectable") is not False,
            }
            ctx["validation_results"] = validation_results
            state = replace(
                state,
                iterations=max(state.iterations, ledger.count_budget_rows()),
                validation_evaluations=len(
                    [
                        row
                        for row in ledger.iter_rows()
                        if row.extra.get("row_scope") == "validation"
                    ]
                ),
            )
            state = _checkpoint(state, workspace_root, "promote", ctx)
            continue
        candidate_root = candidate_roots[lane_state.lane]
        validation_outcomes = []
        repetitions = state.acceptance_max_repetitions if vector_validation else 1
        if not vector_validation and state.iterations >= state.max_iterations:
            validation_results[lane_state.lane] = {
                "selectable": False,
                "reason_code": "validation_budget_exhausted",
            }
            ctx["validation_results"] = validation_results
            state = _checkpoint(state, workspace_root, "promote", ctx)
            continue
        for repetition in range(1, repetitions + 1):
            with _chdir(candidate_root):
                state, validation_outcome = _evaluate_validation_candidate(
                    state,
                    candidate_root=candidate_root,
                    workspace_root=workspace_root,
                    lane=validation_lane,
                    vector_incumbent_hash=(
                        incumbent_vectors[0].key.candidate_hash
                        if incumbent_vectors
                        else None
                    ),
                    vector_repetition=(repetition if vector_validation else None),
                )
            if validation_outcome.summary.get("commit_sha") != lane_state.candidate_sha:
                raise typer.BadParameter(
                    "Lane proposal differs from the commit scored by the harness; "
                    "refusing promotion."
                )
            validation_outcomes.append(validation_outcome)
        failures = tuple(
            failure
            for outcome in validation_outcomes
            for failure in evaluation_infrastructure_failures(outcome.records)
        )
        if failures:
            failure_history = dict(ctx.get("validation_infrastructure_failures") or {})
            failure_history[lane_state.lane] = {
                "evaluation_error_count": len(failures),
                "error_kinds": ["infrastructure_failure"],
            }
            ctx["validation_infrastructure_failures"] = failure_history
            state = _checkpoint(state, workspace_root, "promote", ctx)
            typer.echo(
                f"Held-out validation failed for {lane_state.lane}; no selection "
                "decision was memoized. Recover the evaluator and retry "
                "`gepa run select`.",
                err=True,
            )
            raise typer.Exit(code=1)
        result: dict[str, Any] = {
            "candidate_id": validation_outcomes[-1].summary["candidate_id"],
            "commit_sha": validation_outcomes[-1].summary.get("commit_sha"),
            "mean_score": sum(
                float(outcome.summary["mean_score"]) for outcome in validation_outcomes
            )
            / len(validation_outcomes),
            "selectable": all(
                outcome.summary.get("selectable") is not False
                for outcome in validation_outcomes
            ),
        }
        if vector_validation:
            candidate_vectors = tuple(
                outcome.selection_vector
                for outcome in validation_outcomes
                if outcome.selection_vector is not None
            )
            if not incumbent_vectors or len(candidate_vectors) != repetitions:
                raise typer.BadParameter(
                    "Vector held-out validation requires incumbent and candidate "
                    "vector metric records."
                )
            assert validation_comparator is not None
            validation_comparison = compare_vectors(
                validation_comparator,
                VectorComparisonRequest(
                    incumbent=incumbent_vectors,
                    candidate=candidate_vectors,
                    attempt=repetitions,
                    escalation=max(0, repetitions - state.acceptance_repetitions),
                    journal_context={
                        "run_id": run_id,
                        "lane": lane_state.lane,
                        "iteration": lane_state.iteration,
                        "comparison_kind": "validation_selection",
                        "accepted_promotion_count": state.accepted_promotion_count,
                        "run_start_baseline": state.run_start_baseline,
                    },
                ),
            )
            ranking_key = _numeric_ranking_key(
                validation_comparison.ranking_key,
                label="Vector validation comparison",
            )
            prior_arities = {
                len(
                    _numeric_ranking_key(
                        item["comparison"]["ranking_key"],
                        label="Vector validation comparison",
                    )
                )
                for item in validation_results.values()
                if isinstance(item, dict) and isinstance(item.get("comparison"), dict)
            }
            if prior_arities and prior_arities != {len(ranking_key)}:
                raise typer.BadParameter(
                    "Vector validation comparison ranking_key must have identical "
                    "arity for every lane."
                )
            result["comparison"] = {
                "verdict": validation_comparison.verdict,
                "ranking_key": list(ranking_key),
            }
        validation_results[lane_state.lane] = result
        failure_history = dict(ctx.get("validation_infrastructure_failures") or {})
        failure_history.pop(lane_state.lane, None)
        if failure_history:
            ctx["validation_infrastructure_failures"] = failure_history
        else:
            ctx.pop("validation_infrastructure_failures", None)
        ctx["validation_results"] = validation_results
        state = _checkpoint(state, workspace_root, "promote", ctx)

    if vector_validation:
        selectable_candidates = [
            lane_state
            for lane_state in accepted
            if validation_results[lane_state.lane]["selectable"]
            and validation_results[lane_state.lane]["comparison"]["verdict"]
            == "accepted"
        ]
    elif validation_enabled:
        selectable_candidates = [
            lane_state
            for lane_state in accepted
            if validation_results[lane_state.lane]["selectable"]
        ]
    else:
        selectable_candidates = accepted
    winner: LaneState | None = None
    if selectable_candidates:
        if vector_validation:

            def validation_rank(
                lane_state: LaneState,
            ) -> tuple[float | str, ...]:
                ranking = _numeric_ranking_key(
                    validation_results[lane_state.lane]["comparison"].get(
                        "ranking_key", []
                    ),
                    label="Vector validation comparison",
                )
                return tuple(-item for item in ranking) + (lane_state.lane,)

            winner = sorted(selectable_candidates, key=validation_rank)[0]
        elif validation_enabled:
            winner = sorted(
                selectable_candidates,
                key=lambda lane_state: (
                    -float(validation_results[lane_state.lane]["mean_score"]),
                    lane_state.lane,
                ),
            )[0]
        else:
            vector_mode = config.acceptance.mode == "vector"

            def training_rank(lane_state: LaneState) -> tuple[float | str, ...]:
                if not vector_mode:
                    return (-(lane_state.verdict_delta or 0.0), lane_state.lane)
                raw = _load_comparison(lane_state).get("ranking_key", [])
                if not isinstance(raw, list) or not all(
                    isinstance(item, (int, float)) for item in raw
                ):
                    raise typer.BadParameter(
                        "Vector comparison ranking_key must be a numeric tuple."
                    )
                return tuple(-float(item) for item in raw) + (lane_state.lane,)

            winner = sorted(selectable_candidates, key=training_rank)[0]
    confirmation_outcomes = []
    if winner is not None and validation_enabled and not vector_validation:
        from .run import _confirm_validation_candidate

        candidate_root = candidate_roots[winner.lane]
        with _chdir(candidate_root):
            state, confirmation_outcomes, confirmation = _confirm_validation_candidate(
                state,
                candidate_root=candidate_root,
                workspace_root=workspace_root,
                lane=f"{winner.lane}:confirmation",
            )
        ctx["validation_confirmation"] = confirmation
        state = replace(state, last_comparison=confirmation)
        if confirmation.get("outcome") == "infrastructure_failure":
            state = replace(state, status="paused_after_infrastructure_error")
            state = _checkpoint(state, workspace_root, "promote", ctx)
            typer.echo(
                "Held-out finalist confirmation failed; incumbent preserved. "
                "Recover the evaluator and retry `gepa run select`.",
                err=True,
            )
            raise typer.Exit(code=1)
        if (
            confirmation_outcomes
            and confirmation_outcomes[0].summary["candidate_id"]
            != validation_results[winner.lane]["candidate_id"]
        ):
            raise typer.BadParameter(
                "The lane finalist changed between validation screening and confirmation; "
                "refusing to promote mixed candidate evidence."
            )
        if confirmation.get("verdict") != "accepted":
            if confirmation.get("reason_code"):
                typer.echo(f"Finalist not promoted: {confirmation['reason_code']}.")
            winner = None
        else:
            validation_results[winner.lane]["mean_score"] = confirmation[
                "candidate_mean"
            ]

    losers = [lane_state.lane for lane_state in valid if lane_state is not winner]

    ctx.update(
        {
            "baseline_sha": str(baseline_sha),
            "winner": winner.lane if winner else None,
            "winner_branch": winner.branch if winner else None,
            "winner_commit_sha": winner.candidate_sha if winner else None,
            "losers": losers,
            "stragglers": stragglers,
            "invalidated": invalidated,
            "new_lane_iteration": max(
                (lane_state.iteration for lane_state in lane_states), default=0
            )
            + 1,
        }
    )

    if winner is None:
        from .run import _consume_candidate_verdict

        counts_as_failed_hypothesis = (
            bool(accepted)
            and ctx.get("validation_confirmation", {}).get("verdict") != "inconclusive"
            if validation_enabled
            else any(
                lane_state.verdict in {"rejected", "equivalent"} for lane_state in valid
            )
        )
        if counts_as_failed_hypothesis:
            state = _consume_candidate_verdict(state, accepted=False)
        reason = (
            "improved held-out validation" if validation_enabled else "was accepted"
        )
        typer.echo(
            f"No lane {reason}; no promotion. All resolved lanes are journaled "
            "as losers and re-fan onto the current best."
        )
        ctx["primary_promoted"] = False
        return state, ctx, "journal"

    # 4. Promotion: the winner's commit becomes the run's best. The primary
    #    checkout is fast-forwarded only when clean and still on the old best;
    #    otherwise promotion happens in run state only (user work is never
    #    destroyed).
    comparison = _load_comparison(winner)
    if validation_enabled:
        winner_mean = float(validation_results[winner.lane]["mean_score"])
    else:
        winner_mean = comparison.get("candidate_mean", comparison.get("display_score"))
    if winner_mean is None and winner.eval_samples:
        winner_mean = sum(winner.eval_samples) / len(winner.eval_samples)
    resolved_winner_mean = (
        float(winner_mean) if winner_mean is not None else state.best_mean_score
    )
    winner_candidate_id = (
        validation_results[winner.lane]["candidate_id"]
        if validation_enabled
        else comparison.get("candidate_id") or str(winner.candidate_sha)[:12]
    )
    winner_sha = str(winner.candidate_sha)

    _journal_lane_outcome(
        workspace_root,
        _lane_outcome_entry(
            run_id,
            winner,
            outcome="promoted",
            diff_summary=_diff_stat(workspace_root, str(baseline_sha), winner_sha),
            confidence=comparison.get("confidence"),
        ),
    )
    promotion_count = _record_accepted_promotion(
        workspace_root,
        state,
        winner,
        candidate_id=str(winner_candidate_id),
    )
    acceptance = config.acceptance
    scheduled_run_start_rebaseline = bool(
        acceptance.mode == "vector"
        and acceptance.rebaseline_interval is not None
        and promotion_count % acceptance.rebaseline_interval == 0
        and state.run_start_baseline is not None
    )

    repositories.promote(winner_sha)
    primary_promoted = True

    state = replace(
        state,
        best_candidate_id=str(winner_candidate_id),
        best_commit_sha=winner_sha,
        best_mean_score=resolved_winner_mean,
        accepted_promotion_count=promotion_count,
    )
    if confirmation_outcomes:
        from .run import _mark_best_validation_samples

        state = _mark_best_validation_samples(state, confirmation_outcomes)
    from .run import _consume_candidate_verdict

    state = _consume_candidate_verdict(state, accepted=True)
    ctx["primary_promoted"] = primary_promoted
    ctx["winner_mean"] = resolved_winner_mean
    ctx["accepted_promotion_count"] = promotion_count
    ctx["scheduled_run_start_rebaseline"] = scheduled_run_start_rebaseline
    # Checkpoint immediately after the promotion decision: the recorded phase
    # state must always be a prefix of externally visible side effects
    # (primary reset, journal entry), so a crash here never leaves the
    # on-disk ctx claiming primary_promoted=False for a reset that happened.
    state = _checkpoint(state, workspace_root, "promote", ctx)
    if vector_validation:
        ranking_key = validation_results[winner.lane]["comparison"]["ranking_key"]
        selection_detail = (
            f"by held-out validation comparator (ranking key {tuple(ranking_key)})"
        )
    elif validation_enabled:
        selection_detail = (
            f"on held-out validation (mean {float(resolved_winner_mean):.4f})"
        )
    else:
        selection_detail = f"(delta {(winner.verdict_delta or 0.0):+.4f})"
    typer.echo(
        f"Promoted lane {winner.lane} {selection_detail} as the run's best: "
        f"{winner_sha[:12]}"
        + ("" if primary_promoted else " (run state only; primary untouched)")
    )

    # 5. merge_opportunity for accepted lanes with disjoint diffs.
    if len(selectable_candidates) >= 2:
        _emit_merge_opportunities(workspace_root, state, ctx, selectable_candidates)
    else:
        ctx["merge_pairs"] = []

    return state, ctx, "journal"


def _phase_journal(
    workspace_root: Path, state: Any, ctx: dict[str, Any]
) -> tuple[Any, dict[str, Any], str]:
    """Journal every non-promoted resolved lane before its branch is deleted."""
    run_id = state.run_id
    baseline_sha = str(ctx.get("baseline_sha") or state.reflection_baseline_commit_sha)
    journaled: list[str] = list(ctx.get("journaled_lanes", []))
    for lane in list(ctx.get("losers", [])):
        if lane in journaled:
            continue
        lane_state = load_lane_state(workspace_root, run_id, lane)
        comparison = _load_comparison(lane_state)
        diff_summary = (
            _diff_stat(workspace_root, baseline_sha, lane_state.candidate_sha)
            if lane_state.candidate_sha
            else ""
        )
        _journal_lane_outcome(
            workspace_root,
            _lane_outcome_entry(
                run_id,
                lane_state,
                outcome="loser",
                diff_summary=diff_summary,
                confidence=comparison.get("confidence"),
                reason="not promoted",
            ),
        )
        journaled.append(lane)
        ctx["journaled_lanes"] = journaled
        # Per-lane checkpoint: a kill here resumes with the next loser.
        state = _checkpoint(state, workspace_root, "journal", ctx)

    # Budget enforcement (dec-msy): the pareto ledger is the sole budget
    # authority; the stop condition for lane runs is checked here, not per-eval.
    rows = ParetoLog(run_id, workspace_root).count_budget_rows()
    ctx["budget_rows"] = rows
    ctx["overshoot"] = max(0, rows - state.max_iterations)
    config = GepaConfig.load(config_path(workspace_root))
    if state.heldout_required and config.acceptance.mode == "vector":
        validation_rounds = int(ctx.get("validation_rounds", 0))
        overshoot_bound = (
            state.lanes * state.acceptance_max_repetitions
            + validation_rounds * (state.lanes + 1) * state.acceptance_max_repetitions
        )
        bound_detail = (
            "lanes x acceptance max-repetitions, plus completed vector "
            "validation rounds x (lanes + incumbent) x max-repetitions"
        )
    else:
        overshoot_bound = state.lanes * (state.acceptance_max_repetitions + 1)
        bound_detail = "lanes x (acceptance max-repetitions + validation)"
    if ctx["overshoot"] > overshoot_bound:
        typer.echo(
            f"Warning: budget overshoot {ctx['overshoot']} exceeds the "
            f"lockstep bound {overshoot_bound} ({bound_detail}); "
            "investigate eval accounting.",
            err=True,
        )
    # budget_low early warning (dec-d0d): emitted once per iteration while
    # remaining budget is below lanes x acceptance max-repetitions.
    remaining = state.max_iterations - rows
    if 0 < remaining < overshoot_bound:
        existing = list_events(run_id, workspace_root)
        already = any(
            event.type == "budget_low"
            and event.payload.get("remaining_evals") == remaining
            for event in existing
        )
        if not already:
            emit(
                run_id,
                SELECT_PRODUCER_ID,
                EventDraft(
                    type="budget_low",
                    lane=None,
                    payload={"remaining_evals": remaining},
                ),
                root=workspace_root,
            )
            typer.echo(
                f"budget_low: {remaining} evals remain (threshold {overshoot_bound})."
            )
    if rows >= state.max_iterations:
        return state, ctx, "finalize"
    if ctx.get("scheduled_run_start_rebaseline"):
        return state, ctx, "run_start_rebaseline"
    return state, ctx, "refan"


def _run_start_rebaseline_root(
    workspace_root: Path, state: Any, ctx: dict[str, Any]
) -> Path:
    """Use retained objects, independent of the current lane contents."""
    from .lane_repositories import load

    return load(workspace_root, state.run_id).candidate(str(state.best_commit_sha))


def _journal_run_start_rebaseline(
    workspace_root: Path,
    state: Any,
    ctx: dict[str, Any],
    *,
    outcome: str,
    incumbent_candidate_id: str,
    candidate_component_hashes: dict[str, Any],
    comparison: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Persist one re-baseline result per scheduled promotion."""
    promotion_count = int(ctx["accepted_promotion_count"])
    return _append_journal_once(
        workspace_root,
        {
            "timestamp": utc_now_iso(),
            "kind": "run_start_rebaseline",
            "run_id": state.run_id,
            "promotion_count": promotion_count,
            "incumbent_candidate_id": incumbent_candidate_id,
            "incumbent_commit_sha": state.best_commit_sha,
            "candidate_component_hashes": candidate_component_hashes,
            "run_start_baseline": state.run_start_baseline,
            "outcome": outcome,
            "comparison": comparison,
            "error": error,
            "content": (
                f"run-start re-baseline after accepted promotion {promotion_count}: "
                f"{outcome}"
            ),
        },
        identity={"promotion_count": promotion_count},
    )


def _phase_run_start_rebaseline(
    workspace_root: Path, state: Any, ctx: dict[str, Any]
) -> tuple[Any, dict[str, Any], str]:
    """Pair the promoted incumbent with the immutable run-start vector set.

    A comparator rejection is evidence, not rollback authority: the incumbent
    remains the selected candidate and lanes are re-fanned from it either way.
    The journal row is the idempotence marker for resume after a crash.
    """
    if ctx.get("run_start_rebaseline_finished"):
        return state, ctx, "refan"
    existing = _journal_rows(workspace_root, state.run_id, "run_start_rebaseline")
    promotion_count = int(ctx["accepted_promotion_count"])
    if any(int(row.get("promotion_count", -1)) == promotion_count for row in existing):
        ctx["run_start_rebaseline_finished"] = True
        return state, ctx, "refan"

    baseline = state.run_start_baseline
    if not isinstance(baseline, dict):
        _journal_run_start_rebaseline(
            workspace_root,
            state,
            ctx,
            outcome="unavailable",
            incumbent_candidate_id=str(state.best_candidate_id),
            candidate_component_hashes={},
            error="run-start baseline identity is unavailable",
        )
        ctx["run_start_rebaseline_finished"] = True
        return state, ctx, "refan"
    raw_keys = baseline.get("vector_record_keys")
    minibatch_id = baseline.get("minibatch_id")
    if (
        not isinstance(raw_keys, list)
        or not raw_keys
        or not isinstance(minibatch_id, str)
    ):
        _journal_run_start_rebaseline(
            workspace_root,
            state,
            ctx,
            outcome="unavailable",
            incumbent_candidate_id=str(state.best_candidate_id),
            candidate_component_hashes={},
            error="run-start baseline has no stored vector keys or minibatch",
        )
        ctx["run_start_rebaseline_finished"] = True
        return state, ctx, "refan"

    root = _run_start_rebaseline_root(workspace_root, state, ctx)
    cfg = GepaConfig.load(config_path(workspace_root))
    store = VectorRecordStore(vector_records_path(state.run_id, workspace_root))
    try:
        incumbent_records = tuple(store.records_for_keys(raw_keys))
    except ValueError as exc:
        _journal_run_start_rebaseline(
            workspace_root,
            state,
            ctx,
            outcome="unavailable",
            incumbent_candidate_id=str(state.best_candidate_id),
            candidate_component_hashes={},
            error=str(exc),
        )
        ctx["run_start_rebaseline_finished"] = True
        return state, ctx, "refan"

    current_records: list[VectorRecord] = []
    latest_summary: dict[str, Any] | None = None
    with _chdir(root):
        for raw_key in raw_keys:
            if not isinstance(raw_key, dict):
                raise typer.BadParameter("Stored run-start vector key is malformed.")
            outcome = run_eval_once(
                spend_state=state,
                spend_kind="baseline",
                candidate_file=None,
                minibatch_id=minibatch_id,
                size=state.size,
                seed=state.seed,
                epoch=state.next_epoch,
                run_id=state.run_id,
                concurrency=state.concurrency,
                max_iterations=state.max_iterations,
                threshold=state.threshold,
                capture_traces=True,
                candidate_source=state.candidate_source,
                lane="run-start-rebaseline",
                candidate_root=root,
                workspace_root=workspace_root,
                vector_incumbent_hash=str(baseline["candidate_id"]),
                vector_repetition=int(raw_key["repetition"]),
                row_scope="run_start_rebaseline",
            )
            latest_summary = outcome.summary
            failures = evaluation_infrastructure_failures(outcome.records)
            raw_vector = outcome.summary.get("vector_record")
            if failures or not isinstance(raw_vector, dict):
                _journal_run_start_rebaseline(
                    workspace_root,
                    state,
                    ctx,
                    outcome="infrastructure_failure",
                    incumbent_candidate_id=str(outcome.summary["candidate_id"]),
                    candidate_component_hashes=dict(
                        outcome.summary.get("component_hashes") or {}
                    ),
                    error="required re-baseline rollout did not yield a scored vector",
                )
                ctx["run_start_rebaseline_finished"] = True
                return state, ctx, "refan"
            current_records.append(VectorRecord.from_dict(raw_vector))

    assert latest_summary is not None
    comparator = resolve_vector_comparator(
        str(cfg.acceptance.comparator),
        expected_root=workspace_root if cfg.acceptance.pinned_scorer else root,
    )
    comparison = compare_vectors(
        comparator,
        VectorComparisonRequest(
            incumbent=incumbent_records,
            candidate=tuple(current_records),
            attempt=len(current_records),
            escalation=0,
            journal_context={
                "run_id": state.run_id,
                "lane": ctx.get("winner"),
                "iteration": ctx.get("new_lane_iteration"),
                "comparison_kind": "run_start_rebaseline",
                "accepted_promotion_count": state.accepted_promotion_count,
                "run_start_baseline": baseline,
            },
        ),
    )
    _journal_run_start_rebaseline(
        workspace_root,
        state,
        ctx,
        outcome="passed" if comparison.verdict == "accepted" else "failed",
        incumbent_candidate_id=str(state.best_candidate_id),
        candidate_component_hashes=dict(latest_summary.get("component_hashes") or {}),
        comparison=comparison.to_dict(),
    )
    ctx["run_start_rebaseline_finished"] = True
    return state, ctx, "refan"


def _refan_lane(
    workspace_root: Path,
    state: Any,
    lane_state: LaneState,
    *,
    new_branch: str,
    new_iteration: int,
    new_best: str,
) -> LaneState:
    """Reset one lane worktree onto the new best with a fresh branch.

    Idempotent: a lane already on ``new_branch`` at ``new_best`` (resume after
    a mid-phase kill) is left alone; its state is still rewritten below.
    """
    run_id = state.run_id
    from .lane_repositories import load

    repositories = load(workspace_root, run_id)
    worktree = lane_worktree_path(workspace_root, run_id, lane_state.lane)
    try:
        on_new_branch = (
            worktree.exists()
            and (worktree / ".git").is_dir()
            and (
                _git(worktree, "rev-parse", "--abbrev-ref", "HEAD") == new_branch
                and _git(worktree, "rev-parse", "HEAD") == new_best
            )
        )
    except (OSError, ValueError, typer.BadParameter, subprocess.CalledProcessError):
        on_new_branch = False
    if not on_new_branch:
        worktree = repositories.create_lane(
            lane_state.lane, new_best, new_branch, replace=True
        )
    # A crash can publish the lane before its scorer snapshot. Resume must
    # repair the snapshot even when the lane already has the expected branch.
    if GepaConfig.load(config_path(workspace_root)).acceptance.pinned_scorer:
        repositories.ensure_scorer(lane_state.lane, new_best)
    return LaneState(
        **{
            **lane_state.to_dict(),
            "status": "paused_for_reflection",
            "iteration": new_iteration,
            "branch": new_branch,
            "worktree_path": str(worktree),
            "candidate_project_path": str(
                candidate_project_root(workspace_root, worktree)
            ),
            "packet_path": None,
            "lease_epoch": lane_state.lease_epoch + 1,
            "lease_expires_at": None,
            "candidate_sha": None,
            "eval_samples": (),
            "verdict": None,
            "verdict_delta": None,
            "comparison_path": None,
            "eval_pid": None,
            "heartbeat_at": None,
            "updated_at": utc_now_iso(),
        }
    )


def _phase_refan(
    workspace_root: Path, state: Any, ctx: dict[str, Any]
) -> tuple[Any, dict[str, Any], str]:
    """Reset all lanes onto the new best and delete journaled branches."""
    run_id = state.run_id
    new_best = state.best_commit_sha
    if not new_best:
        typer.echo(
            f"Run {run_id} has no best commit to re-fan onto; cannot continue.",
            err=True,
        )
        raise typer.Exit(code=1)
    new_iteration = int(ctx.get("new_lane_iteration", 0)) or 1
    refanned: list[str] = list(ctx.get("refanned_lanes", []))
    for lane_state in load_all_lane_states(workspace_root, run_id):
        if lane_state.lane in refanned:
            continue
        new_branch = lane_branch(run_id, lane_state.lane, new_iteration)
        fresh = _refan_lane(
            workspace_root,
            state,
            lane_state,
            new_branch=new_branch,
            new_iteration=new_iteration,
            new_best=str(new_best),
        )
        fresh.save(workspace_root, run_id)
        refanned.append(lane_state.lane)
        ctx["refanned_lanes"] = refanned
        state = _checkpoint(state, workspace_root, "refan", ctx)
        typer.echo(
            f"Lane {lane_state.lane} re-fanned onto {str(new_best)[:12]} "
            f"(branch {new_branch}, iteration {new_iteration})."
        )
    return state, ctx, "rebaseline"


def _phase_rebaseline(
    workspace_root: Path, state: Any, ctx: dict[str, Any]
) -> tuple[Any, dict[str, Any], str]:
    """Sample the next minibatch and re-measure the shared baseline once.

    The baseline is measured against the new best tree: the primary checkout
    when it carries the new best, otherwise the first lane's freshly re-fanned
    worktree (identical content). Baseline evals are paid once per iteration,
    not once per lane (spec-er3).
    """
    from .run import _mark_reflection_pause, _with_last_outcome, _with_timestamp

    run_id = state.run_id
    if ctx.get("baseline_captured"):
        return state, ctx, "emit"

    new_best = str(state.best_commit_sha)
    from .lane_repositories import load

    baseline_root = load(workspace_root, run_id).candidate(new_best)

    ledger = ParetoLog(run_id, workspace_root)
    remaining = state.max_iterations - ledger.count_budget_rows()
    affordable_repetitions = max(1, (remaining + 1) // 2)
    target_repetitions = min(state.acceptance_max_repetitions, affordable_repetitions)
    scalar_mode = (
        GepaConfig.load(config_path(workspace_root)).acceptance.mode != "vector"
    )

    outcomes: list[Any] = []

    def fail_on_infrastructure_error(outcome: Any) -> None:
        failures = evaluation_infrastructure_failures(outcome.records)
        if not failures:
            return
        diagnostic = {
            "outcome": "infrastructure_failure",
            "selectable": False,
            "retryable": True,
            "reason_code": "required_rollout_failed",
            "valid_samples_before_failure": [
                float(item.summary["mean_score"]) for item in outcomes[:-1]
            ],
            "report_path": str(outcome.summary["report_path"]),
            "trace_path": outcome.summary.get("trace_path"),
            "evaluation_errors": [failure.to_dict() for failure in failures],
        }
        history = list(ctx.get("rebaseline_infrastructure_failures", []))
        history.append(diagnostic)
        ctx["rebaseline_infrastructure_failures"] = history
        _checkpoint(state, workspace_root, "rebaseline", ctx)
        typer.echo(
            "Shared baseline re-evaluation hit an infrastructure failure; "
            f"incumbent preserved. Inspect {outcome.summary['report_path']} and "
            "retry `gepa run select` after recovery.",
            err=True,
        )
        raise typer.Exit(code=1)

    with _chdir(baseline_root):
        first = run_eval_once(
            spend_state=state,
            spend_kind="baseline",
            candidate_file=None,
            minibatch_id=None,
            size=state.size,
            seed=state.seed,
            epoch=state.next_epoch,
            run_id=run_id,
            concurrency=state.concurrency,
            max_iterations=state.max_iterations,
            threshold=state.threshold,
            capture_traces=True,
            candidate_source=state.candidate_source,
            lane=None,
            candidate_root=baseline_root,
            workspace_root=workspace_root,
        )
        state = _with_timestamp(state, next_epoch=state.next_epoch + 1)
        state = _with_last_outcome(state, first)
        outcomes.append(first)
        fail_on_infrastructure_error(first)
        if scalar_mode:
            from .run import _acceptance_schedule, _inconclusive_comparison

            initial, maximum = _acceptance_schedule(state, len(first.records))
            affordable_repetitions = remaining // (state.lanes + 1)
            if affordable_repetitions < initial:
                state = replace(
                    state,
                    last_comparison=_inconclusive_comparison(
                        "baseline_budget_exhausted"
                    ),
                )
                return state, ctx, "finalize"
            target_repetitions = min(maximum, affordable_repetitions)
        expected_candidate_id = str(first.summary["candidate_id"])
        minibatch_id = str(first.summary["minibatch_id"])
        while len(outcomes) < target_repetitions:
            outcome = run_eval_once(
                spend_state=state,
                spend_kind="baseline",
                candidate_file=None,
                minibatch_id=minibatch_id,
                size=state.size,
                seed=state.seed,
                epoch=state.next_epoch,
                run_id=run_id,
                concurrency=state.concurrency,
                max_iterations=state.max_iterations,
                threshold=state.threshold,
                capture_traces=True,
                candidate_source=state.candidate_source,
                lane=None,
                candidate_root=baseline_root,
                workspace_root=workspace_root,
            )
            if str(outcome.summary["candidate_id"]) != expected_candidate_id:
                typer.echo(
                    "The baseline candidate changed while collecting repeated "
                    "evaluations; refusing to compare mixed candidates.",
                    err=True,
                )
                raise typer.Exit(code=1)
            state = _with_last_outcome(state, outcome)
            outcomes.append(outcome)
            fail_on_infrastructure_error(outcome)

    state = _mark_reflection_pause(state, outcomes)
    # Lane runs stay "running" — lanes carry the reflection pause.
    state = replace(state, status="running")
    ctx["baseline_captured"] = True
    # Mid-phase checkpoint so a kill after capture never re-measures.
    state = _checkpoint(state, workspace_root, "rebaseline", ctx)
    typer.echo(
        f"Shared baseline re-measured at {new_best[:12]} "
        f"(samples {list(state.reflection_baseline_samples)})."
    )
    return state, ctx, "emit"


def _phase_emit(
    workspace_root: Path, state: Any, ctx: dict[str, Any]
) -> tuple[Any, dict[str, Any], str | None]:
    """Write fresh packets and emit lane_ready once per lane."""
    run_id = state.run_id
    started_ms = int(ctx.get("started_ms", 0))
    existing = list_events(run_id, workspace_root)
    emitted: list[str] = list(ctx.get("emitted_lanes", []))
    if "iteration_started_at" not in ctx:
        # Reset once, immediately before the next iteration becomes visible.
        # Persist the timestamp in the resume context so an interrupted emit
        # cannot extend the straggler window on every retry (spec-er3).
        iteration_started_at = utc_now_iso()
        ctx["iteration_started_at"] = iteration_started_at
        state = replace(state, iteration_started_at=iteration_started_at)
        state = _checkpoint(state, workspace_root, "emit", ctx)
    for lane_state in load_all_lane_states(workspace_root, run_id):
        if lane_state.lane in emitted:
            continue
        worktree = Path(str(lane_state.worktree_path))
        packet_path = write_packet(
            workspace_root,
            state,
            lane_state.lane,
            lane_state.iteration,
            worktree,
            str(lane_state.branch),
        )
        fresh = LaneState(
            **{
                **lane_state.to_dict(),
                "packet_path": str(packet_path),
                "updated_at": utc_now_iso(),
            }
        )
        fresh.save(workspace_root, run_id)
        duplicate = any(
            event.type == "lane_ready"
            and event.lane == lane_state.lane
            and int(event.id.split("-", 1)[0]) >= started_ms
            for event in existing
        )
        if not duplicate:
            emit(
                run_id,
                SELECT_PRODUCER_ID,
                EventDraft(
                    type="lane_ready",
                    lane=lane_state.lane,
                    payload={
                        "packet_path": str(packet_path),
                        "worktree_path": str(worktree),
                    },
                ),
                root=workspace_root,
            )
        emitted.append(lane_state.lane)
        ctx["emitted_lanes"] = emitted
        state = _checkpoint(state, workspace_root, "emit", ctx)
    return state, ctx, None


def _phase_finalize(
    workspace_root: Path, state: Any, ctx: dict[str, Any]
) -> tuple[Any, dict[str, Any], str | None]:
    """Budget exhausted: mark done, record overshoot, emit run_done.

    Lane repositories are removed after their nominated commits have been
    retained in the controller's store for replay and adoption.
    """
    from .run import _write_final_report

    run_id = state.run_id
    rows = ParetoLog(run_id, workspace_root).count_budget_rows()
    overshoot = max(0, rows - state.max_iterations)
    state = replace(state, iterations=rows, status="done")

    from .lane_repositories import load

    repositories = load(workspace_root, run_id)
    for lane_state in load_all_lane_states(workspace_root, run_id):
        if (
            lane_state.eval_pid
            and lane_state.status == "evaluating"
            and _pid_alive(lane_state.eval_pid)
        ):
            _terminate_eval_pid(lane_state.eval_pid, lane=lane_state.lane)
        repositories.remove_lane(lane_state.lane)
    repositories.remove_scorers()

    final_path, final_text = _write_final_report(
        state, overshoot=overshoot, root=workspace_root
    )
    duplicate = any(
        event.type == "run_done" for event in list_events(run_id, workspace_root)
    )
    if not duplicate:
        emit(
            run_id,
            SELECT_PRODUCER_ID,
            EventDraft(
                type="run_done",
                lane=None,
                payload={"final_report_path": str(final_path)},
            ),
            root=workspace_root,
        )
    typer.echo(
        f"Evaluation budget exhausted ({rows}/{state.max_iterations} rows"
        + (f", overshoot {overshoot}" if overshoot else "")
        + "); run marked done."
    )
    typer.echo(final_text.rstrip())
    return state, ctx, None


# ----------------------------- entry point -------------------------------


def _preselect_check(workspace_root: Path, state: Any) -> None:
    """Fresh-invocation preconditions: reaper pass + selection due-ness."""
    run_id = state.run_id
    reaper_pass_for_run(workspace_root, state)
    scan = scan_lane_states(workspace_root, run_id, state)
    lane_states = load_all_lane_states(workspace_root, run_id)
    resolved = [s for s in lane_states if s.status == "awaiting_selection"]
    unresolved = [s for s in lane_states if s.status != "awaiting_selection"]
    if not resolved:
        typer.echo(
            f"Run {run_id} has no lanes in awaiting_selection; nothing to "
            "select. Drive lanes with `gepa lane continue` first.",
            err=True,
        )
        raise typer.Exit(code=1)
    if unresolved and scan.selection_due is None:
        names = ", ".join(s.lane for s in unresolved)
        typer.echo(
            f"Selection is not due: lanes {names} are still in flight and the "
            "straggler timeout has not elapsed. Wait for `selection_due` "
            "(`gepa next --wait`) or resolve the lanes.",
            err=True,
        )
        raise typer.Exit(code=1)


@contextmanager
def _select_lock(workspace_root: Path, run_id: str) -> Iterator[None]:
    """Exclusive flock serializing select for a run (spec-er3: select never
    runs concurrently with itself).

    Held for the whole verb: two fresh selects serialize instead of both
    seeing select_phase=None (the pid guard alone raced on that window), and
    a crashed holder auto-releases — so a recycled pid can never wedge the
    run the way the pid-only guard could.
    """
    import fcntl

    lock_path = run_dir(run_id, workspace_root) / "select.lock"
    from .harness_record import view_file
    from .validation import heldout_dataset

    if heldout_dataset(required=False):
        with view_file(lock_path, root=workspace_root) as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def run_select(run_id: str | None) -> Any:
    """Execute `gepa run select` (see module docstring for the phase model)."""
    workspace_root, run_state = _resolve_lane_run(run_id)
    from .lane_repositories import load
    from .reflector import run_lock
    from .harness_record import for_run

    record = for_run(run_state.run_id, workspace_root)

    with (
        (
            run_lock(run_state.run_id, workspace_root, wait=True)
            if record is not None
            else nullcontext()
        ),
        _select_lock(workspace_root, run_state.run_id),
        load(workspace_root, run_state.run_id).route(),
    ):
        # Resolve before taking the lock only to locate the run. A second
        # selector may have waited while the first completed every phase, so
        # its pre-lock snapshot must never drive a second selection.
        from .run import RunState

        from .harness_record import read_text

        raw = json.loads(
            read_text(
                run_state_path(run_state.run_id, workspace_root), root=workspace_root
            )
        )
        if not isinstance(raw, dict):
            raise typer.BadParameter("Managed run state must be a JSON object.")
        return _run_select_locked(
            workspace_root,
            RunState.from_dict(raw).restore_validation_evidence(workspace_root),
        )


def _run_select_locked(workspace_root: Path, run_state: Any) -> Any:
    if run_state.status == "done":
        typer.echo(
            f"Run {run_state.run_id} is done; there is nothing to select.",
            err=True,
        )
        raise typer.Exit(code=1)

    if getattr(run_state, "lane_repository_version", 0) != 1:
        raise typer.BadParameter("Linked lane runs cannot resume; start a new run.")

    from .validation import heldout_dataset

    if heldout_dataset(required=False):
        # Refuse planted directories/committed workspace links before any
        # scoring or checkpoint writes. Actual I/O still uses SafeDir.
        directory = run_dir(run_state.run_id, workspace_root)
        for path in (directory / "merge_opportunities", directory / "events/.reaped"):
            harness_record.check_view_path(workspace_root, path)
    if run_state.heldout_required:
        from .run import _assert_validation_dataset_unchanged
        from . import scoring_sandbox

        _assert_validation_dataset_unchanged(run_state, workspace_root=workspace_root)
        if scoring_sandbox.required():
            scoring_sandbox.require_supported(
                GepaConfig.load(config_path(workspace_root)), run_state.candidate_source
            )

    state = run_state
    if state.select_phase is None:
        _preselect_check(workspace_root, state)
        ctx: dict[str, Any] = {
            "pid": os.getpid(),
            "started_ms": int(time.time() * 1000),
        }
        state = _checkpoint(state, workspace_root, "promote", ctx)
        phase: str | None = "promote"
    else:
        phase = state.select_phase
        ctx = dict(state.select_context or {})
        pid = ctx.get("pid")
        if isinstance(pid, int) and pid != os.getpid() and _pid_alive(pid):
            typer.echo(
                f"Select already in flight for run {state.run_id} (phase "
                f"{phase!r}, pid {pid}); a second concurrent select is "
                "rejected (spec-er3).",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(
            f"Resuming select for run {state.run_id} from phase {phase!r} "
            "(recorded pid is this process or no longer alive)."
        )

    handlers = {
        "promote": _phase_promote,
        "journal": _phase_journal,
        "run_start_rebaseline": _phase_run_start_rebaseline,
        "refan": _phase_refan,
        "rebaseline": _phase_rebaseline,
        "emit": _phase_emit,
        "finalize": _phase_finalize,
    }
    while phase is not None:
        handler = handlers.get(phase)
        if handler is None:
            typer.echo(
                f"Run {state.run_id} has an unknown select phase {phase!r}; "
                "refusing to resume. Inspect the run state manually.",
                err=True,
            )
            raise typer.Exit(code=1)
        state, ctx, next_phase = handler(workspace_root, state, ctx)
        state = _checkpoint(
            state, workspace_root, next_phase, ctx if next_phase else None
        )
        phase = next_phase

    winner = ctx.get("winner")
    if state.status == "done":
        typer.echo(f"Select complete: run {state.run_id} is done.")
    elif winner:
        typer.echo(
            f"Select complete: promoted {winner}; lanes re-fanned for "
            f"iteration {ctx.get('new_lane_iteration')}."
        )
    else:
        typer.echo("Select complete: no promotion; lanes re-fanned.")

    from .run import _public_state

    final_path = (
        final_report_path(state.run_id, workspace_root)
        if state.status == "done"
        else None
    )
    typer.echo(
        json.dumps({"run": _public_state(state, outcomes=[], final_report=final_path)})
    )
    return state


__all__ = ["run_select"]

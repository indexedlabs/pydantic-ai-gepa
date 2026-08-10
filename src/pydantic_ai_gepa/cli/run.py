"""`gepa run` — managed external-reflection optimization loop.

This command group keeps the coding agent in the reflector role while the CLI
owns loop state. `start` evaluates minibatches until reflection is useful;
`continue` evaluates the edited component baseline or git tree against the
same mini-valset, compares against the pre-reflection baseline, and either
pauses with discard guidance or advances to the next reflection point.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Any, Literal, cast

import typer

from ..acceptance import AcceptanceComparison, compare_candidate_samples
from .candidates import (
    GitCandidateError,
    candidate_id_from_components,
    git_candidate_state,
)
from .eval import DEFAULT_FAILURE_THRESHOLD, EvalOutcome, run_eval_once
from .layout import (
    GepaConfig,
    CandidateSource,
    config_path,
    final_report_path,
    insert_repo_root_on_path,
    journal_path,
    new_run_id,
    repo_root,
    resolve_agent,
    resolve_skills,
    run_dir,
    run_state_path,
    runs_dir,
)
from .runs import ParetoLog, utc_now_iso
from .store import ComponentStore


app = typer.Typer(
    no_args_is_help=True,
    help="Start and resume a managed pause-for-reflection GEPA run.",
)

DEFAULT_STRAGGLER_TIMEOUT_SECS = 3600.0

RunStatus = Literal[
    "running",
    "paused_for_reflection",
    "paused_after_candidate_eval",
    "done",
]


@dataclass(frozen=True)
class RunState:
    run_id: str
    status: RunStatus
    max_iterations: int
    size: int
    seed: int
    next_epoch: int
    concurrency: int
    threshold: float
    acceptance_repetitions: int
    acceptance_max_repetitions: int
    acceptance_confidence: float
    acceptance_min_delta: float
    candidate_source: CandidateSource
    iterations: int
    created_at: str
    updated_at: str
    reflection_minibatch_id: str | None = None
    reflection_baseline_candidate_id: str | None = None
    reflection_baseline_commit_sha: str | None = None
    reflection_baseline_mean_score: float | None = None
    reflection_baseline_samples: tuple[float, ...] = ()
    reflection_baseline_iteration: int | None = None
    reflection_baseline_report_path: str | None = None
    reflection_baseline_report_paths: tuple[str, ...] = ()
    reflection_baseline_trace_path: str | None = None
    reflection_baseline_trace_paths: tuple[str, ...] = ()
    last_candidate_id: str | None = None
    last_minibatch_id: str | None = None
    last_mean_score: float | None = None
    last_report_path: str | None = None
    last_trace_path: str | None = None
    last_comparison: dict[str, Any] | None = None
    best_candidate_id: str | None = None
    best_commit_sha: str | None = None
    best_mean_score: float | None = None
    # Lane-run fields (spec-1do). All defaulted so run state files written
    # before lanes existed load unchanged.
    lanes: int = 0
    heartbeat_interval_secs: float = 10.0
    reflection_lease_secs: float = 1800.0
    eval_stall_timeout_secs: float = 600.0
    straggler_timeout_secs: float = DEFAULT_STRAGGLER_TIMEOUT_SECS
    journal_tail_lines: int = 20
    # Set at fan-out/re-fan: the straggler-timeout clock starts here, immune
    # to unrelated run-state saves refreshing updated_at (spec-er3).
    iteration_started_at: str | None = None
    # Select-phase resumption markers (spec-er3). ``select_phase`` is the
    # in-flight marker: non-None while a `gepa run select` is between phase
    # checkpoints; ``select_context`` carries the resumption record (pid,
    # winner, per-lane progress) so an interrupted select resumes idempotently.
    select_phase: str | None = None
    select_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "max_iterations": self.max_iterations,
            "size": self.size,
            "seed": self.seed,
            "next_epoch": self.next_epoch,
            "concurrency": self.concurrency,
            "threshold": self.threshold,
            "acceptance_repetitions": self.acceptance_repetitions,
            "acceptance_max_repetitions": self.acceptance_max_repetitions,
            "acceptance_confidence": self.acceptance_confidence,
            "acceptance_min_delta": self.acceptance_min_delta,
            "candidate_source": self.candidate_source,
            "iterations": self.iterations,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reflection_minibatch_id": self.reflection_minibatch_id,
            "reflection_baseline_candidate_id": self.reflection_baseline_candidate_id,
            "reflection_baseline_commit_sha": self.reflection_baseline_commit_sha,
            "reflection_baseline_mean_score": self.reflection_baseline_mean_score,
            "reflection_baseline_samples": list(self.reflection_baseline_samples),
            "reflection_baseline_iteration": self.reflection_baseline_iteration,
            "reflection_baseline_report_path": self.reflection_baseline_report_path,
            "reflection_baseline_report_paths": list(
                self.reflection_baseline_report_paths
            ),
            "reflection_baseline_trace_path": self.reflection_baseline_trace_path,
            "reflection_baseline_trace_paths": list(
                self.reflection_baseline_trace_paths
            ),
            "last_candidate_id": self.last_candidate_id,
            "last_minibatch_id": self.last_minibatch_id,
            "last_mean_score": self.last_mean_score,
            "last_report_path": self.last_report_path,
            "last_trace_path": self.last_trace_path,
            "last_comparison": self.last_comparison,
            "best_candidate_id": self.best_candidate_id,
            "best_commit_sha": self.best_commit_sha,
            "best_mean_score": self.best_mean_score,
            "lanes": self.lanes,
            "heartbeat_interval_secs": self.heartbeat_interval_secs,
            "reflection_lease_secs": self.reflection_lease_secs,
            "eval_stall_timeout_secs": self.eval_stall_timeout_secs,
            "straggler_timeout_secs": self.straggler_timeout_secs,
            "journal_tail_lines": self.journal_tail_lines,
            "iteration_started_at": self.iteration_started_at,
            "select_phase": self.select_phase,
            "select_context": self.select_context,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> RunState:
        return RunState(
            run_id=str(data["run_id"]),
            status=str(data["status"]),  # type: ignore[arg-type]
            max_iterations=int(data["max_iterations"]),
            size=int(data["size"]),
            seed=int(data["seed"]),
            next_epoch=int(data["next_epoch"]),
            concurrency=int(data["concurrency"]),
            threshold=float(data["threshold"]),
            acceptance_repetitions=int(data.get("acceptance_repetitions", 1)),
            acceptance_max_repetitions=int(
                data.get(
                    "acceptance_max_repetitions",
                    data.get("acceptance_repetitions", 1),
                )
            ),
            acceptance_confidence=float(data.get("acceptance_confidence", 0.9)),
            acceptance_min_delta=float(data.get("acceptance_min_delta", 0.0)),
            candidate_source=cast(
                CandidateSource, data.get("candidate_source", "components")
            ),
            iterations=int(data["iterations"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            reflection_minibatch_id=data.get("reflection_minibatch_id"),
            reflection_baseline_candidate_id=data.get(
                "reflection_baseline_candidate_id"
            ),
            reflection_baseline_commit_sha=data.get("reflection_baseline_commit_sha"),
            reflection_baseline_mean_score=(
                float(data["reflection_baseline_mean_score"])
                if data.get("reflection_baseline_mean_score") is not None
                else None
            ),
            reflection_baseline_samples=tuple(
                float(value)
                for value in data.get(
                    "reflection_baseline_samples",
                    (
                        [data["reflection_baseline_mean_score"]]
                        if data.get("reflection_baseline_mean_score") is not None
                        else []
                    ),
                )
            ),
            reflection_baseline_iteration=(
                int(data["reflection_baseline_iteration"])
                if data.get("reflection_baseline_iteration") is not None
                else None
            ),
            reflection_baseline_report_path=data.get("reflection_baseline_report_path"),
            reflection_baseline_report_paths=tuple(
                str(value)
                for value in data.get(
                    "reflection_baseline_report_paths",
                    (
                        [data["reflection_baseline_report_path"]]
                        if data.get("reflection_baseline_report_path")
                        else []
                    ),
                )
            ),
            reflection_baseline_trace_path=data.get("reflection_baseline_trace_path"),
            reflection_baseline_trace_paths=tuple(
                str(value)
                for value in data.get(
                    "reflection_baseline_trace_paths",
                    (
                        [data["reflection_baseline_trace_path"]]
                        if data.get("reflection_baseline_trace_path")
                        else []
                    ),
                )
            ),
            last_candidate_id=data.get("last_candidate_id"),
            last_minibatch_id=data.get("last_minibatch_id"),
            last_mean_score=(
                float(data["last_mean_score"])
                if data.get("last_mean_score") is not None
                else None
            ),
            last_report_path=data.get("last_report_path"),
            last_trace_path=data.get("last_trace_path"),
            last_comparison=(
                dict(data["last_comparison"])
                if isinstance(data.get("last_comparison"), dict)
                else None
            ),
            best_candidate_id=data.get("best_candidate_id"),
            best_commit_sha=data.get("best_commit_sha"),
            best_mean_score=(
                float(data["best_mean_score"])
                if data.get("best_mean_score") is not None
                else None
            ),
            lanes=int(data.get("lanes", 0)),
            heartbeat_interval_secs=float(data.get("heartbeat_interval_secs", 10.0)),
            reflection_lease_secs=float(data.get("reflection_lease_secs", 1800.0)),
            eval_stall_timeout_secs=float(data.get("eval_stall_timeout_secs", 600.0)),
            straggler_timeout_secs=float(
                data.get("straggler_timeout_secs", DEFAULT_STRAGGLER_TIMEOUT_SECS)
            ),
            journal_tail_lines=int(data.get("journal_tail_lines", 20)),
            iteration_started_at=data.get("iteration_started_at"),
            select_phase=(
                str(data["select_phase"])
                if data.get("select_phase") is not None
                else None
            ),
            select_context=(
                dict(data["select_context"])
                if isinstance(data.get("select_context"), dict)
                else None
            ),
        )

    def save(self, root: Path | None = None) -> Path:
        path = run_state_path(self.run_id, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic (tmpfile + os.replace): lane evals, select checkpoints, and
        # operator verbs all write this file; a kill mid-write must never
        # leave torn JSON for the resume logic to trip over.
        import tempfile

        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2)
                handle.write("\n")
            os.replace(tmp_name, path)
        except BaseException:
            os.unlink(tmp_name)
            raise
        return path


def _load_state(run_id: str | None) -> RunState:
    active_run_id = run_id or _latest_managed_run_id()
    if active_run_id is None:
        typer.echo("No run found. Start one with `gepa run start`.", err=True)
        raise typer.Exit(code=1)
    path = run_state_path(active_run_id)
    if not path.exists():
        typer.echo(
            f"No managed run state at {path}. Start one with `gepa run start`.",
            err=True,
        )
        raise typer.Exit(code=1)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        typer.echo(f"Run state at {path} is not a JSON object.", err=True)
        raise typer.Exit(code=1)
    return RunState.from_dict(raw)


def _latest_managed_run_id() -> str | None:
    base = runs_dir()
    if not base.is_dir():
        return None
    for candidate in sorted(
        (p.name for p in base.iterdir() if p.is_dir()), reverse=True
    ):
        if run_state_path(candidate).exists():
            return candidate
    return None


def _with_timestamp(state: RunState, **changes: Any) -> RunState:
    return replace(state, updated_at=utc_now_iso(), **changes)


def _with_last_outcome(state: RunState, outcome: EvalOutcome) -> RunState:
    summary = outcome.summary
    return _with_timestamp(
        state,
        iterations=int(summary["iterations"]),
        last_candidate_id=str(summary["candidate_id"]),
        last_minibatch_id=str(summary["minibatch_id"]),
        last_mean_score=float(summary["mean_score"]),
        last_report_path=str(summary["report_path"]),
        last_trace_path=(
            str(summary["trace_path"]) if summary.get("trace_path") else None
        ),
        best_candidate_id=state.best_candidate_id or str(summary["candidate_id"]),
        best_commit_sha=(
            state.best_commit_sha
            or (
                _summary_commit_sha(summary)
                if state.candidate_source == "git"
                else None
            )
        ),
        best_mean_score=(
            state.best_mean_score
            if state.best_mean_score is not None
            else float(summary["mean_score"])
        ),
    )


def _mark_reflection_pause(state: RunState, outcomes: list[EvalOutcome]) -> RunState:
    if not outcomes:
        raise ValueError("A reflection pause requires at least one baseline outcome.")
    first_summary = outcomes[0].summary
    baseline_samples = tuple(
        float(outcome.summary["mean_score"]) for outcome in outcomes
    )
    baseline_mean_score = sum(baseline_samples) / len(baseline_samples)
    best_candidate_id = state.best_candidate_id or str(first_summary["candidate_id"])
    best_commit_sha = state.best_commit_sha or (
        _summary_commit_sha(first_summary) if state.candidate_source == "git" else None
    )
    best_mean_score = (
        state.best_mean_score
        if state.best_mean_score is not None
        else baseline_mean_score
    )
    return _with_timestamp(
        _with_last_outcome(state, outcomes[-1]),
        status="paused_for_reflection",
        reflection_minibatch_id=str(first_summary["minibatch_id"]),
        reflection_baseline_candidate_id=str(first_summary["candidate_id"]),
        reflection_baseline_commit_sha=_summary_commit_sha(first_summary),
        reflection_baseline_mean_score=baseline_mean_score,
        reflection_baseline_samples=baseline_samples,
        reflection_baseline_iteration=int(first_summary["iterations"]),
        reflection_baseline_report_path=str(first_summary["report_path"]),
        reflection_baseline_report_paths=tuple(
            str(outcome.summary["report_path"]) for outcome in outcomes
        ),
        reflection_baseline_trace_path=(
            str(first_summary["trace_path"])
            if first_summary.get("trace_path")
            else None
        ),
        reflection_baseline_trace_paths=tuple(
            str(outcome.summary["trace_path"])
            for outcome in outcomes
            if outcome.summary.get("trace_path")
        ),
        best_candidate_id=best_candidate_id,
        best_commit_sha=best_commit_sha,
        best_mean_score=best_mean_score,
    )


def _summary_commit_sha(summary: dict[str, Any]) -> str | None:
    value = summary.get("commit_sha")
    return str(value) if value else None


def _mark_best_candidate(
    state: RunState, outcome: EvalOutcome, *, mean_score: float | None = None
) -> RunState:
    summary = outcome.summary
    return _with_timestamp(
        state,
        best_candidate_id=str(summary["candidate_id"]),
        best_commit_sha=(
            _summary_commit_sha(summary)
            if state.candidate_source == "git"
            else state.best_commit_sha
        ),
        best_mean_score=(
            float(summary["mean_score"]) if mean_score is None else mean_score
        ),
    )


def _clear_reflection_baseline(state: RunState) -> RunState:
    return _with_timestamp(
        state,
        reflection_minibatch_id=None,
        reflection_baseline_candidate_id=None,
        reflection_baseline_commit_sha=None,
        reflection_baseline_mean_score=None,
        reflection_baseline_samples=(),
        reflection_baseline_iteration=None,
        reflection_baseline_report_path=None,
        reflection_baseline_report_paths=(),
        reflection_baseline_trace_path=None,
        reflection_baseline_trace_paths=(),
    )


def _mark_done(state: RunState) -> RunState:
    return _with_timestamp(state, status="done")


def _fresh_baseline_outcome(state: RunState) -> tuple[RunState, EvalOutcome]:
    epoch = state.next_epoch
    outcome = run_eval_once(
        candidate_file=None,
        minibatch_id=None,
        size=state.size,
        seed=state.seed,
        epoch=epoch,
        run_id=state.run_id,
        concurrency=state.concurrency,
        max_iterations=state.max_iterations,
        threshold=state.threshold,
        capture_traces=True,
        candidate_source=state.candidate_source,
    )
    return _with_timestamp(state, next_epoch=epoch + 1), outcome


def _capture_reflection_baseline(
    state: RunState, first_outcome: EvalOutcome
) -> tuple[RunState, list[EvalOutcome]]:
    """Measure the stochastic baseline before yielding the tree for edits."""

    remaining_iterations = state.max_iterations - state.iterations
    affordable_repetitions = max(1, (remaining_iterations + 1) // 2)
    target_repetitions = min(state.acceptance_max_repetitions, affordable_repetitions)
    outcomes = [first_outcome]
    expected_candidate_id = str(first_outcome.summary["candidate_id"])
    minibatch_id = str(first_outcome.summary["minibatch_id"])

    while len(outcomes) < target_repetitions:
        outcome = run_eval_once(
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
        )
        if str(outcome.summary["candidate_id"]) != expected_candidate_id:
            typer.echo(
                "The baseline candidate changed while collecting repeated "
                "evaluations; refusing to compare mixed candidates.",
                err=True,
            )
            raise typer.Exit(code=1)
        outcomes.append(outcome)
        state = _with_last_outcome(state, outcome)

    return _mark_reflection_pause(state, outcomes), outcomes


def _advance_to_reflection_or_done(
    state: RunState,
) -> tuple[RunState, list[EvalOutcome]]:
    outcomes: list[EvalOutcome] = []
    state = _with_timestamp(_clear_reflection_baseline(state), status="running")
    while state.iterations < state.max_iterations:
        state, outcome = _fresh_baseline_outcome(state)
        outcomes.append(outcome)
        state = _with_last_outcome(state, outcome)

        if state.iterations >= state.max_iterations:
            return _mark_done(state), outcomes

        if outcome.n_failures > 0:
            state, baseline_outcomes = _capture_reflection_baseline(state, outcome)
            outcomes.extend(baseline_outcomes[1:])
            return state, outcomes

    return _mark_done(state), outcomes


def _evaluate_reflected_candidate(
    state: RunState,
) -> tuple[RunState, list[EvalOutcome], dict[str, Any]]:
    if state.reflection_minibatch_id is None:
        typer.echo(
            "Run is not waiting on a reflection minibatch; use `gepa run status`.",
            err=True,
        )
        raise typer.Exit(code=1)
    if not state.reflection_baseline_samples:
        typer.echo("Run state is missing reflection baseline samples.", err=True)
        raise typer.Exit(code=1)

    max_candidate_samples = min(
        len(state.reflection_baseline_samples),
        state.max_iterations - state.iterations,
    )
    if max_candidate_samples < 1:
        typer.echo(
            "No evaluation budget remains for the reflected candidate.", err=True
        )
        raise typer.Exit(code=70)
    initial_candidate_samples = min(state.acceptance_repetitions, max_candidate_samples)

    outcomes: list[EvalOutcome] = []
    candidate_samples: list[float] = []
    candidate_id: str | None = None
    comparison_result: AcceptanceComparison | None = None
    while len(candidate_samples) < max_candidate_samples:
        outcome = run_eval_once(
            candidate_file=None,
            minibatch_id=state.reflection_minibatch_id,
            size=state.size,
            seed=state.seed,
            epoch=state.next_epoch,
            run_id=state.run_id,
            concurrency=state.concurrency,
            max_iterations=state.max_iterations,
            threshold=state.threshold,
            capture_traces=True,
            candidate_source=state.candidate_source,
        )
        state = _with_last_outcome(state, outcome)
        outcomes.append(outcome)
        current_candidate_id = str(outcome.summary["candidate_id"])
        if candidate_id is None:
            candidate_id = current_candidate_id
        elif current_candidate_id != candidate_id:
            typer.echo(
                "The reflected candidate changed while collecting repeated "
                "evaluations; refusing to compare mixed candidates.",
                err=True,
            )
            raise typer.Exit(code=1)
        candidate_samples.append(float(outcome.summary["mean_score"]))

        if len(candidate_samples) < initial_candidate_samples:
            continue
        comparison_result = compare_candidate_samples(
            state.reflection_baseline_samples[: len(candidate_samples)],
            candidate_samples,
            confidence=state.acceptance_confidence,
            min_delta=state.acceptance_min_delta,
        )
        if comparison_result.verdict != "inconclusive":
            break

    assert comparison_result is not None
    first_outcome = outcomes[0]
    last_outcome = outcomes[-1]
    recommendation = {
        "accepted": "keep_and_advance",
        "rejected": "discard_or_revise",
        "equivalent": "discard_no_material_change",
        "inconclusive": "inconclusive_revise_or_end",
    }[comparison_result.verdict]
    comparison = {
        "minibatch_id": state.reflection_minibatch_id,
        "baseline_candidate_id": state.reflection_baseline_candidate_id,
        "baseline_commit_sha": state.reflection_baseline_commit_sha,
        "baseline_iteration": state.reflection_baseline_iteration,
        "baseline_mean_score": comparison_result.baseline_mean,
        "baseline_samples": list(comparison_result.baseline_samples),
        "baseline_report_path": state.reflection_baseline_report_path,
        "baseline_report_paths": list(state.reflection_baseline_report_paths),
        "baseline_trace_path": state.reflection_baseline_trace_path,
        "baseline_trace_paths": list(state.reflection_baseline_trace_paths),
        "candidate_id": first_outcome.summary["candidate_id"],
        "candidate_commit_sha": first_outcome.summary.get("commit_sha"),
        "candidate_iteration": first_outcome.summary["iterations"],
        "candidate_mean_score": comparison_result.candidate_mean,
        "candidate_samples": list(comparison_result.candidate_samples),
        "candidate_report_path": last_outcome.summary["report_path"],
        "candidate_report_paths": [
            outcome.summary["report_path"] for outcome in outcomes
        ],
        "candidate_trace_path": last_outcome.summary["trace_path"],
        "candidate_trace_paths": [
            outcome.summary["trace_path"]
            for outcome in outcomes
            if outcome.summary.get("trace_path")
        ],
        **comparison_result.to_dict(),
        "recommendation": recommendation,
    }
    if state.candidate_source == "git" and state.reflection_baseline_commit_sha:
        comparison["discard_command"] = (
            f"git reset --hard {state.reflection_baseline_commit_sha}"
        )
    state = _with_timestamp(state, last_comparison=comparison)
    return state, outcomes, comparison


def _current_baseline_candidate_id(
    candidate_source: CandidateSource = "components",
    *,
    active_run_id: str | None = None,
) -> str:
    """Return the candidate id for the current component files or git tree."""

    cfg = GepaConfig.load(config_path())
    insert_repo_root_on_path()
    if candidate_source == "git":
        try:
            state = git_candidate_state(
                exclude_paths=[run_dir(active_run_id)] if active_run_id else []
            )
        except GitCandidateError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        return state.candidate_id

    agent = resolve_agent(cfg)
    skills_fs = resolve_skills(cfg)
    components = ComponentStore().effective_candidate(agent, skills_fs=skills_fs)
    return candidate_id_from_components(components)


def _write_final_report(
    state: RunState, *, overshoot: int | None = None, root: Path | None = None
) -> tuple[Path, str]:
    rows = ParetoLog(state.run_id, root).iter_rows()
    path = final_report_path(state.run_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# GEPA Run Final Report",
        "",
        f"- run_id: {state.run_id}",
        f"- status: {state.status}",
        f"- iterations: {state.iterations}/{state.max_iterations}",
        f"- pareto_log: {ParetoLog(state.run_id, root).path}",
    ]
    if overshoot:
        lines.append(
            f"- budget_overshoot: {overshoot} eval row(s) beyond "
            f"--max-iterations (in-flight lane evals; bounded by "
            f"lanes x --acceptance-max-repetitions, pydanticaigepa-dec-msy)"
        )
    if rows:
        best = max(rows, key=lambda row: row.mean_score)
        latest = rows[-1]
        lines.extend(
            [
                f"- best_candidate_id: {best.candidate_id}",
                f"- best_mean_score: {best.mean_score:.6f}",
                f"- latest_candidate_id: {latest.candidate_id}",
                f"- latest_mean_score: {latest.mean_score:.6f}",
            ]
        )
    if state.best_candidate_id:
        lines.append(f"- accepted_best_candidate_id: {state.best_candidate_id}")
    if state.best_commit_sha:
        lines.append(f"- accepted_best_commit_sha: {state.best_commit_sha}")
        if state.candidate_source == "git":
            lines.append(
                f"- best_restore_command: git checkout {state.best_commit_sha}"
            )
    if state.last_comparison:
        comparison = state.last_comparison
        lines.extend(
            [
                "",
                "## Last Candidate Comparison",
                "",
                f"- minibatch_id: {comparison['minibatch_id']}",
                f"- baseline_mean_score: {comparison['baseline_mean_score']:.6f}",
                f"- candidate_mean_score: {comparison['candidate_mean_score']:.6f}",
                f"- delta: {comparison['delta']:.6f}",
                f"- verdict: {comparison.get('verdict', 'unknown')}",
                f"- recommendation: {comparison['recommendation']}",
            ]
        )
        if "lower_bound" in comparison and "upper_bound" in comparison:
            lines.extend(
                [
                    f"- confidence: {comparison['confidence']:.3f}",
                    f"- confidence_interval: "
                    f"[{comparison['lower_bound']:.6f}, "
                    f"{comparison['upper_bound']:.6f}]",
                ]
            )
    if rows:
        lines.extend(["", "## History", ""])
        for row in rows[-10:]:
            lines.append(
                f"- {row.timestamp}: {row.candidate_id} "
                f"mean={row.mean_score:.6f} minibatch={row.minibatch_id} "
                f"status={row.status}"
            )

    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return path, text


def _public_state(
    state: RunState,
    *,
    outcomes: list[EvalOutcome],
    final_report: Path | None = None,
) -> dict[str, Any]:
    payload = state.to_dict()
    payload["state_path"] = str(run_state_path(state.run_id))
    payload["final_report_path"] = str(final_report) if final_report else None
    if state.status == "done":
        payload["next_command"] = None
    elif state.lanes > 0:
        payload["next_command"] = f"gepa next --wait --run-id {state.run_id}"
    else:
        payload["next_command"] = f"gepa run continue --run-id {state.run_id}"
    payload["evaluations_this_call"] = [outcome.summary for outcome in outcomes]
    return payload


def _emit_status(
    state: RunState,
    *,
    outcomes: list[EvalOutcome],
    final_report: Path | None = None,
    final_report_text: str | None = None,
) -> None:
    if state.status == "paused_for_reflection":
        editable_surface = (
            "source/artifacts and commit the result"
            if state.candidate_source == "git"
            else "components or source"
        )
        typer.echo(
            "Paused for reflection. Inspect the report and trace file, edit "
            f"{editable_surface}, then run:"
        )
        typer.echo(f"  gepa run continue --run-id {state.run_id}")
        typer.echo(f"Report: {state.reflection_baseline_report_path}")
        typer.echo(f"Trace: {state.reflection_baseline_trace_path}")
    elif state.status == "paused_after_candidate_eval":
        comparison = state.last_comparison or {}
        verdict = comparison.get("verdict", "rejected")
        if verdict == "inconclusive":
            typer.echo(
                "Candidate comparison remains inconclusive after the configured "
                "repetitions. Revise the candidate or restore the baseline, then run:"
            )
        elif verdict == "equivalent":
            typer.echo(
                "Candidate is equivalent within the configured practical delta. "
                "Restore the baseline or revise the candidate, then run:"
            )
        else:
            typer.echo(
                "Candidate did not beat the reflection baseline. Recommendation: "
                "discard or revise the edits, then run:"
            )
        typer.echo(f"  gepa run continue --run-id {state.run_id}")
        if comparison:
            typer.echo(
                f"Baseline {comparison['baseline_mean_score']:.6f}; "
                f"candidate {comparison['candidate_mean_score']:.6f}; "
                f"delta {comparison['delta']:.6f}."
            )
            if "lower_bound" in comparison and "upper_bound" in comparison:
                typer.echo(
                    f"{comparison['confidence']:.0%} interval "
                    f"[{comparison['lower_bound']:.6f}, "
                    f"{comparison['upper_bound']:.6f}]; "
                    f"verdict {verdict}."
                )
            typer.echo(f"Candidate report: {comparison['candidate_report_path']}")
            typer.echo(f"Candidate trace: {comparison['candidate_trace_path']}")
            if comparison.get("discard_command"):
                typer.echo(
                    "To discard the git candidate and restore the reflection "
                    "baseline, run:"
                )
                typer.echo(f"  {comparison['discard_command']}")
    elif state.status == "done":
        typer.echo("Run complete.")
        if final_report_text:
            typer.echo(final_report_text.rstrip())
    else:
        typer.echo(f"Run status: {state.status}")

    typer.echo(
        json.dumps(
            {"run": _public_state(state, outcomes=outcomes, final_report=final_report)}
        )
    )


def _validate_max_iterations(max_iterations: int) -> None:
    if max_iterations < 1:
        typer.echo("--max-iterations must be >= 1.", err=True)
        raise typer.Exit(code=2)


def _validate_acceptance_options(
    *,
    repetitions: int,
    max_repetitions: int,
    confidence: float,
    min_delta: float,
) -> None:
    if repetitions < 1:
        typer.echo("--acceptance-repetitions must be >= 1.", err=True)
        raise typer.Exit(code=2)
    if max_repetitions < repetitions:
        typer.echo(
            "--acceptance-max-repetitions must be >= --acceptance-repetitions.",
            err=True,
        )
        raise typer.Exit(code=2)
    if not 0.0 < confidence < 1.0:
        typer.echo("--acceptance-confidence must be between 0 and 1.", err=True)
        raise typer.Exit(code=2)
    if min_delta < 0.0:
        typer.echo("--acceptance-min-delta must be >= 0.", err=True)
        raise typer.Exit(code=2)


@app.command("start")
def start(
    max_iterations: int = typer.Option(
        100,
        "--max-iterations",
        help="Total evaluation-row budget for this managed run.",
    ),
    size: int = typer.Option(
        10, "--size", help="Number of cases in each sampled mini-valset."
    ),
    seed: int = typer.Option(0, "--seed", help="Deterministic minibatch seed."),
    epoch: int = typer.Option(0, "--epoch", help="Initial minibatch epoch."),
    concurrency: int | None = typer.Option(
        None,
        "--concurrency",
        help="Max parallel agent calls during evaluation. Defaults to --size.",
    ),
    threshold: float = typer.Option(
        DEFAULT_FAILURE_THRESHOLD,
        "--threshold",
        help="Score below which a case requires reflection.",
    ),
    acceptance_repetitions: int = typer.Option(
        3,
        "--acceptance-repetitions",
        help=(
            "Initial repeated evaluations per candidate on the saved mini-valset. "
            "Use more than one for stochastic pipelines."
        ),
    ),
    acceptance_max_repetitions: int | None = typer.Option(
        None,
        "--acceptance-max-repetitions",
        help=(
            "Maximum repetitions used when the initial comparison is inconclusive. "
            "Defaults to --acceptance-repetitions."
        ),
    ),
    acceptance_confidence: float = typer.Option(
        0.9,
        "--acceptance-confidence",
        help="Confidence level for the candidate delta interval.",
    ),
    acceptance_min_delta: float = typer.Option(
        0.0,
        "--acceptance-min-delta",
        help="Smallest practical score improvement required for acceptance.",
    ),
    candidate_source: str | None = typer.Option(
        None,
        "--candidate-source",
        help="Override gepa.toml candidate_source for this run: components or git.",
    ),
    lanes: int = typer.Option(
        0,
        "--lanes",
        help="Number of parallel reflection lanes (git candidate mode only). 0 keeps the synchronous single-path loop.",
    ),
    heartbeat_interval_secs: float = typer.Option(
        10.0,
        "--heartbeat-interval-secs",
        help="Lane runs: heartbeat refresh interval for background lane evals.",
    ),
    reflection_lease_secs: float = typer.Option(
        1800.0,
        "--reflection-lease-secs",
        help="Lane runs: dispatch lease expiry; a leased lane that never reaches `gepa lane continue` is stalled after this.",
    ),
    eval_stall_timeout_secs: float = typer.Option(
        600.0,
        "--eval-stall-timeout-secs",
        help="Lane runs: a lane eval whose heartbeat is older than this (with a dead pid) is stalled.",
    ),
    straggler_timeout_secs: float = typer.Option(
        DEFAULT_STRAGGLER_TIMEOUT_SECS,
        "--straggler-timeout-secs",
        help="Lane runs: selection fires once every lane resolves or this timeout elapses.",
    ),
    journal_tail_lines: int = typer.Option(
        20,
        "--journal-tail-lines",
        help="Lane runs: how many journal entries each reflection packet carries.",
    ),
) -> None:
    """Start a managed GEPA run and pause at the first reflection point."""
    _validate_max_iterations(max_iterations)
    if lanes < 0:
        typer.echo("--lanes must be >= 0.", err=True)
        raise typer.Exit(code=2)
    resolved_max_repetitions = (
        acceptance_repetitions
        if acceptance_max_repetitions is None
        else acceptance_max_repetitions
    )
    resolved_concurrency = size if concurrency is None else concurrency
    _validate_acceptance_options(
        repetitions=acceptance_repetitions,
        max_repetitions=resolved_max_repetitions,
        confidence=acceptance_confidence,
        min_delta=acceptance_min_delta,
    )
    if candidate_source not in {None, "components", "git"}:
        typer.echo("--candidate-source must be 'components' or 'git'.", err=True)
        raise typer.Exit(code=2)
    cfg = GepaConfig.load(config_path())
    active_candidate_source = cast(
        CandidateSource, candidate_source or cfg.candidate_source
    )
    if lanes > 0:
        if active_candidate_source != "git":
            typer.echo(
                "--lanes requires git candidate mode (component-mode lanes share "
                "one process-global agent and are out of scope, spec-1do).",
                err=True,
            )
            raise typer.Exit(code=2)
        # Lane branches are always cut from a clean commit; a dirty primary
        # tree is rejected (spec-1do constraint). The journal is tracked
        # bookkeeping the CLI itself appends to — exclude it like select does
        # (one dirtiness definition across verbs).
        from .lanes import ensure_worktrees_ignored, worktrees_root

        workspace_root = repo_root()
        ensure_worktrees_ignored(workspace_root)
        try:
            primary_state = git_candidate_state(
                workspace_root,
                exclude_paths=[
                    runs_dir(workspace_root),
                    worktrees_root(workspace_root),
                    journal_path(workspace_root),
                ],
            )
        except GitCandidateError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        if primary_state.dirty:
            typer.echo(
                "`gepa run start --lanes` requires a clean primary tree; "
                "commit or stash your changes first (lane branches are cut "
                "from a clean commit).",
                err=True,
            )
            raise typer.Exit(code=1)
        # One active lane run per workspace (dec-jh6): an existing lane run
        # that never reached `done` still owns its lane refs and events.
        from .layout import is_run_id

        for entry in sorted(runs_dir(workspace_root).iterdir()):
            if not (entry.is_dir() and is_run_id(entry.name)):
                continue
            prior_state_path = entry / "state.json"
            if not prior_state_path.exists():
                continue
            try:
                prior = RunState.from_dict(
                    json.loads(prior_state_path.read_text(encoding="utf-8"))
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            if prior.lanes > 0 and prior.status != "done":
                typer.echo(
                    f"Lane run {prior.run_id} is still active "
                    f"(status {prior.status}); finish it (`gepa run select`) "
                    "or abandon it before starting another lane run.",
                    err=True,
                )
                raise typer.Exit(code=1)
    run_id = new_run_id()
    run_dir(run_id).mkdir(parents=True, exist_ok=True)
    now = utc_now_iso()
    state = RunState(
        run_id=run_id,
        status="running",
        max_iterations=max_iterations,
        size=size,
        seed=seed,
        next_epoch=epoch,
        concurrency=resolved_concurrency,
        threshold=threshold,
        acceptance_repetitions=acceptance_repetitions,
        acceptance_max_repetitions=resolved_max_repetitions,
        acceptance_confidence=acceptance_confidence,
        acceptance_min_delta=acceptance_min_delta,
        candidate_source=active_candidate_source,
        iterations=0,
        created_at=now,
        updated_at=now,
        lanes=lanes,
        heartbeat_interval_secs=heartbeat_interval_secs,
        reflection_lease_secs=reflection_lease_secs,
        eval_stall_timeout_secs=eval_stall_timeout_secs,
        straggler_timeout_secs=straggler_timeout_secs,
        journal_tail_lines=journal_tail_lines,
    )
    state.save()
    state, outcomes = _advance_to_reflection_or_done(state)
    state.save()
    if lanes > 0 and state.status == "paused_for_reflection":
        # Lane branches are cut from the CURRENT best commit — re-check the
        # primary hasn't moved since the (possibly minutes-old) clean check.
        fresh_state = git_candidate_state(
            workspace_root,
            exclude_paths=[
                runs_dir(workspace_root),
                worktrees_root(workspace_root),
                journal_path(workspace_root),
            ],
        )
        if (
            fresh_state.commit_sha != state.reflection_baseline_commit_sha
            or fresh_state.dirty
        ):
            # The baseline was measured against a HEAD that has since moved;
            # advance again so lanes branch off the code actually scored.
            state = _with_timestamp(state, status="running")
            state.save()
            state, extra_outcomes = _advance_to_reflection_or_done(state)
            outcomes.extend(extra_outcomes)
            state.save()
    if lanes > 0 and state.status == "paused_for_reflection":
        # Fan out: worktrees + packets + lane_ready events. The run-level
        # status returns to "running" — lanes carry the reflection pause.
        from .lanes import fan_out_lanes

        fan_out_lanes(state, repo_root())
        state = _with_timestamp(state, status="running")
        state = replace(state, iteration_started_at=state.updated_at)
        state.save()

    final_path: Path | None = None
    final_text: str | None = None
    if state.status == "done":
        final_path, final_text = _write_final_report(state)
        if lanes > 0:
            # A lane run can finish during the baseline advance; the
            # orchestrator loop terminates on run_done, so emit it here too.
            from .events import EventDraft, emit

            emit(
                state.run_id,
                "run",
                EventDraft(
                    type="run_done",
                    lane=None,
                    payload={"final_report_path": str(final_path)},
                ),
                root=workspace_root,
            )
    _emit_status(
        state, outcomes=outcomes, final_report=final_path, final_report_text=final_text
    )


@app.command("continue")
def continue_(
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Managed run id. Omit to use the latest run with a state file.",
    ),
) -> None:
    """Resume after reflection edits and advance to the next pause or completion."""
    state = _load_state(run_id)
    if state.lanes > 0:
        typer.echo(
            "`gepa run continue` does not drive lane runs. Evaluate a lane with "
            "`gepa lane continue <lane>` and commit the iteration with "
            "`gepa run select`.",
            err=True,
        )
        raise typer.Exit(code=1)
    if state.status == "done":
        final_path, final_text = _write_final_report(state)
        _emit_status(
            state, outcomes=[], final_report=final_path, final_report_text=final_text
        )
        return

    outcomes: list[EvalOutcome] = []
    if (
        state.status == "paused_after_candidate_eval"
        and state.reflection_baseline_candidate_id is not None
        and _current_baseline_candidate_id(
            state.candidate_source, active_run_id=state.run_id
        )
        == state.reflection_baseline_candidate_id
    ):
        typer.echo(
            "Current components match the reflection baseline; discarding the "
            "losing candidate and advancing."
        )
        state, outcomes = _advance_to_reflection_or_done(state)
        state.save()
        final_path = None
        final_text = None
        if state.status == "done":
            final_path, final_text = _write_final_report(state)
        _emit_status(
            state,
            outcomes=outcomes,
            final_report=final_path,
            final_report_text=final_text,
        )
        return

    if state.reflection_minibatch_id is not None:
        state, comparison_outcomes, comparison = _evaluate_reflected_candidate(state)
        outcomes.extend(comparison_outcomes)

        if comparison["improved"]:
            state = _mark_best_candidate(
                state,
                comparison_outcomes[-1],
                mean_score=float(comparison["candidate_mean_score"]),
            )
        if state.iterations >= state.max_iterations:
            state = _mark_done(state)
        elif comparison["improved"]:
            state, advanced_outcomes = _advance_to_reflection_or_done(state)
            outcomes.extend(advanced_outcomes)
        else:
            state = _with_timestamp(state, status="paused_after_candidate_eval")
    else:
        state, outcomes = _advance_to_reflection_or_done(state)

    state.save()
    final_path = None
    final_text = None
    if state.status == "done":
        final_path, final_text = _write_final_report(state)
    _emit_status(
        state, outcomes=outcomes, final_report=final_path, final_report_text=final_text
    )


@app.command("select")
def select(
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Managed run id. Omit to use the latest run with a state file.",
    ),
) -> None:
    """Commit one lockstep lane iteration: pick the winner and re-fan lanes.

    Select is the single sequential authority for lane runs
    (pydanticaigepa-spec-er3). It consumes the memoized lane verdicts (never
    re-deriving them), invalidates stragglers, promotes the best accepted lane
    to the run's best, journals every non-promoted lane before deleting its
    branch, emits merge_opportunity for accepted lanes with disjoint diffs
    (never auto-merges), enforces the evaluation budget (run_done + overshoot
    in the final report), and — when budget remains — re-fans every lane onto
    the new best with a fresh shared baseline and lane_ready events.

    Select records phase progress (promote, journal, re-fan, re-baseline,
    emit) in run state; a second invocation while one is in flight is
    rejected, and an interrupted select resumes idempotently from the recorded
    phase.

    Exit codes: 0 selection committed; 1 not a lane run / not due / already
    in flight / already done.
    """
    from .select import run_select

    run_select(run_id)


@app.command("status")
def status(
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Managed run id. Omit to use the latest run with a state file.",
    ),
) -> None:
    """Print the managed run state as JSON (lane runs include the lane board)."""
    state = _load_state(run_id)
    final_path = final_report_path(state.run_id) if state.status == "done" else None
    payload: dict[str, Any] = {
        "run": _public_state(state, outcomes=[], final_report=final_path)
    }
    if state.lanes > 0:
        # Consumer verbs run the lazy reaper first (dec-pm3).
        from .lanes import load_all_lane_states, reaper_pass_for_run

        workspace_root = repo_root()
        reaper_pass_for_run(workspace_root, state)
        payload["lanes"] = [
            lane_state.to_dict()
            for lane_state in load_all_lane_states(workspace_root, state.run_id)
        ]
    typer.echo(json.dumps(payload))


__all__ = ["app"]

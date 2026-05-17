"""`gepa run` — managed external-reflection optimization loop.

This command group keeps the coding agent in the reflector role while the CLI
owns loop state. `start` evaluates minibatches until reflection is useful;
`continue` evaluates the agent's edited components against the same
mini-valset, compares against the pre-reflection baseline, and either pauses
with discard guidance or advances to the next reflection point.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Literal

import typer

from .candidates import candidate_id_from_components
from .eval import DEFAULT_FAILURE_THRESHOLD, EvalOutcome, run_eval_once
from .layout import (
    GepaConfig,
    config_path,
    final_report_path,
    insert_repo_root_on_path,
    new_run_id,
    resolve_agent,
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
    iterations: int
    created_at: str
    updated_at: str
    reflection_minibatch_id: str | None = None
    reflection_baseline_candidate_id: str | None = None
    reflection_baseline_mean_score: float | None = None
    reflection_baseline_iteration: int | None = None
    reflection_baseline_report_path: str | None = None
    reflection_baseline_trace_path: str | None = None
    last_candidate_id: str | None = None
    last_minibatch_id: str | None = None
    last_mean_score: float | None = None
    last_report_path: str | None = None
    last_trace_path: str | None = None
    last_comparison: dict[str, Any] | None = None

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
            "iterations": self.iterations,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reflection_minibatch_id": self.reflection_minibatch_id,
            "reflection_baseline_candidate_id": self.reflection_baseline_candidate_id,
            "reflection_baseline_mean_score": self.reflection_baseline_mean_score,
            "reflection_baseline_iteration": self.reflection_baseline_iteration,
            "reflection_baseline_report_path": self.reflection_baseline_report_path,
            "reflection_baseline_trace_path": self.reflection_baseline_trace_path,
            "last_candidate_id": self.last_candidate_id,
            "last_minibatch_id": self.last_minibatch_id,
            "last_mean_score": self.last_mean_score,
            "last_report_path": self.last_report_path,
            "last_trace_path": self.last_trace_path,
            "last_comparison": self.last_comparison,
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
            iterations=int(data["iterations"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            reflection_minibatch_id=data.get("reflection_minibatch_id"),
            reflection_baseline_candidate_id=data.get(
                "reflection_baseline_candidate_id"
            ),
            reflection_baseline_mean_score=(
                float(data["reflection_baseline_mean_score"])
                if data.get("reflection_baseline_mean_score") is not None
                else None
            ),
            reflection_baseline_iteration=(
                int(data["reflection_baseline_iteration"])
                if data.get("reflection_baseline_iteration") is not None
                else None
            ),
            reflection_baseline_report_path=data.get("reflection_baseline_report_path"),
            reflection_baseline_trace_path=data.get("reflection_baseline_trace_path"),
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
        )

    def save(self) -> Path:
        path = run_state_path(self.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
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
    )


def _mark_reflection_pause(state: RunState, outcome: EvalOutcome) -> RunState:
    summary = outcome.summary
    return _with_timestamp(
        _with_last_outcome(state, outcome),
        status="paused_for_reflection",
        reflection_minibatch_id=str(summary["minibatch_id"]),
        reflection_baseline_candidate_id=str(summary["candidate_id"]),
        reflection_baseline_mean_score=float(summary["mean_score"]),
        reflection_baseline_iteration=int(summary["iterations"]),
        reflection_baseline_report_path=str(summary["report_path"]),
        reflection_baseline_trace_path=(
            str(summary["trace_path"]) if summary.get("trace_path") else None
        ),
    )


def _clear_reflection_baseline(state: RunState) -> RunState:
    return _with_timestamp(
        state,
        reflection_minibatch_id=None,
        reflection_baseline_candidate_id=None,
        reflection_baseline_mean_score=None,
        reflection_baseline_iteration=None,
        reflection_baseline_report_path=None,
        reflection_baseline_trace_path=None,
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
    )
    return _with_timestamp(state, next_epoch=epoch + 1), outcome


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
            return _mark_reflection_pause(state, outcome), outcomes

    return _mark_done(state), outcomes


def _evaluate_reflected_candidate(
    state: RunState,
) -> tuple[RunState, EvalOutcome, dict[str, Any]]:
    if state.reflection_minibatch_id is None:
        typer.echo(
            "Run is not waiting on a reflection minibatch; use `gepa run status`.",
            err=True,
        )
        raise typer.Exit(code=1)
    if state.reflection_baseline_mean_score is None:
        typer.echo("Run state is missing the reflection baseline score.", err=True)
        raise typer.Exit(code=1)

    baseline_score = state.reflection_baseline_mean_score
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
    )
    state = _with_last_outcome(state, outcome)
    candidate_score = float(outcome.summary["mean_score"])
    improved = candidate_score > baseline_score
    comparison = {
        "minibatch_id": state.reflection_minibatch_id,
        "baseline_candidate_id": state.reflection_baseline_candidate_id,
        "baseline_iteration": state.reflection_baseline_iteration,
        "baseline_mean_score": baseline_score,
        "baseline_report_path": state.reflection_baseline_report_path,
        "baseline_trace_path": state.reflection_baseline_trace_path,
        "candidate_id": outcome.summary["candidate_id"],
        "candidate_iteration": outcome.summary["iterations"],
        "candidate_mean_score": candidate_score,
        "candidate_report_path": outcome.summary["report_path"],
        "candidate_trace_path": outcome.summary["trace_path"],
        "delta": candidate_score - baseline_score,
        "improved": improved,
        "recommendation": "keep_and_advance" if improved else "discard_or_revise",
    }
    state = _with_timestamp(state, last_comparison=comparison)
    return state, outcome, comparison


def _current_baseline_candidate_id() -> str:
    """Return the candidate id for the currently confirmed component files."""

    cfg = GepaConfig.load(config_path())
    insert_repo_root_on_path()
    agent = resolve_agent(cfg)
    components = ComponentStore().effective_candidate(agent)
    return candidate_id_from_components(components)


def _write_final_report(state: RunState) -> tuple[Path, str]:
    rows = ParetoLog(state.run_id).iter_rows()
    path = final_report_path(state.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# GEPA Run Final Report",
        "",
        f"- run_id: {state.run_id}",
        f"- status: {state.status}",
        f"- iterations: {state.iterations}/{state.max_iterations}",
        f"- pareto_log: {ParetoLog(state.run_id).path}",
    ]
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
                f"- recommendation: {comparison['recommendation']}",
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
    payload["next_command"] = (
        None if state.status == "done" else f"gepa run continue --run-id {state.run_id}"
    )
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
        typer.echo(
            "Paused for reflection. Inspect the report and trace file, edit "
            "components or source, then run:"
        )
        typer.echo(f"  gepa run continue --run-id {state.run_id}")
        typer.echo(f"Report: {state.reflection_baseline_report_path}")
        typer.echo(f"Trace: {state.reflection_baseline_trace_path}")
    elif state.status == "paused_after_candidate_eval":
        comparison = state.last_comparison or {}
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
            typer.echo(f"Candidate report: {comparison['candidate_report_path']}")
            typer.echo(f"Candidate trace: {comparison['candidate_trace_path']}")
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
    concurrency: int = typer.Option(
        5, "--concurrency", help="Max parallel agent calls during evaluation."
    ),
    threshold: float = typer.Option(
        DEFAULT_FAILURE_THRESHOLD,
        "--threshold",
        help="Score below which a case requires reflection.",
    ),
) -> None:
    """Start a managed GEPA run and pause at the first reflection point."""
    _validate_max_iterations(max_iterations)
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
        concurrency=concurrency,
        threshold=threshold,
        iterations=0,
        created_at=now,
        updated_at=now,
    )
    state.save()
    state, outcomes = _advance_to_reflection_or_done(state)
    state.save()

    final_path: Path | None = None
    final_text: str | None = None
    if state.status == "done":
        final_path, final_text = _write_final_report(state)
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
    if state.status == "done":
        final_path, final_text = _write_final_report(state)
        _emit_status(
            state, outcomes=[], final_report=final_path, final_report_text=final_text
        )
        return

    outcomes: list[EvalOutcome] = []
    comparison_outcome: EvalOutcome | None = None
    if (
        state.status == "paused_after_candidate_eval"
        and state.reflection_baseline_candidate_id is not None
        and _current_baseline_candidate_id() == state.reflection_baseline_candidate_id
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
        state, comparison_outcome, comparison = _evaluate_reflected_candidate(state)
        outcomes.append(comparison_outcome)

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


@app.command("status")
def status(
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Managed run id. Omit to use the latest run with a state file.",
    ),
) -> None:
    """Print the managed run state as JSON."""
    state = _load_state(run_id)
    final_path = final_report_path(state.run_id) if state.status == "done" else None
    typer.echo(
        json.dumps({"run": _public_state(state, outcomes=[], final_report=final_path)})
    )


__all__ = ["app"]

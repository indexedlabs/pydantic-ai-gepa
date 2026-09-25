"""Replay a killed single-path continuation from its paid ledger prefix.

The continuation checkpoint fixes the candidate, options and ledger offset.
Replaying the controller from that checkpoint preserves its original sampling
and budget decisions, including validation and subsequent baseline advancement.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import typer

from ..evaluation import EvaluationRecord
from ..types import RolloutOutput
from .eval import EvalOutcome, _trace_file_path
from .layout import repo_root, run_dir
from .runs import ParetoLog, ParetoRow

if TYPE_CHECKING:
    from .run import RunState


@dataclass
class Replay:
    state: RunState
    rows: list[tuple[int, ParetoRow]]
    cursor: int = 0
    gate_cursor: int = 0


_replay: ContextVar[Replay | None] = ContextVar("reflector_replay", default=None)


def state_for_save(state: RunState) -> RunState:
    """The controller saves only at terminal pauses during a continuation."""
    replay = _replay.get()
    if replay is not None:
        if replay.cursor < len(replay.rows):
            raise typer.BadParameter(
                "Continuation ended before recovering all paid evaluations; "
                "its checkpoint has been preserved."
            )
        return replace(state, continuation=None)
    return state


def remember_comparison(state: RunState, comparison: dict[str, Any]) -> RunState:
    if comparison.get("verdict") is not None:
        return replace(state, last_reflector_comparison=dict(comparison))
    return state


def _row_outcome(
    state: RunState, row: ParetoRow, iteration: int, threshold: float
) -> EvalOutcome:
    validation = row.extra.get("dataset_role") == "validation"
    eval_id = row.extra.get("eval_id", "")
    stem = f"{iteration:04d}-{eval_id}-{row.candidate_id}"
    report = run_dir(state.run_id) / "reports" / f"{stem}.md"
    trace = _trace_file_path(
        run_id=state.run_id,
        iteration=iteration,
        eval_id=eval_id,
        candidate_id=row.candidate_id,
        minibatch_id=row.minibatch_id,
    )
    # The paid row may precede artifact writes when the process is killed.
    # Preserve existing artifacts and expose only paths which actually exist.
    report_path = report if not validation and report.exists() else None
    trace_path = trace if not validation and trace.exists() else None
    scores = row.per_case_scores
    if validation and state.acceptance_paired_min_cases is not None:
        from .validation import read_validation_evidence
        from .run import _validation_schedule

        scores = read_validation_evidence(
            state.validation_dataset_path,
            project_root=repo_root(),
            run_id=f"{state.run_id}:eval:{eval_id}",
            identity={"candidate_id": row.candidate_id, "eval_id": eval_id},
        )
        if not scores and _validation_schedule(state)[0] == 1:
            raise typer.BadParameter(
                "Private paired validation evidence is missing for a paid "
                "evaluation; restore the harness evidence before continuing."
            )
    records = [
        EvaluationRecord(case_id=key, score=value, feedback=None, payload={})
        for key, value in scores.items()
    ]
    errors = row.extra.get("evaluation_errors") or []
    for error in errors:
        records.append(
            EvaluationRecord(
                case_id=error.get("case_id", "withheld"),
                score=0.0,
                feedback=None,
                payload={
                    "output": RolloutOutput(
                        result=None,
                        success=False,
                        error_message=error.get(
                            "error_message", "Evaluation infrastructure failure"
                        ),
                        error_kind=error.get("error_kind"),
                    )
                },
            )
        )
    summary = {
        "candidate_id": row.candidate_id,
        "commit_sha": row.commit_sha,
        "candidate_source": state.candidate_source,
        "minibatch_id": row.minibatch_id,
        "run_id": state.run_id,
        "eval_id": eval_id,
        "mean_score": row.mean_score,
        "iterations": iteration,
        "report_path": str(report_path) if report_path else None,
        "trace_path": str(trace_path) if trace_path else None,
        "n_failures": sum(value < threshold for value in row.per_case_scores.values()),
        "dataset_role": "validation" if validation else "training",
        "evaluation_outcome": row.extra.get("outcome", "valid"),
        "evaluation_errors": errors,
        "selectable": row.extra.get("selectable", True),
    }
    return EvalOutcome(records, summary, report_path, trace_path)


def durable_eval(evaluate: Callable[..., EvalOutcome], **kwargs: Any) -> EvalOutcome:
    replay = _replay.get()
    if replay is None:
        return evaluate(**kwargs)
    if not kwargs.get("write_pareto", True):
        return _durable_gate(replay, evaluate, kwargs)
    if replay.cursor >= len(replay.rows):
        return evaluate(**kwargs)
    iteration, row = replay.rows[replay.cursor]
    role = kwargs.get("dataset_role", "training")
    minibatch = kwargs.get("minibatch_id")
    if row.extra.get("dataset_role", "training") != role or (
        minibatch is not None and row.minibatch_id != minibatch
    ):
        raise typer.BadParameter(
            "Interrupted continuation does not match its paid evaluation ledger; "
            "restore its original candidate and continuation options."
        )
    replay.cursor += 1
    return _row_outcome(replay.state, row, iteration, kwargs["threshold"])


def _durable_gate(
    replay: Replay, evaluate: Callable[..., EvalOutcome], kwargs: dict[str, Any]
) -> EvalOutcome:
    """Gates have no ledger rows; checkpoint their training-only sample data."""
    checkpoint = dict(replay.state.continuation or {})
    gates = list(checkpoint.get("gates", []))
    if replay.gate_cursor < len(gates):
        saved = gates[replay.gate_cursor]
        records = []
        for record in saved["records"]:
            payload: dict[str, Any] = {"side_info": record.get("side_info", {})}
            if record.get("error"):
                payload["output"] = RolloutOutput(**record["error"])
            records.append(
                EvaluationRecord(
                    record["case_id"], record["score"], record["feedback"], payload
                )
            )
        summary = saved["summary"]
        outcome = EvalOutcome(
            records,
            summary,
            Path(summary["report_path"]) if summary.get("report_path") else None,
            Path(summary["trace_path"]) if summary.get("trace_path") else None,
        )
    else:
        outcome = evaluate(**kwargs)
        records = []
        for record in outcome.records:
            output = record.payload.get("output")
            error = None
            if isinstance(output, RolloutOutput) and not output.success:
                error = {
                    "result": None,
                    "success": False,
                    "error_message": output.error_message,
                    "error_kind": output.error_kind,
                }
            info = record.payload.get("side_info")
            if not isinstance(info, dict):
                info = record.payload.get("metric_side_info")
            if not isinstance(info, dict):
                info = getattr(record.payload.get("trajectory"), "metric_side_info", {})
            info = info if isinstance(info, dict) else {}
            records.append(
                {
                    "side_info": {
                        key: info[key]
                        for key in ("scores", "selectable", "infrastructure_valid")
                        if key in info
                    },
                    "case_id": record.case_id,
                    "score": record.score,
                    "feedback": record.feedback,
                    "error": error,
                }
            )
        gates.append({"summary": outcome.summary, "records": records})
        checkpoint["gates"] = gates
        replay.state = replace(replay.state, continuation=checkpoint)
        # This is an in-flight checkpoint, not a terminal controller save.
        token = _replay.set(None)
        try:
            replay.state.save()
        finally:
            _replay.reset(token)
    replay.gate_cursor += 1
    return outcome


def continue_run(
    run_id: str | None,
    gate_case: list[str],
    reflector_epoch: int | None,
    execute: Callable[..., None],
) -> None:
    from .reflector import run_lock
    from .run import (
        _current_baseline_candidate_id,
        _emit_status,
        _load_state,
        _validate_gate_cases,
        _assert_validation_dataset_unchanged,
    )

    initial = _load_state(run_id)
    with run_lock(initial.run_id):
        state = _load_state(initial.run_id)
        epoch = state.reflector["epoch"]
        if reflector_epoch is not None and reflector_epoch != epoch:
            typer.echo(
                f"Stale reflector epoch {reflector_epoch}; current epoch is {epoch}. "
                f"Use `gepa run resume --run-id {state.run_id}` for a fresh packet.",
                err=True,
            )
            raise typer.Exit(code=2)
        if state.lanes > 0 or state.status == "done":
            execute(state.run_id, gate_case)
            return
        if state.validation_dataset_path is not None:
            _assert_validation_dataset_unchanged(state)
        if gate_case and state.reflection_minibatch_id is not None:
            _validate_gate_cases(state, gate_case)
        candidate = _current_baseline_candidate_id(
            state.candidate_source, active_run_id=state.run_id
        )
        comparison = state.last_comparison or state.last_reflector_comparison or {}
        if (
            state.continuation is None
            and comparison.get("candidate_id") == candidate
            and comparison.get("verdict") is not None
            and (
                comparison.get("minibatch_id") == state.reflection_minibatch_id
                or comparison.get("improved")
            )
            and (
                candidate != state.reflection_baseline_candidate_id
                or comparison.get("improved")
            )
        ):
            typer.echo("Candidate already scored; re-issuing the recorded result.")
            _emit_status(state, outcomes=[])
            return
        ledger = ParetoLog(state.run_id)
        rows = ledger.iter_rows()
        checkpoint = state.continuation
        if checkpoint is None:
            checkpoint = {
                "candidate_id": candidate,
                "gate_case": gate_case,
                "ledger_offset": len(rows),
            }
            state = replace(state, continuation=checkpoint)
            if state.best_validation_per_case_scores and state.validation_dataset_path:
                from .validation import write_validation_evidence

                # A final save writes incumbent evidence before state.json.
                # Keep the original evidence until that state transition commits.
                write_validation_evidence(
                    state.validation_dataset_path,
                    project_root=repo_root(),
                    run_id=f"{state.run_id}:continuation:{candidate}",
                    identity=state._validation_evidence_identity(),
                    scores=state.best_validation_per_case_scores,
                )
            state.save()
        elif (
            checkpoint["candidate_id"] != candidate
            or checkpoint["gate_case"] != gate_case
        ):
            raise typer.BadParameter(
                "An interrupted continuation is pending. Restore candidate "
                f"{checkpoint['candidate_id']} and reuse its --gate-case options "
                "to finish the paid comparison."
            )
        budget = state.gate_consumed_iterations
        pending: list[tuple[int, ParetoRow]] = []
        for index, row in enumerate(rows):
            if row.extra.get("row_scope", "acceptance") not in {
                "acceptance",
                "validation",
            }:
                continue
            budget += 1
            if index >= checkpoint["ledger_offset"]:
                if row.candidate_id != candidate:
                    raise typer.BadParameter(
                        "Paid continuation rows contain another candidate."
                    )
                pending.append((budget, row))
        token = _replay.set(Replay(state, pending))
        try:
            execute(state.run_id, gate_case)
        except (typer.Exit, typer.BadParameter):
            saved = _load_state(state.run_id)
            if (
                saved.continuation is not None
                and not saved.continuation.get("gates")
                and len(ledger.iter_rows()) == checkpoint["ledger_offset"]
            ):
                # A refused, unpaid attempt must not fence off corrective edits.
                cleanup_token = _replay.set(None)
                try:
                    replace(saved, continuation=None).save()
                finally:
                    _replay.reset(cleanup_token)
            raise
        finally:
            _replay.reset(token)

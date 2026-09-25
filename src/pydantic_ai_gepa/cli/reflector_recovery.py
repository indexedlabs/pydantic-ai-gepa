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

from .validation import public_echo, private_evaluation

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
    budget_adjustment: int = 0


_replay: ContextVar[Replay | None] = ContextVar("reflector_replay", default=None)


class CandidateChanged(typer.Exit):
    """A paid sample belongs to a different tree than this continuation."""

    def __init__(self) -> None:
        public_echo(
            "Candidate changed during continuation; abandoning its checkpoint. "
            "Paid evaluations remain charged. Run continue again for the current tree.",
            err=True,
        )
        super().__init__(code=1)


def state_for_replay(state: RunState) -> RunState:
    replay = _replay.get()
    return (
        replace(state, iterations=state.iterations + replay.budget_adjustment)
        if replay
        else state
    )


def state_for_save(state: RunState) -> RunState:
    """The controller saves only at terminal pauses during a continuation."""
    replay = _replay.get()
    if replay is not None:
        budget_exhausted = (state.last_comparison or {}).get("reason_code") in {
            "cost_budget_exhausted",
            "candidate_budget_exhausted",
            "baseline_budget_exhausted",
            "validation_budget_exhausted",
        }
        if replay.cursor < len(replay.rows) and not budget_exhausted:
            raise typer.BadParameter(
                "Continuation ended before recovering all paid evaluations; "
                "its checkpoint has been preserved."
            )
        return replace(
            state,
            continuation=None,
            iterations=ParetoLog(state.run_id).count_budget_rows()
            + state.gate_consumed_iterations,
        )
    return state


def remember_comparison(state: RunState, comparison: dict[str, Any]) -> RunState:
    if comparison.get("verdict") is not None:
        return replace(state, last_reflector_comparison=dict(comparison))
    return state


def _row_outcome(
    state: RunState, row: ParetoRow, iteration: int, threshold: float
) -> EvalOutcome:
    validation = row.extra.get("dataset_role") == "validation"
    if validation:
        from .validation import check_heldout_pin

        check_heldout_pin(repo_root(), state.run_id)
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
    # Discarded foreign rows shift replay's budget ordinal. The eval ID still
    # identifies the original artifacts, regardless of their saved ordinal.
    if eval_id and not validation:
        if not report.exists():
            report = next(
                report.parent.glob(f"*-{eval_id}-{row.candidate_id}.md"), report
            )
        if not trace.exists():
            trace = next(
                trace.parent.glob(f"*-{eval_id}-{row.candidate_id}.jsonl"), trace
            )
    # The paid row may precede artifact writes when the process is killed.
    # Preserve existing artifacts and expose only paths which actually exist.
    report_path = report if not validation and report.exists() else None
    trace_path = trace if not validation and trace.exists() else None
    scores = row.per_case_scores
    if validation and _paired_validation(state):
        from .validation import read_validation_evidence

        scores = read_validation_evidence(
            state.validation_dataset_path,
            project_root=repo_root(),
            run_id=f"{state.run_id}:eval:{eval_id}",
            identity={"candidate_id": row.candidate_id, "eval_id": eval_id},
        )
        if not scores:
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


@private_evaluation
def durable_eval(evaluate: Callable[..., EvalOutcome], **kwargs: Any) -> EvalOutcome:
    replay = _replay.get()
    if replay is None:
        return evaluate(**kwargs)
    if not kwargs.get("write_pareto", True):
        return _durable_gate(replay, evaluate, kwargs)
    if replay.cursor >= len(replay.rows):
        if kwargs.get("dataset_role") == "validation" and _paired_validation(
            replay.state
        ):
            kwargs["persist_validation_replay"] = True
        outcome = evaluate(**kwargs)
        _check_candidate(replay, outcome)
        return outcome
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
        _check_candidate(replay, outcome)
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


def _paired_validation(state: RunState) -> bool:
    if (
        state.lanes
        or state.acceptance_paired_min_cases is None
        or not state.validation_dataset_path
    ):
        return False
    from .run import _validation_schedule

    return _validation_schedule(state)[0] == 1


def _check_candidate(replay: Replay, outcome: EvalOutcome) -> None:
    assert replay.state.continuation is not None
    if outcome.summary["candidate_id"] != replay.state.continuation["candidate_id"]:
        raise CandidateChanged()


def cleanup_continuation(state: RunState, root: Path | None = None) -> None:
    """Retire private replay snapshots only after the state transition commits."""
    if (
        not state.continuation
        or not state.validation_dataset_path
        or state.acceptance_paired_min_cases is None
    ):
        return
    from .validation import validation_evidence_path

    try:
        if not _paired_validation(state):
            return
        rows = ParetoLog(state.run_id, root).iter_rows()[
            state.continuation["ledger_offset"] :
        ]
        keys = [
            f"{state.run_id}:eval:{row.extra['eval_id']}"
            for row in rows
            if row.extra.get("dataset_role") == "validation"
            and row.extra.get("eval_id")
        ]
        keys.append(f"{state.run_id}:continuation:{state.continuation['candidate_id']}")
        for key in keys:
            path = validation_evidence_path(
                state.validation_dataset_path,
                project_root=root or repo_root(),
                run_id=key,
            )
            path.unlink(missing_ok=True)
    except (OSError, ValueError, typer.BadParameter) as exc:
        public_echo(
            f"Warning: private replay evidence cleanup failed ({type(exc).__name__}).",
            err=True,
        )


def after_state_save(root: Path | None = None) -> None:
    replay = _replay.get()
    if replay is not None:
        cleanup_continuation(replay.state, root)


def abandon_continuation(state: RunState, *, reason: str) -> RunState:
    if state.continuation is None:
        return state
    if state.heldout_required:
        state = state.restore_validation_evidence()
    from .lanes import _append_journal
    from .runs import utc_now_iso

    used = ParetoLog(state.run_id).count_budget_rows() + state.gate_consumed_iterations
    _append_journal(
        repo_root(),
        {
            "kind": "continuation_abandoned",
            "run_id": state.run_id,
            "timestamp": utc_now_iso(),
            "candidate_id": state.continuation["candidate_id"],
            "reason": reason,
            "budget_used": used,
        },
    )
    abandoned = replace(state, continuation=None, iterations=used)
    from .harness import abandoning_scoring_tree

    token = _replay.set(None)
    try:
        with abandoning_scoring_tree():
            abandoned.save()
    finally:
        _replay.reset(token)
    cleanup_continuation(state)
    return abandoned


def _paid_prefix(
    state: RunState, rows: list[ParetoRow]
) -> tuple[RunState, list[tuple[int, ParetoRow]], int]:
    """Foreign candidates end the reusable prefix; later rows stay charged.

    Remember discarded indices so samples newly paid after recovery remain
    reusable if that recovery is interrupted again.
    """
    assert state.continuation is not None
    checkpoint = dict(state.continuation)
    discarded = set(checkpoint.get("discarded_rows", []))
    matching = []
    ended = False
    for index, row in enumerate(rows):
        if index < checkpoint["ledger_offset"] or row.extra.get(
            "row_scope", "acceptance"
        ) not in {"acceptance", "validation"}:
            continue
        if index in discarded:
            continue
        if row.candidate_id != checkpoint["candidate_id"]:
            ended = True
        if ended:
            discarded.add(index)
        else:
            matching.append(row)
    if discarded != set(checkpoint.get("discarded_rows", [])):
        checkpoint["discarded_rows"] = sorted(discarded)
        state = replace(state, continuation=checkpoint)
        state.save()
    adjustment = len(discarded)
    pending = [
        (state.iterations + adjustment + i + 1, row) for i, row in enumerate(matching)
    ]
    return state, pending, adjustment


def continue_run(
    run_id: str | None,
    gate_case: list[str],
    reflector_epoch: int | None,
    execute: Callable[..., None],
    *,
    locked: bool = False,
) -> None:
    from .reflector import already_scored, run_lock
    from .run import (
        _current_baseline_candidate_id,
        _emit_status,
        _load_state,
        _validate_gate_cases,
        _assert_validation_dataset_unchanged,
    )

    initial = _load_state(run_id)
    from contextlib import nullcontext

    with nullcontext() if locked else run_lock(initial.run_id):
        state = _load_state(initial.run_id).restore_validation_evidence()
        epoch = state.reflector["epoch"]
        if reflector_epoch is not None and reflector_epoch != epoch:
            public_echo(
                f"Stale reflector epoch {reflector_epoch}; current epoch is {epoch}. "
                f"Use `gepa run resume --run-id {state.run_id}` for a fresh packet.",
                err=True,
            )
            raise typer.Exit(code=2)
        if state.lanes > 0 or state.status == "done":
            execute(state.run_id, gate_case)
            return
        if state.heldout_required:
            _assert_validation_dataset_unchanged(state)
        if gate_case and state.reflection_minibatch_id is not None:
            _validate_gate_cases(state, gate_case)
        candidate = _current_baseline_candidate_id(
            state.candidate_source, active_run_id=state.run_id
        )
        if already_scored(state, candidate):
            public_echo("Candidate already scored; re-issuing the recorded result.")
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
            state = replace(
                state,
                continuation=checkpoint,
                iterations=ledger.count_budget_rows() + state.gate_consumed_iterations,
            )
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
                "to finish the paid comparison, or use "
                f"`gepa run resume --run-id {state.run_id} --abandon-continuation` to start over."
            )
        state, pending, adjustment = _paid_prefix(state, rows)
        token = _replay.set(Replay(state, pending, budget_adjustment=adjustment))
        try:
            execute(state.run_id, gate_case)
        except Exception as exc:
            if not locked and not isinstance(exc, (typer.Exit, typer.BadParameter)):
                raise
            from .harness import StaleNomination

            saved = _load_state(state.run_id)
            changed = isinstance(exc, (CandidateChanged, StaleNomination)) or any(
                row.candidate_id != checkpoint["candidate_id"]
                for row in ledger.iter_rows()[checkpoint["ledger_offset"] :]
            )
            try:
                changed = (
                    changed
                    or _current_baseline_candidate_id(
                        state.candidate_source, active_run_id=state.run_id
                    )
                    != checkpoint["candidate_id"]
                )
            except Exception:
                # A broken import must not prevent retiring an unpaid checkpoint.
                pass
            if saved.continuation is not None and (
                changed
                or (
                    not saved.continuation.get("gates")
                    and len(ledger.iter_rows()) == checkpoint["ledger_offset"]
                )
            ):
                abandon_continuation(
                    saved, reason="candidate_changed" if changed else "unpaid_refusal"
                )
            raise
        finally:
            _replay.reset(token)

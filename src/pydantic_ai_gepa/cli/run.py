"""`gepa run` — managed external-reflection optimization loop.

This command group keeps the coding agent in the reflector role while the CLI
owns loop state. Held-out runs nominate through `continue`; the orchestrator's
harness evaluates the training gate and held-out confirmation. Training-only
runs keep their synchronous continuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import typer

from ..acceptance import AcceptanceComparison, compare_candidate_samples
from ..evaluation_health import (
    EvaluationInfrastructureFailure,
    evaluation_infrastructure_failures,
)
from .candidates import (
    GitCandidateError,
    candidate_id_from_components,
    git_candidate_state,
)
from .eval import DEFAULT_FAILURE_THRESHOLD, EvalOutcome, run_eval_once
from .layout import (
    GepaConfig,
    CandidateSource,
    candidate_identity_exempt_paths,
    config_path,
    final_report_path,
    insert_repo_root_on_path,
    new_run_id,
    repo_root,
    resolve_agent,
    resolve_skills,
    run_dir,
    run_state_path,
    runs_dir,
)
from .reflector import default_reflector, write_packet
from .reflector_recovery import (
    after_state_save,
    continue_run,
    state_for_replay,
    durable_eval,
    remember_comparison,
    state_for_save,
)
from .runs import MinibatchStore, ParetoLog, utc_now_iso
from .store import ComponentStore
from .validation import (
    public_echo,
    harness_environment,
    heldout_dataset,
    private_evaluation,
    heldout_identity,
    check_heldout_pin,
    pin_heldout,
    validation_dataset_path,
)


app = typer.Typer(
    no_args_is_help=True,
    help="Start and resume a managed pause-for-reflection GEPA run.",
)

DEFAULT_STRAGGLER_TIMEOUT_SECS = 3600.0

RunStatus = Literal[
    "running",
    "paused_for_reflection",
    "paused_after_candidate_eval",
    "paused_after_infrastructure_error",
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
    max_token_cost: float | None = None
    reflection_minibatch_id: str | None = None
    reflection_baseline_candidate_id: str | None = None
    reflection_baseline_commit_sha: str | None = None
    reflection_baseline_mean_score: float | None = None
    reflection_baseline_samples: tuple[float, ...] = ()
    reflection_baseline_eval_ids: tuple[str, ...] = ()
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
    next_parent_candidate_id: str | None = None
    next_parent_commit_sha: str | None = None
    best_mean_score: float | None = None
    heldout_required: bool = False
    validation_seeded: bool = False
    validation_evaluations: int = 0
    # Harness-only memory; never serialized or restored by public state reads.
    validation_dataset_path: str | None = None
    validation_dataset_digest: str | None = None
    # Vector runs preserve their initial scored identity for generic periodic
    # incumbent-vs-run-start re-baselines. The mapping is intentionally
    # opaque to the optimizer apart from the documented identity fields.
    run_start_baseline: dict[str, Any] | None = None
    accepted_promotion_count: int = 0
    # Lane-run fields (spec-1do). All defaulted so run state files written
    # before lanes existed load unchanged.
    lanes: int = 0
    heartbeat_interval_secs: float = 10.0
    reflection_lease_secs: float = 1800.0
    eval_stall_timeout_secs: float = 600.0
    straggler_timeout_secs: float = DEFAULT_STRAGGLER_TIMEOUT_SECS
    journal_tail_lines: int = 20
    stall_threshold: int = 5
    iterations_since_acceptance: int = 0
    # Gate comparisons do not write Pareto rows, but a rejected gate is still
    # a consumed managed-run iteration. Keep this independently so budget
    # accounting never depends on emitting forbidden gate rows.
    gate_consumed_iterations: int = 0
    # Set at fan-out/re-fan: the straggler-timeout clock starts here, immune
    # to unrelated run-state saves refreshing updated_at (spec-er3).
    iteration_started_at: str | None = None
    # Select-phase resumption markers (spec-er3). ``select_phase`` is the
    # in-flight marker: non-None while a `gepa run select` is between phase
    # checkpoints; ``select_context`` carries the resumption record (pid,
    # winner, per-lane progress) so an interrupted select resumes idempotently.
    select_phase: str | None = None
    select_context: dict[str, Any] | None = None
    infrastructure_retry_minibatch_id: str | None = None
    acceptance_paired_min_cases: int | None = None
    best_validation_samples: tuple[float, ...] = ()
    best_validation_per_case_scores: dict[str, float] = field(default_factory=dict)

    reflector: dict[str, Any] = field(default_factory=default_reflector)
    last_reflector_comparison: dict[str, Any] | None = None
    continuation: dict[str, Any] | None = None
    project_root: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reflector": self.reflector,
            "last_reflector_comparison": self.last_reflector_comparison,
            "continuation": self.continuation,
            "project_root": self.project_root,
            "run_id": self.run_id,
            "status": self.status,
            "max_iterations": self.max_iterations,
            "max_token_cost": self.max_token_cost,
            "size": self.size,
            "seed": self.seed,
            "next_epoch": self.next_epoch,
            "concurrency": self.concurrency,
            "threshold": self.threshold,
            "acceptance_repetitions": self.acceptance_repetitions,
            "acceptance_max_repetitions": self.acceptance_max_repetitions,
            "acceptance_confidence": self.acceptance_confidence,
            "acceptance_min_delta": self.acceptance_min_delta,
            "acceptance_paired_min_cases": self.acceptance_paired_min_cases,
            "best_validation_samples": list(self.best_validation_samples),
            "candidate_source": self.candidate_source,
            "iterations": self.iterations,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reflection_minibatch_id": self.reflection_minibatch_id,
            "reflection_baseline_candidate_id": self.reflection_baseline_candidate_id,
            "reflection_baseline_commit_sha": self.reflection_baseline_commit_sha,
            "reflection_baseline_mean_score": self.reflection_baseline_mean_score,
            "reflection_baseline_samples": list(self.reflection_baseline_samples),
            "reflection_baseline_eval_ids": list(self.reflection_baseline_eval_ids),
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
            "next_parent_candidate_id": self.next_parent_candidate_id,
            "next_parent_commit_sha": self.next_parent_commit_sha,
            "best_mean_score": self.best_mean_score,
            "validation_seeded": self.validation_seeded,
            "validation_evaluations": self.validation_evaluations,
            "heldout_required": self.heldout_required,
            "run_start_baseline": self.run_start_baseline,
            "accepted_promotion_count": self.accepted_promotion_count,
            "lanes": self.lanes,
            "heartbeat_interval_secs": self.heartbeat_interval_secs,
            "reflection_lease_secs": self.reflection_lease_secs,
            "eval_stall_timeout_secs": self.eval_stall_timeout_secs,
            "straggler_timeout_secs": self.straggler_timeout_secs,
            "journal_tail_lines": self.journal_tail_lines,
            "stall_threshold": self.stall_threshold,
            "iterations_since_acceptance": self.iterations_since_acceptance,
            "gate_consumed_iterations": self.gate_consumed_iterations,
            "iteration_started_at": self.iteration_started_at,
            "select_phase": self.select_phase,
            "select_context": self.select_context,
            "infrastructure_retry_minibatch_id": self.infrastructure_retry_minibatch_id,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> RunState:
        if any(
            data.get(key) is not None
            for key in (
                "validation_dataset_path",
                "validation_dataset_digest",
                "best_validation_per_case_scores",
            )
        ):
            raise typer.BadParameter(
                "Legacy held-out state is unsafe for reflection; start a new run from a clean "
                "workspace with GEPA_HELDOUT_DATASET in the harness's environment."
            )
        return RunState(
            reflector=dict(data.get("reflector") or default_reflector()),
            last_reflector_comparison=data.get("last_reflector_comparison"),
            continuation=data.get("continuation"),
            project_root=data.get("project_root"),
            run_id=str(data["run_id"]),
            status=str(data["status"]),  # type: ignore[arg-type]
            max_iterations=int(data["max_iterations"]),
            max_token_cost=data.get("max_token_cost"),
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
            acceptance_paired_min_cases=data.get("acceptance_paired_min_cases"),
            best_validation_samples=tuple(
                float(x) for x in data.get("best_validation_samples", ())
            ),
            best_validation_per_case_scores={
                str(k): float(v)
                for k, v in data.get("best_validation_per_case_scores", {}).items()
            },
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
            reflection_baseline_eval_ids=tuple(
                str(value) for value in data.get("reflection_baseline_eval_ids", ())
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
            next_parent_candidate_id=data.get("next_parent_candidate_id"),
            next_parent_commit_sha=data.get("next_parent_commit_sha"),
            best_mean_score=(
                float(data["best_mean_score"])
                if data.get("best_mean_score") is not None
                else None
            ),
            validation_seeded=bool(data.get("validation_seeded", False)),
            validation_evaluations=int(data.get("validation_evaluations", 0)),
            heldout_required=bool(data.get("heldout_required", False)),
            run_start_baseline=(
                dict(data["run_start_baseline"])
                if isinstance(data.get("run_start_baseline"), dict)
                else None
            ),
            accepted_promotion_count=int(data.get("accepted_promotion_count", 0)),
            lanes=int(data.get("lanes", 0)),
            heartbeat_interval_secs=float(data.get("heartbeat_interval_secs", 10.0)),
            reflection_lease_secs=float(data.get("reflection_lease_secs", 1800.0)),
            eval_stall_timeout_secs=float(data.get("eval_stall_timeout_secs", 600.0)),
            straggler_timeout_secs=float(
                data.get("straggler_timeout_secs", DEFAULT_STRAGGLER_TIMEOUT_SECS)
            ),
            journal_tail_lines=int(data.get("journal_tail_lines", 20)),
            stall_threshold=int(data.get("stall_threshold", 5)),
            iterations_since_acceptance=int(data.get("iterations_since_acceptance", 0)),
            gate_consumed_iterations=int(data.get("gate_consumed_iterations", 0)),
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
            infrastructure_retry_minibatch_id=data.get(
                "infrastructure_retry_minibatch_id"
            ),
        )

    def _validation_evidence_identity(self) -> dict[str, Any]:
        return {
            "candidate_id": self.best_candidate_id,
            "dataset_digest": self.validation_dataset_digest,
            "samples": list(self.best_validation_samples),
        }

    def restore_validation_evidence(self, root: Path | None = None) -> RunState:
        from .validation import read_validation_evidence

        if not self.heldout_required:
            return self
        path, digest = check_heldout_pin(root or repo_root(), self.run_id)
        self = replace(
            self, validation_dataset_path=path, validation_dataset_digest=digest
        )
        if self.acceptance_paired_min_cases is None:
            return replace(self, best_validation_per_case_scores={})
        scores = read_validation_evidence(
            self.validation_dataset_path,
            project_root=root or repo_root(),
            run_id=self.run_id,
            identity=self._validation_evidence_identity(),
        )
        if not scores and self.continuation:
            scores = read_validation_evidence(
                self.validation_dataset_path,
                project_root=root or repo_root(),
                run_id=f"{self.run_id}:continuation:{self.continuation['candidate_id']}",
                identity=self._validation_evidence_identity(),
            )
        return replace(self, best_validation_per_case_scores=scores)

    def save(self, root: Path | None = None) -> Path:
        from .validation import write_validation_evidence
        from .harness import check_scoring_tree

        check_scoring_tree()
        if self.best_validation_per_case_scores and self.validation_dataset_path:
            write_validation_evidence(
                self.validation_dataset_path,
                project_root=root or repo_root(),
                run_id=self.run_id,
                identity=self._validation_evidence_identity(),
                scores=self.best_validation_per_case_scores,
            )
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
                json.dump(state_for_save(self).to_dict(), handle, indent=2)
                handle.write("\n")
            os.replace(tmp_name, path)
        except BaseException:
            os.unlink(tmp_name)
            raise
        after_state_save(root)
        if self.lanes == 0 and self.status != "running":
            try:
                packet_path = write_packet(self.run_id, root)
                if self.next_parent_candidate_id:
                    from .front import write_parent_packet

                    write_parent_packet(
                        packet_path, state_for_save(self), root or repo_root()
                    )
            except Exception as exc:
                public_echo(
                    f"Warning: state saved but reflector packet could not be refreshed "
                    f"({type(exc).__name__}). Regenerate it with `gepa run resume --run-id {self.run_id}`.",
                    err=True,
                )
        return path


def _load_state(run_id: str | None) -> RunState:
    active_run_id = run_id or _latest_managed_run_id()
    if active_run_id is None:
        public_echo("No run found. Start one with `gepa run start`.", err=True)
        raise typer.Exit(code=1)
    path = run_state_path(active_run_id)
    if not path.exists():
        public_echo(
            f"No managed run state at {path}. Start one with `gepa run start`.",
            err=True,
        )
        raise typer.Exit(code=1)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        public_echo(f"Run state at {path} is not a JSON object.", err=True)
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


def _outcome_infrastructure_failures(
    outcome: EvalOutcome,
) -> tuple[EvaluationInfrastructureFailure, ...]:
    return evaluation_infrastructure_failures(outcome.records)


def _infrastructure_failure_comparison(
    state: RunState,
    outcomes: list[EvalOutcome],
    *,
    phase: Literal["baseline", "candidate"],
    failures: tuple[EvaluationInfrastructureFailure, ...],
    valid_samples: tuple[float, ...] = (),
) -> dict[str, Any]:
    """Build the persisted, non-selectable result for a failed comparison."""

    first_summary = outcomes[0].summary
    reports = [
        str(outcome.summary["report_path"])
        for outcome in outcomes
        if outcome.summary.get("report_path")
    ]
    traces = [
        str(outcome.summary["trace_path"])
        for outcome in outcomes
        if outcome.summary.get("trace_path")
    ]
    comparison: dict[str, Any] = {
        "outcome": "infrastructure_failure",
        "selectable": False,
        "verdict": None,
        "improved": False,
        "phase": phase,
        "reason_code": "required_rollout_failed",
        "retryable": state.iterations < state.max_iterations,
        "recommendation": (
            "retry_after_infrastructure_recovery"
            if state.iterations < state.max_iterations
            else "stop_budget_exhausted"
        ),
        "minibatch_id": str(first_summary["minibatch_id"]),
        "candidate_id": str(first_summary["candidate_id"]),
        "candidate_commit_sha": first_summary.get("commit_sha"),
        "candidate_report_path": reports[-1] if reports else None,
        "candidate_report_paths": reports,
        "candidate_trace_path": traces[-1] if traces else None,
        "candidate_trace_paths": traces,
        "valid_samples_before_failure": list(valid_samples),
        "evaluation_error_count": len(failures),
        "evaluation_errors": (
            list(first_summary.get("evaluation_errors") or [])
            if first_summary.get("dataset_role") == "validation"
            else [failure.to_dict() for failure in failures]
        ),
    }
    if phase == "candidate":
        comparison.update(
            {
                "baseline_candidate_id": state.reflection_baseline_candidate_id,
                "baseline_commit_sha": state.reflection_baseline_commit_sha,
                "baseline_iteration": state.reflection_baseline_iteration,
                "baseline_samples": list(state.reflection_baseline_samples),
                "baseline_report_path": state.reflection_baseline_report_path,
                "baseline_report_paths": list(state.reflection_baseline_report_paths),
                "baseline_trace_path": state.reflection_baseline_trace_path,
                "baseline_trace_paths": list(state.reflection_baseline_trace_paths),
            }
        )
    return comparison


def _pause_after_infrastructure_failure(
    state: RunState,
    outcomes: list[EvalOutcome],
    *,
    phase: Literal["baseline", "candidate"],
    failures: tuple[EvaluationInfrastructureFailure, ...],
    valid_samples: tuple[float, ...] = (),
) -> tuple[RunState, dict[str, Any]]:
    comparison = _infrastructure_failure_comparison(
        state,
        outcomes,
        phase=phase,
        failures=failures,
        valid_samples=valid_samples,
    )
    if phase == "baseline":
        state = _with_timestamp(
            _clear_reflection_baseline(state),
            infrastructure_retry_minibatch_id=str(outcomes[-1].summary["minibatch_id"]),
        )
    status: RunStatus = (
        "paused_after_infrastructure_error"
        if state.iterations < state.max_iterations
        else "done"
    )
    return (
        _with_timestamp(
            state,
            status=status,
            last_comparison=comparison,
        ),
        comparison,
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
    run_start_baseline = state.run_start_baseline
    if run_start_baseline is None:
        vector_keys = []
        for outcome in outcomes:
            raw_vector = outcome.summary.get("vector_record")
            if isinstance(raw_vector, dict) and isinstance(raw_vector.get("key"), dict):
                vector_keys.append(dict(raw_vector["key"]))
        if vector_keys:
            run_start_baseline = {
                "candidate_id": str(first_summary["candidate_id"]),
                "commit_sha": _summary_commit_sha(first_summary),
                "component_hashes": dict(first_summary.get("component_hashes") or {}),
                "minibatch_id": str(first_summary["minibatch_id"]),
                "vector_record_keys": vector_keys,
            }
    return _with_timestamp(
        _with_last_outcome(state, outcomes[-1]),
        status="paused_for_reflection",
        reflection_minibatch_id=str(first_summary["minibatch_id"]),
        reflection_baseline_candidate_id=str(first_summary["candidate_id"]),
        reflection_baseline_commit_sha=_summary_commit_sha(first_summary),
        reflection_baseline_mean_score=baseline_mean_score,
        reflection_baseline_samples=baseline_samples,
        reflection_baseline_eval_ids=tuple(
            str(outcome.summary["eval_id"]) for outcome in outcomes
        ),
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
        run_start_baseline=run_start_baseline,
        infrastructure_retry_minibatch_id=None,
        last_comparison=None,
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


def _held_out_validation_enabled(root: Path | None = None) -> bool:
    return heldout_dataset(required=False) is not None


def _validation_dataset_identity(root: Path | None = None) -> tuple[str, str]:
    return heldout_identity((root or repo_root()).resolve())


def _assert_validation_dataset_unchanged(
    state: RunState,
    *,
    workspace_root: Path | None = None,
    candidate_root: Path | None = None,
) -> None:
    primary_root = (workspace_root or repo_root()).resolve()
    configured_path, _ = check_heldout_pin(primary_root, state.run_id)
    if candidate_root is not None:
        primary_config = config_path(primary_root).resolve()
        if primary_config.is_relative_to(primary_root):
            candidate_config = candidate_root / primary_config.relative_to(primary_root)
            if candidate_config.is_file():
                GepaConfig.load(candidate_config)
    validation_dataset_path(
        configured_path, project_root=primary_root, candidate_root=candidate_root
    )


def _evaluate_validation_candidate(
    state: RunState,
    *,
    candidate_root: Path | None = None,
    workspace_root: Path | None = None,
    lane: str | None = None,
    vector_incumbent_hash: str | None = None,
    vector_repetition: int | None = None,
) -> tuple[RunState, EvalOutcome]:
    """Score the current candidate on held-out validation without reflection artifacts."""

    _assert_validation_dataset_unchanged(
        state,
        workspace_root=workspace_root,
        candidate_root=candidate_root,
    )
    front = None
    if lane is None and state.lanes == 0:
        from .front import ValidationFront

        front = ValidationFront(workspace_root or repo_root(), state.run_id)
        front.preflight()
    outcome = private_evaluation(durable_eval)(
        run_eval_once,
        spend_state=state,
        candidate_file=None,
        minibatch_id=None,
        size=state.size,
        seed=state.seed,
        epoch=0,
        run_id=state.run_id,
        concurrency=state.concurrency,
        max_iterations=state.max_iterations,
        threshold=state.threshold,
        capture_traces=False,
        candidate_source=state.candidate_source,
        lane=lane,
        candidate_root=candidate_root,
        workspace_root=workspace_root,
        row_scope="validation",
        vector_incumbent_hash=vector_incumbent_hash,
        vector_repetition=vector_repetition,
        dataset_role="validation",
        persist_report=False,
        redact_selection_evidence=True,
        persist_validation_replay=bool(state.continuation)
        and lane is None
        and state.lanes == 0,
    )
    if front is not None:
        from .front import snapshot_components

        snapshot_components(outcome, workspace_root or repo_root())
        front.record(outcome)
        if (
            state.acceptance_paired_min_cases is None
            or _validation_schedule(state, workspace_root)[0] != 1
        ):
            front.retire_replay(outcome)
    return (
        _with_timestamp(
            state,
            iterations=int(outcome.summary["iterations"]),
            validation_evaluations=state.validation_evaluations + 1,
        ),
        outcome,
    )


def _mark_best_from_validation(state: RunState, outcome: EvalOutcome) -> RunState:
    """Adopt a validation-scored candidate as the run's best."""

    return _with_timestamp(
        _mark_best_candidate(state, outcome),
        validation_seeded=True,
    )


def _acceptance_schedule(state: RunState, case_count: int) -> tuple[int, int]:
    """Use one paired repetition only for an explicitly configured large set."""
    if (
        state.acceptance_paired_min_cases is not None
        and case_count >= state.acceptance_paired_min_cases
    ):
        return 1, 1
    initial = max(3, state.acceptance_repetitions)
    return initial, max(initial, state.acceptance_max_repetitions)


def _validation_schedule(state: RunState, root: Path | None = None) -> tuple[int, int]:
    from .dataset import load_dataset

    root = root or repo_root()
    path, _ = check_heldout_pin(root, state.run_id)
    try:
        cases = load_dataset(Path(path))
    except (OSError, ValueError, TypeError):
        raise typer.BadParameter(
            "Could not load held-out validation dataset."
        ) from None
    return _acceptance_schedule(state, len(cases))


def _case_scores(outcome: EvalOutcome) -> dict[str, float]:
    return {record.case_id: record.score for record in outcome.records}


def _mark_best_validation_samples(
    state: RunState, outcomes: Sequence[EvalOutcome]
) -> RunState:
    samples = tuple(float(outcome.summary["mean_score"]) for outcome in outcomes)
    return _with_timestamp(
        _mark_best_from_validation(state, outcomes[-1]),
        best_validation_samples=samples,
        best_validation_per_case_scores=_case_scores(outcomes[0])
        if len(outcomes) == 1
        else {},
        best_mean_score=sum(samples) / len(samples),
    )


def _inconclusive_comparison(reason: str) -> dict[str, Any]:
    return {
        "outcome": "valid",
        "selectable": False,
        "verdict": "inconclusive",
        "improved": False,
        "reason_code": reason,
    }


def _validation_improved(
    state: RunState, outcomes: Sequence[EvalOutcome], *, initial: int, maximum: int
) -> dict[str, Any]:
    if not state.best_validation_samples or (
        initial == 1 and not state.best_validation_per_case_scores
    ):
        return _inconclusive_comparison("incumbent_evidence_missing")
    if any(outcome.summary.get("selectable") is False for outcome in outcomes):
        return _inconclusive_comparison("validation_not_selectable")
    result = compare_candidate_samples(
        state.best_validation_samples,
        [float(outcome.summary["mean_score"]) for outcome in outcomes],
        confidence=state.acceptance_confidence,
        min_delta=state.acceptance_min_delta,
        max_looks=maximum - initial + 1,
        paired_baseline_scores=state.best_validation_per_case_scores
        if initial == 1
        else None,
        paired_candidate_scores=_case_scores(outcomes[0]) if initial == 1 else None,
    )
    return {"outcome": "valid", "selectable": True, **result.to_dict()}


def _confirm_validation_candidate(
    state: RunState,
    *,
    candidate_root: Path | None = None,
    workspace_root: Path | None = None,
    lane: str | None = None,
) -> tuple[RunState, list[EvalOutcome], dict[str, Any]]:
    initial, maximum = _validation_schedule(state, workspace_root)
    if not state.best_validation_samples or (
        initial == 1 and not state.best_validation_per_case_scores
    ):
        return state, [], _inconclusive_comparison("incumbent_evidence_missing")
    if state.max_iterations - state.iterations < initial:
        return state, [], _inconclusive_comparison("validation_budget_exhausted")
    outcomes: list[EvalOutcome] = []
    for _ in range(min(maximum, state.max_iterations - state.iterations)):
        state, outcome = _evaluate_validation_candidate(
            state,
            candidate_root=candidate_root,
            workspace_root=workspace_root,
            lane=lane,
        )
        outcomes.append(outcome)
        if outcome.summary["candidate_id"] != outcomes[0].summary["candidate_id"]:
            raise typer.BadParameter("Candidate changed during validation sampling.")
        failures = _outcome_infrastructure_failures(outcome)
        if failures:
            state, comparison = _pause_after_infrastructure_failure(
                state, outcomes, phase="candidate", failures=failures
            )
            return state, outcomes, comparison
        if len(outcomes) >= initial:
            comparison = _validation_improved(
                state, outcomes, initial=initial, maximum=maximum
            )
            if comparison["verdict"] != "inconclusive":
                break
    return state, outcomes, comparison


def _ensure_validation_seed(state: RunState) -> tuple[RunState, list[EvalOutcome]]:
    """Collect incumbent evidence while its tree is still available."""
    if not state.heldout_required:
        return state, []
    _assert_validation_dataset_unchanged(state)
    initial = None
    if state.best_validation_samples:
        initial, _ = _validation_schedule(state)
        if initial != 1 or state.best_validation_per_case_scores:
            return state, []
    if (
        state.best_candidate_id is not None
        and _current_baseline_candidate_id(
            state.candidate_source, active_run_id=state.run_id
        )
        != state.best_candidate_id
    ):
        return _with_timestamp(
            state,
            last_comparison=_inconclusive_comparison("incumbent_evidence_missing"),
        ), []
    if initial is None:
        initial, _ = _validation_schedule(state)
    if state.max_iterations - state.iterations < initial:
        return _with_timestamp(
            state,
            status="done",
            last_comparison=_inconclusive_comparison("validation_budget_exhausted"),
        ), []
    validation_path, validation_digest = _validation_dataset_identity()
    state = _with_timestamp(
        state,
        validation_dataset_path=validation_path,
        validation_dataset_digest=validation_digest,
    )
    outcomes: list[EvalOutcome] = []
    for _ in range(initial):
        state, outcome = _evaluate_validation_candidate(state)
        outcomes.append(outcome)
        if outcome.summary["candidate_id"] != outcomes[0].summary["candidate_id"]:
            raise typer.BadParameter("Candidate changed during validation sampling.")
        failures = _outcome_infrastructure_failures(outcome)
        if failures:
            state, comparison = _pause_after_infrastructure_failure(
                state, outcomes, phase="baseline", failures=failures
            )
            comparison["reason_code"] = "validation_rollout_failed"
            return _with_timestamp(
                state,
                last_comparison=comparison,
                infrastructure_retry_minibatch_id=None,
            ), outcomes
    return _mark_best_validation_samples(state, outcomes), outcomes


def _consume_candidate_verdict(state: RunState, *, accepted: bool) -> RunState:
    """Record one completed reflection verdict without changing lifecycle state."""

    return _with_timestamp(
        state,
        iterations_since_acceptance=0
        if accepted
        else state.iterations_since_acceptance + 1,
    )


def _consume_gate_rejection(state: RunState) -> RunState:
    """Record a rejected gate as one budgeted candidate iteration.

    Gate evaluations intentionally have no Pareto rows, so their budget cost
    must be represented explicitly in managed-run state.
    """

    state = _consume_candidate_verdict(state, accepted=False)
    return _with_timestamp(
        state,
        iterations=state.iterations + 1,
        gate_consumed_iterations=state.gate_consumed_iterations + 1,
    )


def _clear_reflection_baseline(state: RunState) -> RunState:
    return _with_timestamp(
        state,
        reflection_minibatch_id=None,
        reflection_baseline_candidate_id=None,
        reflection_baseline_commit_sha=None,
        reflection_baseline_mean_score=None,
        reflection_baseline_samples=(),
        reflection_baseline_eval_ids=(),
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
    retry_minibatch_id = state.infrastructure_retry_minibatch_id
    outcome = durable_eval(
        run_eval_once,
        spend_state=state,
        spend_kind="baseline",
        candidate_file=None,
        minibatch_id=retry_minibatch_id,
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
    return (
        _with_timestamp(
            state,
            next_epoch=epoch + (0 if retry_minibatch_id is not None else 1),
        ),
        outcome,
    )


def _capture_reflection_baseline(
    state: RunState, first_outcome: EvalOutcome
) -> tuple[RunState, list[EvalOutcome]]:
    """Measure the stochastic baseline before yielding the tree for edits."""

    remaining_iterations = state.max_iterations - state.iterations
    initial, maximum = _acceptance_schedule(state, len(first_outcome.records))
    validation_reserve = _validation_schedule(state)[0] if state.heldout_required else 0
    # The failure-selected outcome is already charged and is never evidence.
    affordable_repetitions = (remaining_iterations - validation_reserve) // 2
    if affordable_repetitions < initial:
        return _with_timestamp(
            state,
            status="done",
            last_comparison=_inconclusive_comparison("baseline_budget_exhausted"),
        ), []
    target_repetitions = min(maximum, affordable_repetitions)
    outcomes: list[EvalOutcome] = []
    first_failures = _outcome_infrastructure_failures(first_outcome)
    if first_failures:
        paused, _ = _pause_after_infrastructure_failure(
            state,
            [first_outcome],
            phase="baseline",
            failures=first_failures,
        )
        return paused, outcomes
    expected_candidate_id = str(first_outcome.summary["candidate_id"])
    minibatch_id = str(first_outcome.summary["minibatch_id"])

    while len(outcomes) < target_repetitions:
        outcome = durable_eval(
            run_eval_once,
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
        )
        if str(outcome.summary["candidate_id"]) != expected_candidate_id:
            public_echo(
                "The baseline candidate changed while collecting repeated "
                "evaluations; refusing to compare mixed candidates.",
                err=True,
            )
            raise typer.Exit(code=1)
        outcomes.append(outcome)
        state = _with_last_outcome(state, outcome)
        failures = _outcome_infrastructure_failures(outcome)
        if failures:
            paused, _ = _pause_after_infrastructure_failure(
                state,
                outcomes,
                phase="baseline",
                failures=failures,
                valid_samples=tuple(
                    float(item.summary["mean_score"]) for item in outcomes[:-1]
                ),
            )
            return paused, outcomes

    state = _mark_reflection_pause(state, outcomes)
    # Keep the failure that triggered reflection visible, but never use its
    # selected score as statistical evidence.
    selected_report = first_outcome.summary.get("report_path")
    selected_trace = first_outcome.summary.get("trace_path")
    if selected_report:
        state = replace(
            state,
            reflection_baseline_report_path=str(selected_report),
            reflection_baseline_report_paths=(
                str(selected_report),
                *state.reflection_baseline_report_paths,
            ),
        )
    if selected_trace:
        state = replace(
            state,
            reflection_baseline_trace_path=str(selected_trace),
            reflection_baseline_trace_paths=(
                str(selected_trace),
                *state.reflection_baseline_trace_paths,
            ),
        )
    return state, outcomes


def _advance_to_reflection_or_done(
    state: RunState,
) -> tuple[RunState, list[EvalOutcome]]:
    outcomes: list[EvalOutcome] = []
    state = _with_timestamp(_clear_reflection_baseline(state), status="running")
    state, validation_outcomes = _ensure_validation_seed(state)
    outcomes.extend(validation_outcomes)
    if state.status in {"paused_after_infrastructure_error", "done"}:
        return state, outcomes
    if (
        state.heldout_required
        and state.lanes == 0
        and state.iterations < state.max_iterations
    ):
        from .front import ValidationFront

        parent = ValidationFront(repo_root(), state.run_id).select(
            seed=state.seed, round_id=state.iterations
        )
        if parent is None and state.best_candidate_id:
            parent = {
                "candidate_id": state.best_candidate_id,
                "commit_sha": state.best_commit_sha,
            }
        if parent and parent["candidate_id"] != _current_baseline_candidate_id(
            state.candidate_source, active_run_id=state.run_id
        ):
            return _with_timestamp(
                state,
                status="paused_after_candidate_eval",
                next_parent_candidate_id=parent["candidate_id"],
                next_parent_commit_sha=parent["commit_sha"],
            ), outcomes
        state = _with_timestamp(
            state, next_parent_candidate_id=None, next_parent_commit_sha=None
        )
    while state.iterations < state.max_iterations:
        state, outcome = _fresh_baseline_outcome(state)
        outcomes.append(outcome)
        state = _with_last_outcome(state, outcome)

        failures = _outcome_infrastructure_failures(outcome)
        if failures:
            state, _ = _pause_after_infrastructure_failure(
                state,
                outcomes,
                phase="baseline",
                failures=failures,
            )
            return state, outcomes

        state = _with_timestamp(
            state,
            infrastructure_retry_minibatch_id=None,
            last_comparison=None,
        )
        if (
            not state.heldout_required
            and outcome.n_failures == 0
            and state.best_candidate_id is None
        ):
            state = _mark_best_candidate(state, outcome)
        if state.iterations >= state.max_iterations:
            if state.best_candidate_id is None and not state.heldout_required:
                state = _mark_best_candidate(state, outcome)
            return _mark_done(state), outcomes

        if outcome.n_failures > 0:
            state, baseline_outcomes = _capture_reflection_baseline(state, outcome)
            outcomes.extend(baseline_outcomes)
            return state, outcomes

    return _mark_done(state), outcomes


def _evaluate_reflected_candidate(
    state: RunState,
    *,
    gate_outcomes: Sequence[EvalOutcome] = (),
    gate_case_ids: Sequence[str] = (),
) -> tuple[RunState, list[EvalOutcome], dict[str, Any]]:
    if state.reflection_minibatch_id is None:
        public_echo(
            "Run is not waiting on a reflection minibatch; use `gepa run status`.",
            err=True,
        )
        raise typer.Exit(code=1)
    if not state.reflection_baseline_samples:
        public_echo("Run state is missing reflection baseline samples.", err=True)
        raise typer.Exit(code=1)

    validation_reserve = _validation_schedule(state)[0] if state.heldout_required else 0
    minibatch = MinibatchStore(state.run_id).load(state.reflection_minibatch_id)
    initial, maximum = _acceptance_schedule(state, len(minibatch.case_ids))
    max_candidate_samples = min(
        len(state.reflection_baseline_samples),
        state.max_iterations - state.iterations - validation_reserve,
    )
    if max_candidate_samples < initial:
        return state, [], _inconclusive_comparison("candidate_budget_exhausted")
    initial_candidate_samples = initial

    outcomes: list[EvalOutcome] = []
    candidate_samples: list[float] = []
    candidate_id: str | None = None
    comparison_result: AcceptanceComparison | None = None
    while len(candidate_samples) < max_candidate_samples:
        selected_case_ids: Sequence[str] | None = None
        supplemental_records = ()
        if not outcomes and gate_outcomes:
            assert state.reflection_minibatch_id is not None
            minibatch = MinibatchStore(state.run_id).load(state.reflection_minibatch_id)
            selected_case_ids = [
                case_id
                for case_id in minibatch.case_ids
                if case_id not in gate_case_ids
            ]
            supplemental_records = gate_outcomes[0].records
        outcome = durable_eval(
            run_eval_once,
            spend_state=state,
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
            selected_case_ids=selected_case_ids,
            supplemental_records=supplemental_records,
        )
        state = _with_last_outcome(state, outcome)
        outcomes.append(outcome)
        current_candidate_id = str(outcome.summary["candidate_id"])
        if candidate_id is None:
            candidate_id = current_candidate_id
        elif current_candidate_id != candidate_id:
            public_echo(
                "The reflected candidate changed while collecting repeated "
                "evaluations; refusing to compare mixed candidates.",
                err=True,
            )
            raise typer.Exit(code=1)
        failures = _outcome_infrastructure_failures(outcome)
        if failures:
            state, comparison = _pause_after_infrastructure_failure(
                state,
                outcomes,
                phase="candidate",
                failures=failures,
                valid_samples=tuple(candidate_samples),
            )
            return state, outcomes, comparison
        candidate_samples.append(float(outcome.summary["mean_score"]))

        if len(candidate_samples) < initial_candidate_samples:
            continue
        comparison_result = compare_candidate_samples(
            state.reflection_baseline_samples[: len(candidate_samples)],
            candidate_samples,
            confidence=state.acceptance_confidence,
            min_delta=state.acceptance_min_delta,
            max_looks=maximum - initial + 1,
            paired_baseline_scores=_reflection_case_scores(state)
            if initial == 1
            else None,
            paired_candidate_scores=_case_scores(outcomes[0]) if initial == 1 else None,
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
        "outcome": "valid",
        "selectable": True,
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
        "gate": {
            "cases": list(gate_case_ids),
            "report_paths": [
                outcome.summary["report_path"] for outcome in gate_outcomes
            ],
            "trace_paths": [
                outcome.summary["trace_path"]
                for outcome in gate_outcomes
                if outcome.summary.get("trace_path")
            ],
        }
        if gate_outcomes
        else None,
        "recommendation": recommendation,
    }
    if state.candidate_source == "git" and state.reflection_baseline_commit_sha:
        comparison["discard_command"] = (
            f"git reset --hard {state.reflection_baseline_commit_sha}"
        )
    state = _with_timestamp(state, last_comparison=comparison)
    return state, outcomes, comparison


def _reflection_case_scores(
    state: RunState, *, root: Path | None = None
) -> dict[str, float]:
    for row in ParetoLog(state.run_id, root).iter_rows():
        if (
            state.reflection_baseline_eval_ids
            and row.extra.get("eval_id") == state.reflection_baseline_eval_ids[0]
        ):
            return dict(row.per_case_scores)
    return {}


def _gate_baseline_samples(
    state: RunState, gate_case_ids: Sequence[str], *, root: Path | None = None
) -> tuple[float, ...]:
    """Recover the saved baseline's scores for a declared gate subset."""

    assert state.reflection_minibatch_id is not None
    if not state.reflection_baseline_eval_ids:
        raise typer.BadParameter(
            "The saved reflection baseline lacks evaluation identifiers. "
            "Start a new reflection iteration."
        )
    rows_by_eval_id = {
        str(row.extra.get("eval_id")): row
        for row in ParetoLog(state.run_id, root).iter_rows()
        if row.status not in {"infrastructure_failure"}
    }
    rows = [
        rows_by_eval_id[eval_id]
        for eval_id in state.reflection_baseline_eval_ids
        if eval_id in rows_by_eval_id
    ]
    expected = len(state.reflection_baseline_samples)
    if len(rows) != expected:
        raise typer.BadParameter(
            "The saved reflection baseline is missing per-case scores required "
            "for gate comparison. Start a new reflection iteration."
        )
    samples: list[float] = []
    for row in rows:
        missing = [
            case_id for case_id in gate_case_ids if case_id not in row.per_case_scores
        ]
        if missing:
            raise typer.BadParameter(
                "The saved reflection baseline is missing gate case score(s): "
                f"{missing}."
            )
        samples.append(
            sum(float(row.per_case_scores[case_id]) for case_id in gate_case_ids)
            / len(gate_case_ids)
        )
    return tuple(samples)


def _validate_gate_cases(
    state: RunState, gate_case_ids: Sequence[str], *, root: Path | None = None
) -> tuple[str, ...]:
    """Validate gate names before candidate evaluation begins."""

    assert state.reflection_minibatch_id is not None
    minibatch = MinibatchStore(state.run_id, root).load(state.reflection_minibatch_id)
    unknown = [
        case_id for case_id in gate_case_ids if case_id not in minibatch.case_ids
    ]
    if unknown:
        raise typer.BadParameter(
            "Gate case(s) must be in the current reflection minibatch; "
            f"unknown: {unknown}. Available: {list(minibatch.case_ids)}."
        )
    return tuple(dict.fromkeys(gate_case_ids))


def _evaluate_gate_cases(
    state: RunState,
    gate_case_ids: Sequence[str],
    *,
    workspace_root: Path | None = None,
    candidate_root: Path | None = None,
    lane: str | None = None,
) -> tuple[RunState, list[EvalOutcome], dict[str, Any]]:
    """Evaluate a candidate gate without adding gate rows to the Pareto log."""

    gate_case_ids = _validate_gate_cases(state, gate_case_ids, root=workspace_root)
    baseline_samples = _gate_baseline_samples(state, gate_case_ids, root=workspace_root)
    remaining = state.max_iterations - state.iterations
    if lane is not None:
        remaining = len(baseline_samples)
    max_candidate_samples = min(len(baseline_samples), remaining)
    if max_candidate_samples < 1:
        raise typer.BadParameter("No evaluation budget remains for gate comparison.")
    initial_samples, maximum = _acceptance_schedule(state, len(gate_case_ids))
    if max_candidate_samples < initial_samples:
        return state, [], _inconclusive_comparison("gate_baseline_evidence_missing")
    outcomes: list[EvalOutcome] = []
    candidate_samples: list[float] = []
    comparison_result: AcceptanceComparison | None = None
    candidate_id: str | None = None
    while len(candidate_samples) < max_candidate_samples:
        outcome = durable_eval(
            run_eval_once,
            spend_state=state,
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
            selected_case_ids=gate_case_ids,
            write_pareto=False,
            lane=lane,
            candidate_root=candidate_root,
            workspace_root=workspace_root,
        )
        outcomes.append(outcome)
        state = replace(_with_last_outcome(state, outcome), iterations=state.iterations)
        current_candidate_id = str(outcome.summary["candidate_id"])
        if candidate_id is None:
            candidate_id = current_candidate_id
        elif current_candidate_id != candidate_id:
            raise typer.BadParameter(
                "The reflected candidate changed while collecting gate evaluations; "
                "refusing to compare mixed candidates."
            )
        failures = _outcome_infrastructure_failures(outcome)
        if failures:
            state, comparison = _pause_after_infrastructure_failure(
                state,
                outcomes,
                phase="candidate",
                failures=failures,
                valid_samples=tuple(candidate_samples),
            )
            return state, outcomes, comparison
        candidate_samples.append(float(outcome.summary["mean_score"]))
        if len(candidate_samples) < initial_samples:
            continue
        comparison_result = compare_candidate_samples(
            baseline_samples[: len(candidate_samples)],
            candidate_samples,
            confidence=state.acceptance_confidence,
            min_delta=state.acceptance_min_delta,
            max_looks=maximum - initial_samples + 1,
            paired_baseline_scores={
                k: v
                for k, v in _reflection_case_scores(state, root=workspace_root).items()
                if k in gate_case_ids
            }
            if initial_samples == 1
            else None,
            paired_candidate_scores=_case_scores(outcomes[0])
            if initial_samples == 1
            else None,
        )
        if comparison_result.verdict != "inconclusive":
            break

    assert comparison_result is not None
    last_outcome = outcomes[-1]
    comparison = {
        "outcome": "valid",
        "selectable": False,
        "minibatch_id": state.reflection_minibatch_id,
        "gate_cases": list(gate_case_ids),
        "candidate_id": candidate_id,
        "baseline_mean_score": comparison_result.baseline_mean,
        "candidate_mean_score": comparison_result.candidate_mean,
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
        "rejection_reason": (
            "gate" if comparison_result.verdict == "rejected" else None
        ),
        "recommendation": "discard_or_revise",
    }
    return state, outcomes, comparison


def _current_baseline_candidate_id(
    candidate_source: CandidateSource = "components",
    *,
    active_run_id: str | None = None,
) -> str:
    """Return the candidate id for the current component files or git tree."""

    cfg = GepaConfig.load(config_path())
    if candidate_source == "git":
        try:
            state = git_candidate_state(exclude_paths=candidate_identity_exempt_paths())
        except GitCandidateError as exc:
            public_echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        return state.candidate_id

    insert_repo_root_on_path()
    agent = resolve_agent(cfg)
    skills_fs = resolve_skills(cfg)
    components = ComponentStore().effective_candidate(agent, skills_fs=skills_fs)
    return candidate_id_from_components(components)


def _write_final_report(
    state: RunState, *, overshoot: int | None = None, root: Path | None = None
) -> tuple[Path, str]:
    pareto = ParetoLog(state.run_id, root)
    rows = pareto.iter_rows()
    validation_rows = pareto.validation_rows()
    selectable_rows = validation_rows or pareto.selectable_rows()
    path = final_report_path(state.run_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# GEPA Run Final Report",
        "",
        f"- run_id: {state.run_id}",
        f"- status: {state.status}",
        f"- iterations: {state.iterations}/{state.max_iterations}",
        f"- validation_evaluations: {state.validation_evaluations}",
        f"- pareto_log: {ParetoLog(state.run_id, root).path}",
    ]
    if overshoot:
        lines.append(
            f"- budget_overshoot: {overshoot} eval row(s) beyond "
            f"--max-iterations (in-flight lane evals; bounded by "
            f"lanes x --acceptance-max-repetitions, pydanticaigepa-dec-msy)"
        )
    if selectable_rows:
        best = max(selectable_rows, key=lambda row: row.mean_score)
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
                f"- best_restore_command: git reset --hard {state.best_commit_sha}"
            )
    if state.last_comparison:
        comparison = state.last_comparison
        lines.extend(
            [
                "",
                "## Last Candidate Comparison",
                "",
                f"- minibatch_id: {comparison.get('minibatch_id')}",
                f"- outcome: {comparison.get('outcome', 'valid')}",
                f"- verdict: {comparison.get('verdict', 'unknown')}",
                f"- recommendation: {comparison.get('recommendation', comparison.get('reason_code'))}",
            ]
        )
        if comparison.get("outcome", "valid") == "valid" and "delta" in comparison:
            lines.extend(
                [
                    f"- baseline_mean_score: {comparison['baseline_mean_score']:.6f}",
                    f"- candidate_mean_score: {comparison['candidate_mean_score']:.6f}",
                    f"- delta: {comparison['delta']:.6f}",
                ]
            )
        elif comparison.get("evaluation_error_count"):
            lines.append(
                f"- evaluation_error_count: {comparison['evaluation_error_count']}"
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

    from .spend import spend_report

    lines.extend(
        [
            "",
            "## Rollout Spend",
            "",
            json.dumps(
                spend_report(state.run_id, root, state.max_token_cost), sort_keys=True
            ),
        ]
    )
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return path, text


def _public_state(
    state: RunState,
    *,
    outcomes: list[EvalOutcome],
    final_report: Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    from .spend import spend_report

    payload = state_for_save(state).to_dict()
    payload["spend"] = spend_report(state.run_id, root, cap=state.max_token_cost)
    payload.pop("best_validation_per_case_scores", None)
    payload["state_path"] = str(run_state_path(state.run_id, root))
    if state.lanes == 0:
        payload["reflector_packet_path"] = str(
            run_dir(state.run_id, root) / "reflector_packet.json"
        )
    payload["final_report_path"] = str(final_report) if final_report else None
    if state.status == "done":
        payload["next_command"] = None
    elif state.status == "paused_after_infrastructure_error":
        payload["next_command"] = f"gepa run continue --run-id {state.run_id}"
    elif state.lanes > 0:
        payload["next_command"] = f"gepa next --wait --run-id {state.run_id}"
    else:
        payload["next_command"] = f"gepa run continue --run-id {state.run_id}"
    payload["evaluations_this_call"] = [outcome.summary for outcome in outcomes]
    if state.next_parent_candidate_id and state.status == "paused_after_candidate_eval":
        from .front import parent_restore_command

        payload["parent_restore_command"] = parent_restore_command(
            state, root or repo_root()
        )
    if state.iterations_since_acceptance >= state.stall_threshold:
        payload["stall"] = {
            "stalled": True,
            "iterations_since_acceptance": state.iterations_since_acceptance,
        }
    return payload


def _emit_status(
    state: RunState,
    *,
    outcomes: list[EvalOutcome],
    final_report: Path | None = None,
    final_report_text: str | None = None,
) -> None:
    if state.next_parent_candidate_id and state.status == "paused_after_candidate_eval":
        public_echo(
            f"Next parent: {state.next_parent_candidate_id}. Restore it, then continue the run."
        )
        from .front import parent_restore_command

        public_echo(f"  {parent_restore_command(state, repo_root())}")
    elif state.status == "paused_for_reflection":
        editable_surface = (
            "source/artifacts and commit the result"
            if state.candidate_source == "git"
            else "components or source"
        )
        public_echo(
            "Paused for reflection. Inspect the report and trace file, edit "
            f"{editable_surface}, then run:"
        )
        public_echo(f"  gepa run continue --run-id {state.run_id}")
        public_echo(f"Report: {state.reflection_baseline_report_path}")
        public_echo(f"Trace: {state.reflection_baseline_trace_path}")
    elif state.status == "paused_after_candidate_eval":
        comparison = state.last_comparison or {}
        verdict = comparison.get("verdict", "rejected")
        if verdict == "inconclusive":
            public_echo(
                "Candidate comparison remains inconclusive after the configured "
                "repetitions. Revise the candidate or restore the baseline, then run:"
            )
        elif verdict == "equivalent":
            public_echo(
                "Candidate is equivalent within the configured practical delta. "
                "Restore the baseline or revise the candidate, then run:"
            )
        else:
            if comparison.get("rejection_reason") == "validation":
                public_echo(
                    "Candidate improved the training minibatch but did not improve "
                    "held-out validation. Discard or revise the edits, then run:"
                )
            else:
                public_echo(
                    "Candidate did not beat the reflection baseline. Recommendation: "
                    "discard or revise the edits, then run:"
                )
        public_echo(f"  gepa run continue --run-id {state.run_id}")
        if comparison and "delta" in comparison:
            public_echo(
                f"Baseline {comparison['baseline_mean_score']:.6f}; "
                f"candidate {comparison['candidate_mean_score']:.6f}; "
                f"delta {comparison['delta']:.6f}."
            )
            if "lower_bound" in comparison and "upper_bound" in comparison:
                public_echo(
                    f"{comparison['confidence']:.0%} interval "
                    f"[{comparison['lower_bound']:.6f}, "
                    f"{comparison['upper_bound']:.6f}]; "
                    f"verdict {verdict}."
                )
            public_echo(f"Candidate report: {comparison['candidate_report_path']}")
            public_echo(f"Candidate trace: {comparison['candidate_trace_path']}")
            if comparison.get("validation_evaluated"):
                public_echo(
                    "Held-out validation was used for selection; no validation "
                    "report or trace was exposed to reflection."
                )
            if comparison.get("discard_command"):
                public_echo(
                    "To discard the git candidate and restore the reflection "
                    "baseline, run:"
                )
                public_echo(f"  {comparison['discard_command']}")
    elif state.status == "paused_after_infrastructure_error":
        comparison = state.last_comparison or {}
        public_echo(
            "A required evaluation rollout failed outside the quality "
            "comparison. The incumbent was preserved. Recover the service "
            "or configuration, then retry:"
        )
        public_echo(f"  gepa run continue --run-id {state.run_id}")
        if comparison.get("candidate_report_path"):
            public_echo(f"Failure report: {comparison['candidate_report_path']}")
        if comparison.get("candidate_trace_path"):
            public_echo(f"Failure trace: {comparison['candidate_trace_path']}")
    elif state.status == "done":
        public_echo("Run complete.")
        if final_report_text:
            public_echo(final_report_text.rstrip())
    else:
        public_echo(f"Run status: {state.status}")

    public_echo(
        json.dumps(
            {"run": _public_state(state, outcomes=outcomes, final_report=final_report)}
        )
    )


def _validate_max_iterations(max_iterations: int) -> None:
    if max_iterations < 1:
        public_echo("--max-iterations must be >= 1.", err=True)
        raise typer.Exit(code=2)


def _validate_acceptance_options(
    *,
    repetitions: int,
    max_repetitions: int,
    confidence: float,
    min_delta: float,
) -> None:
    if repetitions < 1:
        public_echo("--acceptance-repetitions must be >= 1.", err=True)
        raise typer.Exit(code=2)
    if max_repetitions < repetitions:
        public_echo(
            "--acceptance-max-repetitions must be >= --acceptance-repetitions.",
            err=True,
        )
        raise typer.Exit(code=2)
    if not 0.0 < confidence < 1.0:
        public_echo("--acceptance-confidence must be between 0 and 1.", err=True)
        raise typer.Exit(code=2)
    if min_delta < 0.0:
        public_echo("--acceptance-min-delta must be >= 0.", err=True)
        raise typer.Exit(code=2)


def _fan_out_lane_run_if_ready(
    state: RunState,
    outcomes: list[EvalOutcome],
) -> tuple[RunState, list[EvalOutcome]]:
    """Validate a measured lane baseline and fan out its next iteration."""

    if state.lanes < 1 or state.status != "paused_for_reflection":
        return state, outcomes

    from .lanes import fan_out_lanes

    workspace_root = repo_root()
    fresh_state = git_candidate_state(
        workspace_root,
        exclude_paths=candidate_identity_exempt_paths(workspace_root),
    )
    if (
        fresh_state.commit_sha != state.reflection_baseline_commit_sha
        or fresh_state.dirty
    ):
        state = _with_timestamp(state, status="running")
        state.save()
        state, extra_outcomes = _advance_to_reflection_or_done(state)
        outcomes.extend(extra_outcomes)
        state.save()

    if state.status == "paused_for_reflection":
        fan_out_lanes(state, workspace_root)
        state = _with_timestamp(state, status="running")
        state = replace(state, iteration_started_at=state.updated_at)
        state.save()
    return state, outcomes


@app.command("start")
@harness_environment()
def start(
    heldout_required: bool = typer.Option(
        False,
        "--heldout-required",
        help="Require harness-held validation; fail if its environment is missing.",
    ),
    max_token_cost: float | None = typer.Option(
        None,
        "--max-token-cost",
        help="Run rollout spend cap in US dollars (finite and > 0).",
    ),
    max_iterations: int = typer.Option(
        100,
        "--max-iterations",
        help="Total evaluation-row budget for this managed run.",
    ),
    size: int = typer.Option(
        10, "--size", help="Number of cases in each sampled training minibatch."
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
            "Initial repeated evaluations per candidate on the saved training minibatch. "
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
    acceptance_paired_min_cases: int | None = typer.Option(
        None,
        "--acceptance-paired-min-cases",
        help="Use one paired repetition at or above this case count; disabled by default.",
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
    stall_threshold: int | None = typer.Option(
        None,
        "--stall-threshold",
        help="Candidate verdicts without acceptance before stall reporting begins. Defaults to gepa.toml stall_threshold (5).",
    ),
) -> None:
    """Start a managed GEPA run and pause at the first reflection point."""
    from .spend import validate_cap

    validate_cap(max_token_cost)
    _validate_max_iterations(max_iterations)
    if lanes < 0:
        public_echo("--lanes must be >= 0.", err=True)
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
        public_echo("--candidate-source must be 'components' or 'git'.", err=True)
        raise typer.Exit(code=2)
    cfg = GepaConfig.load(config_path())
    if acceptance_paired_min_cases is None:
        acceptance_paired_min_cases = cfg.acceptance.paired_min_cases
    if acceptance_paired_min_cases is not None and acceptance_paired_min_cases < 2:
        raise typer.BadParameter("--acceptance-paired-min-cases must be >= 2.")
    heldout_required = heldout_required or _held_out_validation_enabled()
    if heldout_required:
        _validation_dataset_identity()
        from . import scoring_sandbox

        if scoring_sandbox.required():
            scoring_sandbox.require_supported(
                cfg, candidate_source or cfg.candidate_source
            )
    vector_validation = heldout_required and cfg.acceptance.mode == "vector"
    if vector_validation and lanes == 0:
        public_echo(
            "Vector held-out validation requires --lanes greater than zero; "
            "the synchronous loop supports scalar validation only.",
            err=True,
        )
        raise typer.Exit(code=2)
    if vector_validation and resolved_max_repetitions < 2:
        public_echo(
            "Vector held-out validation requires at least two maximum "
            "acceptance repetitions.",
            err=True,
        )
        raise typer.Exit(code=2)
    resolved_stall_threshold = (
        cfg.stall_threshold if stall_threshold is None else stall_threshold
    )
    if resolved_stall_threshold < 1:
        public_echo("--stall-threshold must be >= 1.", err=True)
        raise typer.Exit(code=2)
    active_candidate_source = cast(
        CandidateSource, candidate_source or cfg.candidate_source
    )
    workspace_root = repo_root()
    from .lanes import ensure_worktrees_ignored

    if lanes > 0:
        if active_candidate_source != "git":
            public_echo(
                "--lanes requires git candidate mode (component-mode lanes share "
                "one process-global agent and are out of scope, spec-1do).",
                err=True,
            )
            raise typer.Exit(code=2)
        # Lane branches are always cut from a clean commit; a dirty primary
        # tree is rejected (spec-1do constraint). The journal is tracked
        # bookkeeping the CLI itself appends to — exclude it like select does
        # (one dirtiness definition across verbs).
        ensure_worktrees_ignored(workspace_root)
        try:
            primary_state = git_candidate_state(
                workspace_root,
                exclude_paths=candidate_identity_exempt_paths(workspace_root),
            )
        except GitCandidateError as exc:
            public_echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        if primary_state.dirty:
            public_echo(
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
                public_echo(
                    f"Lane run {prior.run_id} is still active "
                    f"(status {prior.status}); finish it (`gepa run select`) "
                    "or abandon it before starting another lane run.",
                    err=True,
                )
                raise typer.Exit(code=1)
    run_id = new_run_id()
    run_dir(run_id).mkdir(parents=True, exist_ok=True)
    now = utc_now_iso()
    if heldout_required:
        pin_heldout(workspace_root, run_id)
    state = RunState(
        heldout_required=heldout_required,
        run_id=run_id,
        status="running",
        max_iterations=max_iterations,
        max_token_cost=max_token_cost,
        size=size,
        seed=seed,
        next_epoch=epoch,
        concurrency=resolved_concurrency,
        threshold=threshold,
        acceptance_repetitions=acceptance_repetitions,
        acceptance_max_repetitions=resolved_max_repetitions,
        acceptance_confidence=acceptance_confidence,
        acceptance_min_delta=acceptance_min_delta,
        acceptance_paired_min_cases=acceptance_paired_min_cases,
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
        stall_threshold=resolved_stall_threshold,
        project_root=str(workspace_root.resolve()),
    )
    state.save()
    state, outcomes = _advance_to_reflection_or_done(state)
    state.save()
    state, outcomes = _fan_out_lane_run_if_ready(state, outcomes)

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
    gate_case: list[str] = typer.Option(
        [],
        "--gate-case",
        help="Case name from the current reflection minibatch to evaluate first. Repeatable.",
    ),
    reflector_epoch: int | None = typer.Option(None, "--reflector-epoch"),
    wait_secs: float = typer.Option(
        300.0,
        "--wait-secs",
        min=0,
        help="Wait for harness scoring; 0 enqueues. Timeout exits 75; retry the same command.",
    ),
) -> None:
    """Resume after reflection edits and advance to the next pause or completion."""
    if _load_state(run_id).heldout_required:
        from .harness import nominate

        nominate(run_id, gate_case, reflector_epoch, wait_secs)
    else:
        continue_run(run_id, gate_case, reflector_epoch, _continue_impl)


def _continue_impl(run_id: str | None, gate_case: list[str]) -> None:
    state = state_for_replay(_load_state(run_id).restore_validation_evidence())
    if state.heldout_required:
        _assert_validation_dataset_unchanged(state)
    if state.lanes > 0 and not (
        state.status == "paused_after_infrastructure_error"
        and state.reflection_minibatch_id is None
    ):
        public_echo(
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
    if state.next_parent_candidate_id:
        if (
            _current_baseline_candidate_id(
                state.candidate_source, active_run_id=state.run_id
            )
            != state.next_parent_candidate_id
        ):
            state.save()
            _emit_status(state, outcomes=[])
            return
        state, outcomes = _advance_to_reflection_or_done(state)
        state.save()
        final_path, final_text = (
            _write_final_report(state) if state.status == "done" else (None, None)
        )
        _emit_status(
            state,
            outcomes=outcomes,
            final_report=final_path,
            final_report_text=final_text,
        )
        return
    if state.lanes > 0:
        state, outcomes = _advance_to_reflection_or_done(state)
        state, outcomes = _fan_out_lane_run_if_ready(state, outcomes)
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

    if (
        state.status == "paused_after_candidate_eval"
        and state.reflection_baseline_candidate_id is not None
        and _current_baseline_candidate_id(
            state.candidate_source, active_run_id=state.run_id
        )
        == state.reflection_baseline_candidate_id
    ):
        public_echo(
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
        gate_outcomes: list[EvalOutcome] = []
        gate_comparison: dict[str, Any] | None = None
        comparison: dict[str, Any]
        comparison_outcomes: list[EvalOutcome]
        if gate_case:
            state, gate_outcomes, gate_comparison = _evaluate_gate_cases(
                state, gate_case
            )
            outcomes.extend(gate_outcomes)
        gate_rejected = (
            gate_comparison is not None
            and gate_comparison.get("rejection_reason") == "gate"
        )
        if gate_rejected:
            assert gate_comparison is not None
            comparison = gate_comparison
            comparison_outcomes = gate_outcomes
            state = _consume_gate_rejection(state)
            state = _with_timestamp(
                state,
                status="paused_after_candidate_eval",
                last_comparison=comparison,
            )
        elif gate_comparison and gate_comparison.get("outcome") != "valid":
            comparison = gate_comparison
            comparison_outcomes = gate_outcomes
        else:
            state, comparison_outcomes, comparison = _evaluate_reflected_candidate(
                state,
                gate_outcomes=gate_outcomes,
                gate_case_ids=gate_case,
            )
            outcomes.extend(comparison_outcomes)

        state = _with_timestamp(state, last_comparison=comparison)
        validation_outcomes: list[EvalOutcome] = []
        if (
            state.heldout_required
            and comparison.get("outcome") == "valid"
            and comparison["improved"]
        ):
            training_comparison = dict(comparison)
            training_verdict = str(comparison["verdict"])
            training_mean = float(comparison["candidate_mean_score"])
            state, validation_outcomes, validation_comparison = (
                _confirm_validation_candidate(state)
            )
            outcomes.extend(validation_outcomes)
            comparison.update(
                {
                    **validation_comparison,
                    "training_comparison": training_comparison,
                    "training_verdict": training_verdict,
                    "training_mean_score": training_mean,
                    "validation_evaluated": bool(validation_outcomes),
                    "validation_improved": validation_comparison["improved"],
                    "validation_comparison": validation_comparison,
                    "validation_mean_score": validation_comparison.get(
                        "candidate_mean"
                    ),
                    "prior_best_validation_mean_score": state.best_mean_score,
                    "rejection_reason": None
                    if validation_comparison["improved"]
                    else "validation",
                    "recommendation": "keep_and_advance"
                    if validation_comparison["improved"]
                    else "discard_or_revise",
                }
            )
            if "baseline_mean" in validation_comparison:
                comparison["baseline_mean_score"] = validation_comparison[
                    "baseline_mean"
                ]
                comparison["candidate_mean_score"] = validation_comparison[
                    "candidate_mean"
                ]
            state = _with_timestamp(state, last_comparison=comparison)

        if (
            validation_outcomes
            and not comparison["improved"]
            and comparison.get("outcome") == "valid"
        ):
            if state.iterations >= state.max_iterations:
                comparison["recommendation"] = "review_best"
            elif (
                comparison.get("candidate_id") != state.reflection_baseline_candidate_id
            ):
                comparison["recommendation"] = "select_next_parent"
            state = _with_timestamp(state, last_comparison=comparison)
        state = remember_comparison(state, comparison)
        if comparison["improved"]:
            state = _consume_candidate_verdict(state, accepted=True)
            if validation_outcomes:
                state = _mark_best_validation_samples(state, validation_outcomes)
            else:
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
        elif (
            not gate_rejected and comparison.get("outcome") != "infrastructure_failure"
        ):
            state = _consume_candidate_verdict(state, accepted=False)
            state = _with_timestamp(state, status="paused_after_candidate_eval")
            if (
                state.heldout_required
                and validation_outcomes
                and comparison.get("candidate_id")
                != state.reflection_baseline_candidate_id
            ):
                state, advanced_outcomes = _advance_to_reflection_or_done(state)
                outcomes.extend(advanced_outcomes)
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
    if (
        state.last_comparison is not None
        and state.last_comparison.get("reason_code") == "candidate_budget_exhausted"
    ):
        raise typer.Exit(code=70)


@app.command("resume")
def resume(
    run_id: str | None = typer.Option(None, "--run-id"),
    reason: str | None = typer.Option(None, "--reason"),
    reflector: str | None = typer.Option(None, "--reflector"),
    abandon_continuation: bool = typer.Option(False, "--abandon-continuation"),
) -> None:
    """Re-issue a durable packet after losing the previous reflector."""
    from .reflector import resume as resume_reflector

    resume_reflector(run_id, reason, reflector, abandon_continuation)
    state = _load_state(run_id)
    if state.next_parent_candidate_id:
        from .front import write_parent_packet

        write_parent_packet(
            run_dir(state.run_id) / "reflector_packet.json", state, repo_root()
        )
        _emit_status(state, outcomes=[])


@app.command("select")
@harness_environment()
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
    public_echo(json.dumps(payload))


__all__ = ["app"]

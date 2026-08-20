"""Budgeted, journal-authenticated one-case assertion-vector probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import typer

from ..vector_acceptance import VectorRecord, VectorRecordStore
from .eval import run_eval_once
from .lanes import (
    _append_journal,
    _candidate_gate,
    _journal_rows,
    _lane_lock,
    _resolve_lane_run,
    load_all_lane_states,
    load_lane_state,
)
from .layout import GepaConfig, config_path, probe_receipts_dir, vector_records_path
from .runs import utc_now_iso


def component_hash(worktree: Path, component_files: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(component_files):
        path = worktree / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _statuses(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(key): value.get("status") if isinstance(value, Mapping) else value
        for key, value in payload.items()
    }


def is_fixed_status_change(before: Any, after: Any) -> bool:
    """Return whether a probe status moved from fail-like to pass-like."""
    fail_like = before is False or (
        isinstance(before, str)
        and before.casefold() in {"fail", "failed", "failing", "false"}
    )
    pass_like = after is True or (
        isinstance(after, str)
        and after.casefold() in {"pass", "passed", "passing", "true"}
    )
    return fail_like and pass_like


def _receipt_proof(
    worktree: Path, case: str, changes: Mapping[str, Any]
) -> dict[str, str] | None:
    """Bind a receipt to one predicted key/case/fixed direction."""
    requested: list[str] = []
    prediction_path = worktree / "prediction.json"
    if prediction_path.is_file():
        try:
            raw = json.loads(prediction_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = {}
        predictions = raw.get("predictions", []) if isinstance(raw, Mapping) else []
        if isinstance(predictions, list):
            requested = [
                str(item["key"])
                for item in predictions
                if isinstance(item, Mapping)
                and item.get("case") == case
                and item.get("direction") == "fail_to_pass"
                and isinstance(item.get("key"), str)
            ]
    candidates = [*requested, *(key for key in sorted(changes) if key not in requested)]
    for key in candidates:
        change = changes.get(key)
        if isinstance(change, Mapping) and is_fixed_status_change(
            change.get("before"), change.get("after")
        ):
            return {"key": key, "case": case, "direction": "fail_to_pass"}
    return None


def probe(
    case: str = typer.Option(..., "--case", help="Frozen dataset case id to probe."),
    lane: str | None = typer.Option(
        None, "--lane", help="Lane to probe; required when a run has multiple lanes."
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Defaults to the latest lane run."
    ),
) -> None:
    """Evaluate one gated case without acceptance/Pareto eligibility."""
    workspace_root, run_state = _resolve_lane_run(run_id)
    if lane is None:
        states = load_all_lane_states(workspace_root, run_state.run_id)
        if len(states) != 1:
            raise typer.BadParameter(
                "--lane is required when the run has multiple lanes."
            )
        lane = states[0].lane
    cfg = GepaConfig.load(config_path(workspace_root))
    if cfg.acceptance.mode != "vector":
        raise typer.BadParameter("gepa probe requires acceptance.mode = 'vector'.")
    with _lane_lock(workspace_root, run_state.run_id, lane):
        state = load_lane_state(workspace_root, run_state.run_id, lane)
        if not state.worktree_path or not state.candidate_project_path:
            raise typer.BadParameter(f"Lane {lane} has no candidate worktree.")
        rejected = _candidate_gate(
            workspace_root=workspace_root,
            run_state=run_state,
            state=state,
            verify_probe_receipt=False,
            rejection_kind="probe_review_rejection",
        )
        if rejected is not None:
            typer.echo(
                f"Lane {lane} candidate review failed; probe rollout was not run.",
                err=True,
            )
            raise typer.Exit(code=1)
        used = len(
            [
                row
                for row in _journal_rows(
                    workspace_root, run_state.run_id, "probe_budget_debit"
                )
                if row.get("lane") == lane and row.get("lease_epoch") == state.lease_epoch
            ]
        )
        if used >= cfg.acceptance.probe_allowance_per_lease:
            raise typer.BadParameter(
                "Probe allowance exhausted for this lane lease "
                f"({used}/{cfg.acceptance.probe_allowance_per_lease})."
            )
        _append_journal(
            workspace_root,
            {
                "timestamp": utc_now_iso(),
                "kind": "probe_budget_debit",
                "run_id": run_state.run_id,
                "lane": lane,
                "iteration": state.iteration,
                "lease_epoch": state.lease_epoch,
                "allowance": cfg.acceptance.probe_allowance_per_lease,
                "used": used + 1,
                "case": case,
            },
        )
        worktree = Path(state.worktree_path)
        candidate_root = Path(state.candidate_project_path)
        outcome = run_eval_once(
            candidate_file=None,
            minibatch_id=None,
            size=1,
            seed=run_state.seed,
            epoch=run_state.next_epoch,
            run_id=run_state.run_id,
            concurrency=run_state.concurrency,
            max_iterations=run_state.max_iterations,
            threshold=run_state.threshold,
            capture_traces=True,
            candidate_source="git",
            lane=lane,
            candidate_root=candidate_root,
            workspace_root=workspace_root,
            case_id=case,
            row_scope="probe",
            vector_incumbent_hash=run_state.reflection_baseline_candidate_id,
        )
        raw = outcome.summary.get("vector_record")
        if not isinstance(raw, dict):
            raise typer.BadParameter("Probe metric did not produce a vector record.")
        candidate = VectorRecord.from_dict(raw)
        store = VectorRecordStore(vector_records_path(run_state.run_id, workspace_root))
        baseline = store.matching_incumbent_for_case(candidate.key, case_id=case)
        if not baseline:
            raise typer.BadParameter(
                "No compatible incumbent vector record is available for this probe."
            )
        before = _statuses(baseline[-1].assertions.get(case))
        after = _statuses(candidate.assertions.get(case))
        changes = {
            key: {"before": before.get(key), "after": after.get(key)}
            for key in sorted(set(before) | set(after))
            if before.get(key) != after.get(key)
        }
        receipt = {
            "receipt_version": 1,
            "run_id": run_state.run_id,
            "lane": lane,
            "iteration": state.iteration,
            "case": case,
            "probe_row_id": str(outcome.summary["eval_id"]),
            "candidate_component_hash": component_hash(
                worktree, cfg.acceptance.component_files
            ),
            "candidate_hash": candidate.key.candidate_hash,
            "incumbent_hash": candidate.key.incumbent_hash,
            "inventory_hash": baseline[-1].key.inventory_hash,
            "scorer_identity": candidate.key.scorer_identity,
            "vector_schema_version": candidate.key.vector_schema_version,
            "telemetry_schema_version": candidate.key.telemetry_schema_version,
            "changes": changes,
            "proof": _receipt_proof(worktree, case, changes),
            "row_scope": "probe",
        }
        _append_journal(
            workspace_root,
            {"timestamp": utc_now_iso(), "kind": "probe_receipt", **receipt},
        )
        directory = probe_receipts_dir(run_state.run_id, workspace_root)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{lane}-{state.iteration:04d}-{case}.json"
        path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    typer.echo(
        json.dumps({"receipt_path": str(path), "changes": changes}, sort_keys=True)
    )

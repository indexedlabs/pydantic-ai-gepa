"""One-case, non-budgeted assertion-vector probes for reflection lanes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import typer

from ..vector_acceptance import VectorRecord, VectorRecordStore
from .eval import run_eval_once
from .lanes import _resolve_lane_run, load_all_lane_states, load_lane_state
from .layout import GepaConfig, config_path, probe_receipts_dir, vector_records_path


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


def probe(
    case: str = typer.Option(..., "--case", help="Frozen dataset case id to probe."),
    lane: str | None = typer.Option(None, "--lane", help="Lane to probe; required when a run has multiple lanes."),
    run_id: str | None = typer.Option(None, "--run-id", help="Defaults to the latest lane run."),
) -> None:
    """Evaluate one case without spending acceptance budget or Pareto eligibility."""
    workspace_root, run_state = _resolve_lane_run(run_id)
    if lane is None:
        states = load_all_lane_states(workspace_root, run_state.run_id)
        if len(states) != 1:
            raise typer.BadParameter("--lane is required when the run has multiple lanes.")
        lane = states[0].lane
    state = load_lane_state(workspace_root, run_state.run_id, lane)
    if not state.worktree_path or not state.candidate_project_path:
        raise typer.BadParameter(f"Lane {lane} has no candidate worktree.")
    cfg = GepaConfig.load(config_path(workspace_root))
    if cfg.acceptance.mode != "vector":
        raise typer.BadParameter("gepa probe requires acceptance.mode = 'vector'.")
    worktree = Path(state.worktree_path)
    candidate_root = Path(state.candidate_project_path)
    outcome = run_eval_once(
        candidate_file=None, minibatch_id=None, size=1, seed=run_state.seed,
        epoch=run_state.next_epoch, run_id=run_state.run_id,
        concurrency=run_state.concurrency, max_iterations=run_state.max_iterations,
        threshold=run_state.threshold, capture_traces=True, candidate_source="git",
        lane=lane, candidate_root=candidate_root, workspace_root=workspace_root,
        case_id=case, row_scope="probe",
        vector_incumbent_hash=run_state.reflection_baseline_candidate_id,
    )
    raw = outcome.summary.get("vector_record")
    if not isinstance(raw, dict):
        raise typer.BadParameter("Probe metric did not produce a vector record.")
    candidate = VectorRecord.from_dict(raw)
    store = VectorRecordStore(vector_records_path(run_state.run_id, workspace_root))
    baseline = store.matching(candidate.key, candidate_hash=candidate.key.incumbent_hash)
    if not baseline:
        raise typer.BadParameter("No compatible incumbent vector record is available for this probe.")
    before = _statuses(baseline[-1].assertions.get(case))
    after = _statuses(candidate.assertions.get(case))
    changes = {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    }
    receipt = {
        "receipt_version": 1, "run_id": run_state.run_id, "lane": lane,
        "iteration": state.iteration, "case": case,
        "candidate_component_hash": component_hash(worktree, cfg.acceptance.component_files),
        "candidate_hash": candidate.key.candidate_hash,
        "incumbent_hash": candidate.key.incumbent_hash,
        "inventory_hash": candidate.key.inventory_hash,
        "scorer_identity": candidate.key.scorer_identity,
        "vector_schema_version": candidate.key.vector_schema_version,
        "telemetry_schema_version": candidate.key.telemetry_schema_version,
        "changes": changes, "row_scope": "probe",
    }
    directory = probe_receipts_dir(run_state.run_id, workspace_root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{lane}-{state.iteration:04d}-{case}.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(json.dumps({"receipt_path": str(path), "changes": changes}, sort_keys=True))

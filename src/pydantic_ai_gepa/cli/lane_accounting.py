"""Harness-owned estimates for reflector-owned held-out lane training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import typer

from . import harness_record
from .layout import config_path, GepaConfig
from .runs import MinibatchStore, ParetoLog, ParetoRow, utc_now_iso

if TYPE_CHECKING:
    from .lanes import LaneState
    from .run import RunState


KIND = "lane_training_estimate"
_KEY = "@lane-training-charges"


def _schedule(state: RunState, root: Path) -> tuple[int, int]:
    """Maximum Pareto budget rows and paid rollouts per consumed verdict.

    The scalar loop cannot exceed the harness's frozen baseline repetitions.
    A gate can repeat that whole minibatch before the normal loop (an all-case
    gate has no complement to reuse). Vector acceptance permits one additional
    comparison sample and one infrastructure retry. Count the full gate too:
    neither a reflector's gate declaration nor its sample list is authority.
    Gates use write_pareto=False, so their cost belongs in the dollar estimate,
    not in the acceptance-row iteration count. Preserve that budget convention.
    """
    cases = len(
        MinibatchStore(state.run_id, root)
        .load(str(state.reflection_minibatch_id))
        .case_ids
    )
    baseline = len(state.reflection_baseline_samples)
    normal = (
        state.acceptance_repetitions + 2
        if GepaConfig.load(config_path(root)).acceptance.mode == "vector"
        else baseline
    )
    return normal, (baseline + normal) * cases


def _highest(rows: list[dict[str, Any]]) -> float | None:
    # Estimated rows must never become observations and compound on replay.
    observed = [
        row
        for row in rows
        if row["kind"] != KIND
        and (
            row.get("rollouts_completed", 0) > row.get("cached_rollouts", 0)
            or row.get("max_rollout_dollars", 0) > 0
        )
        and not row.get("unmetered_rollouts")
        and not row.get("unpriced_usage")
    ]
    return max((row.get("max_rollout_dollars", 0.0) for row in observed), default=None)


def refresh(run_id: str, root: Path | None) -> None:
    """Reprice this selection's verdicts after harness metering, including failures.

    The private ledger includes the candidate's own validation/confirmation
    high-water costs. A charge bounds scheduled training *conditionally*: each
    training rollout must cost no more than this observed maximum. Training-only
    expensive behavior, extra/unconsumed evaluations and direct provider access require
    mandatory harness-owned admission or a provider-side budget to bound.

    Public charges use only already-public training prices. Publishing the full
    charge would reveal the held-out maximum by division by the known schedule.
    The private remainder affects admission only: observers can see its cost
    stop, as with validation today, but no amount derived from its price.
    """
    from .spend import _lock, _rows

    record = harness_record.for_run(run_id, root)
    if record is None or (raw := record.read(_KEY)) is None:
        return
    with _lock(record.directory / "spend.lock"):
        charges = json.loads(record.read(_KEY) or raw)
        highest = _highest(_rows(run_id, root))
        # Uncapped, unmetered test/evaluation callables still consume iterations.
        # Capped selection fails closed in check_budget when no estimate exists.
        highest = highest if highest is not None else 0.0
        rows = [
            json.loads(line) for line in (record.read("spend.jsonl") or "").splitlines()
        ]
        # Allowlist visible training kinds, before _rows can fill in any private
        # checkpoint data. Confirmation is validation today; future held-out
        # kinds must not silently become public price observations either.
        public_highest = (
            _highest(
                [
                    row
                    for row in rows
                    if row["kind"] in {"baseline", "training", "gate", "probe"}
                ]
            )
            or 0.0
        )
        indexed = {row["eval_id"]: row for row in rows if row["kind"] == KIND}
        for key, charge in charges.items():
            row = indexed.get(key)
            if row is None:
                row = {"eval_id": key, "kind": KIND, "total_dollars": 0.0}
                rows.append(row)
            if charge["active"]:
                row["total_dollars"] = max(
                    row["total_dollars"], charge["rollouts"] * public_highest
                )
                charge["total_dollars"] = max(
                    charge.get("total_dollars", 0.0), charge["rollouts"] * highest
                )
        # Persist the enforced total first. Admission subtracts the actual
        # public row, so a crash between these writes cannot lose the remainder.
        record.write(_KEY, json.dumps(charges))
        content = "".join(json.dumps(row) + "\n" for row in rows)
        if content != record.read("spend.jsonl"):
            record.write("spend.jsonl", content)


def private_rows(
    run_id: str, root: Path | None, public_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Unpublished charge remainders, for harness admission under spend.lock."""
    record = harness_record.for_run(run_id, root)
    if record is None or (raw := record.read(_KEY)) is None:
        return []
    published = {row["eval_id"]: row["total_dollars"] for row in public_rows}
    return [
        {
            "eval_id": key + "-private",
            "kind": KIND,
            "total_dollars": max(
                0.0, charge.get("total_dollars", 0.0) - published.get(key, 0.0)
            ),
        }
        for key, charge in json.loads(raw).items()
    ]


def consume(state: RunState, root: Path, lanes: Sequence[LaneState]) -> None:
    """Account once per harness generation/lane, before scoring any proposal."""
    from .lanes import validate_selection_lanes

    if not state.heldout_required or not state.lanes:
        return
    record = harness_record.for_run(state.run_id, root)
    if record is None:
        raise typer.BadParameter("Lane training accounting requires harness access.")
    validate_selection_lanes(lanes, state.lanes)
    charges = json.loads(record.read(_KEY) or "{}")
    evaluations, rollouts = _schedule(state, root)
    for lane in lanes:
        if lane.status != "awaiting_selection":
            continue
        # Lane iteration numbers and sample lists are reflector-writable. The
        # frozen baseline eval IDs identify a generation exclusively in the harness.
        identity = json.dumps([state.reflection_baseline_eval_ids, lane.lane])
        key = "lane-training-" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        charges.setdefault(
            key, {"evaluations": evaluations, "rollouts": rollouts, "active": True}
        )
    record.write(_KEY, json.dumps(charges))
    refresh(state.run_id, root)
    ledger = ParetoLog(state.run_id, root)
    present = {
        (row.extra.get("eval_id"), row.extra.get("charge_index"))
        for row in ledger.iter_rows()
        if row.extra.get("row_scope") == KIND
    }
    # The ledger, not a later checkpoint, owns deduplication. A crash between
    # any two writes repairs missing rows without buying another budget.
    for key, charge in charges.items():
        for index in range(charge["evaluations"]):
            if (key, index) not in present:
                ledger.append(
                    ParetoRow(
                        candidate_id=key,
                        commit_sha=None,
                        component_overrides_id=None,
                        minibatch_id="",
                        per_case_scores={},
                        mean_score=0.0,
                        status=KIND,
                        summary="Estimated lane training budget",
                        timestamp=utc_now_iso(),
                        extra={
                            "row_scope": KIND,
                            "selectable": False,
                            "eval_id": key,
                            "charge_index": index,
                        },
                    )
                )
    check_budget(state, root)


def check_budget(state: RunState, root: Path, *, next_iteration: bool = False) -> None:
    """Keep the incumbent on an exhausted or unaffordable estimated budget.

    Pre-rebaseline refan projection is a heuristic using the previous B and N.
    The binding check at emit uses the new frozen schedule before dispatch, so
    at that price the next lane charges fit the remaining cap.
    A newly more expensive candidate may raise charges
    at select: the residual overshoot is at most one iteration's revised lane
    charges plus the existing metered-rollout overshoot. This assumes work stays
    within consumed verdict schedules and the observed high bounds its rollout
    costs; it is not a provider spending limit. The first fan-out is already in
    flight when consumed and has the same one-iteration charge allowance.
    """
    from .spend import _finish_cost_stop, _lock, _report, _rows
    from ..spend import COST_STOP_REASON

    if (
        state.status == "done"
        or not state.heldout_required
        or not state.lanes
        or state.max_token_cost is None
    ):
        return
    record = harness_record.for_run(state.run_id, root)
    if record is None:
        raise typer.BadParameter("Lane training accounting requires harness access.")
    with _lock(record.directory / "spend.lock"):
        rows = _rows(state.run_id, root)
        highest = _highest(rows)
        spent = _report(rows, state.max_token_cost)["total_dollars"]
    projected = 0.0
    if next_iteration and highest is not None:
        projected = state.lanes * _schedule(state, root)[1] * highest
    if (
        highest is None
        or spent >= state.max_token_cost
        or spent + projected > state.max_token_cost
    ):
        reason = (
            COST_STOP_REASON
            if highest is not None
            else "No harness-metered cost for lane training estimate"
        )
        with _lock(record.directory / "spend.lock"):
            stop_id = "lane-training-budget-stop"
            if not any(row["eval_id"] == stop_id for row in _rows(state.run_id, root)):
                record.write(
                    "spend.jsonl",
                    json.dumps(
                        {
                            "eval_id": stop_id,
                            "kind": "lane_training_stop",
                            "total_dollars": 0.0,
                            "max_token_cost": state.max_token_cost,
                            "stop_reason": reason,
                        }
                    )
                    + "\n",
                    append=True,
                )
        _finish_cost_stop(state.run_id, root, state, reason, state.max_token_cost)
        raise typer.Exit(code=70)


def seal(state: RunState, root: Path) -> None:
    """Later candidates must not reprice a completed selection's charges."""
    if not state.heldout_required or not state.lanes:
        return
    record = harness_record.for_run(state.run_id, root)
    if record is None or (raw := record.read(_KEY)) is None:
        return
    charges = json.loads(raw)
    for charge in charges.values():
        charge["active"] = False
    record.write(_KEY, json.dumps(charges))

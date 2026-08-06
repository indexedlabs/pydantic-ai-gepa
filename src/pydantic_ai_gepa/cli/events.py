"""Managed-run event bus: one JSON file per event, `gepa next` / `gepa ack`.

Implements pydanticaigepa-spec-fmc. Producers create one file per event under
``runs/<run_id>/events/`` with exclusive create — the filename *is* the event
id (``<zero-padded ms timestamp>-<producer id>-<per-producer seq>``), so
uniqueness holds by construction with no shared read-modify-write
(pydanticaigepa-dec-f1s). The consumer cursor lives in
``runs/<run_id>/cursor.json`` and holds the highest acked event id; delivery
order is lexicographic id order.

Every ``gepa next`` first runs a reaper pass (pydanticaigepa-dec-pm3): given a
lane-scan result it synthesizes ``lane_stalled`` / ``selection_due`` events
idempotently keyed by (lane, lease epoch). Lane-state scanning lands with the
lane lifecycle (pydanticaigepa-task-xcb); :func:`scan_lanes` is the stubbed
seam the real scanner will replace.

CLI exit codes (also documented in each verb's help text): 0 event delivered
or ack accepted, 1 run resolution/usage error, 3 no pending events, 4
``--wait`` timeout elapsed, 5 ack rejected (non-oldest unacked event).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import typer

from .layout import latest_run_id, run_dir
from .runs import utc_now_iso

EventType = Literal[
    "lane_ready",
    "verdict",
    "selection_due",
    "merge_opportunity",
    "lane_stalled",
    "budget_low",
    "run_done",
]

EVENT_TYPES: tuple[str, ...] = (
    "lane_ready",
    "verdict",
    "selection_due",
    "merge_opportunity",
    "lane_stalled",
    "budget_low",
    "run_done",
)

# Payload key sets per the spec-fmc Interface block. Event payloads carry
# paths and scalars only — trace or diff content never goes inline.
EVENT_PAYLOAD_FIELDS: dict[str, frozenset[str]] = {
    "lane_ready": frozenset({"packet_path", "worktree_path"}),
    "verdict": frozenset({"verdict", "delta", "comparison_path"}),
    "selection_due": frozenset({"iteration", "resolved_lanes", "straggler_lanes"}),
    "merge_opportunity": frozenset(
        {"lane_a", "lane_b", "branch_a", "branch_b", "diff_stat_path"}
    ),
    "lane_stalled": frozenset({"reason", "lease_epoch"}),
    "budget_low": frozenset({"remaining_evals"}),
    "run_done": frozenset({"final_report_path"}),
}

# Run-scoped events carry ``lane=None``; everything else is lane-scoped.
RUN_SCOPED_TYPES: frozenset[str] = frozenset(
    {"selection_due", "merge_opportunity", "budget_low", "run_done"}
)

EVENTS_DIRNAME = "events"
CURSOR_FILENAME = "cursor.json"

REAPER_PRODUCER_ID = "reaper"

# `gepa next` / `gepa ack` exit codes — distinct per spec-fmc.
EXIT_DELIVERED = 0
EXIT_NONE_PENDING = 3
EXIT_TIMEOUT = 4
EXIT_ACK_REJECTED = 5

_TIMESTAMP_WIDTH = 13  # zero-padded epoch milliseconds
_SEQ_WIDTH = 6
_PRODUCER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EVENT_ID_RE = re.compile(
    rf"^(\d{{{_TIMESTAMP_WIDTH}}})-([A-Za-z0-9][A-Za-z0-9._-]*)-(\d{{{_SEQ_WIDTH}}})$"
)

# Per-process (run dir, producer id) -> next sequence number, guarded by a lock
# so in-process producers (threads) allocate distinct ids without rescanning.
_seq_lock = threading.Lock()
_seq_counters: dict[tuple[Path, str], int] = {}


class EventError(ValueError):
    """Raised when an event draft or producer id fails validation."""


class AckRejected(RuntimeError):
    """Raised when acking anything other than the oldest unacked event."""


# ----------------------------- event records ----------------------------


def _validate_scalar_or_scalar_list(value: Any, *, field_name: str) -> None:
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_scalar_or_scalar_list(item, field_name=field_name)
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    raise EventError(
        f"Payload field {field_name!r} must be a path or scalar (str, int, "
        f"float, bool, null) or a list of those; got {type(value).__name__}. "
        "Inline content (nested objects, traces, diffs) is never allowed."
    )


def _validate_draft(type: str, lane: str | None, payload: dict[str, Any]) -> None:
    if type not in EVENT_PAYLOAD_FIELDS:
        raise EventError(
            f"Unknown event type {type!r}; expected one of: {', '.join(EVENT_TYPES)}."
        )
    expected = EVENT_PAYLOAD_FIELDS[type]
    missing = expected - payload.keys()
    extra = payload.keys() - expected
    if missing:
        raise EventError(
            f"Event type {type!r} is missing payload fields: "
            f"{', '.join(sorted(missing))}."
        )
    if extra:
        raise EventError(
            f"Event type {type!r} got unknown payload fields: "
            f"{', '.join(sorted(extra))} (payload schema is fixed by spec)."
        )
    for key, value in payload.items():
        _validate_scalar_or_scalar_list(value, field_name=key)
    if type in RUN_SCOPED_TYPES:
        if lane is not None:
            raise EventError(f"Run-scoped event {type!r} must carry lane=None.")
    elif lane is None:
        raise EventError(f"Lane-scoped event {type!r} requires a lane id.")


@dataclass(frozen=True)
class EventDraft:
    """Producer-supplied event content; ``emit`` assigns the id and timestamp."""

    type: EventType
    lane: str | None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_draft(self.type, self.lane, self.payload)


@dataclass(frozen=True)
class Event:
    """A persisted event record (one JSON file under ``runs/<run_id>/events/``)."""

    id: str
    type: EventType
    ts: str
    lane: str | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "ts": self.ts,
            "lane": self.lane,
            "payload": dict(self.payload),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Event:
        return Event(
            id=str(data["id"]),
            type=str(data["type"]),  # type: ignore[arg-type]
            ts=str(data["ts"]),
            lane=data.get("lane"),
            payload=dict(data.get("payload", {})),
        )


# ----------------------------- storage paths ----------------------------


def events_dir(run_id: str, root: Path | None = None) -> Path:
    return run_dir(run_id, root) / EVENTS_DIRNAME


def cursor_path(run_id: str, root: Path | None = None) -> Path:
    return run_dir(run_id, root) / CURSOR_FILENAME


def _read_event(path: Path) -> Event:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("id", path.name)
    return Event.from_dict(data)


def list_events(run_id: str, root: Path | None = None) -> list[Event]:
    """Return every event on the run's bus in lexicographic id (delivery) order."""
    base = events_dir(run_id, root)
    if not base.is_dir():
        return []
    events = [
        _read_event(path) for path in sorted(base.iterdir(), key=lambda p: p.name)
    ]
    events.sort(key=lambda event: event.id)
    return events


# ----------------------------- cursor -----------------------------------


def load_cursor(run_id: str, root: Path | None = None) -> str | None:
    """Return the highest acked event id, or ``None`` when nothing was acked."""
    path = cursor_path(run_id, root)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    acked = data.get("acked_id")
    return str(acked) if acked else None


def advance_cursor(run_id: str, event_id: str, root: Path | None = None) -> None:
    """Persist ``event_id`` as the highest acked id (atomic tmp-file replace)."""
    path = cursor_path(run_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"acked_id": event_id}, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise


# ----------------------------- producer API -----------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


def _scan_next_seq(events_path: Path, producer_id: str) -> int:
    """Derive the next per-producer seq from existing files (fresh processes)."""
    next_seq = 0
    if not events_path.is_dir():
        return next_seq
    infix = f"-{producer_id}-"
    for path in events_path.iterdir():
        match = _EVENT_ID_RE.match(path.name)
        if match and match.group(2) == producer_id:
            next_seq = max(next_seq, int(match.group(3)) + 1)
        elif infix in path.name:
            # Defensive: ignore unparseable ids; seq allocation only trusts
            # well-formed producer ids.
            continue
    return next_seq


def _next_seq(events_path: Path, producer_id: str) -> int:
    key = (events_path, producer_id)
    with _seq_lock:
        if key not in _seq_counters:
            _seq_counters[key] = _scan_next_seq(events_path, producer_id)
        seq = _seq_counters[key]
        _seq_counters[key] = seq + 1
    return seq


def _release_seq(events_path: Path, producer_id: str, seq: int) -> None:
    """Return an unused seq to the pool after a create collision or write error."""
    key = (events_path, producer_id)
    with _seq_lock:
        if _seq_counters.get(key, 0) == seq + 1:
            _seq_counters[key] = seq


def _bump_seq_past(events_path: Path, producer_id: str, seq: int) -> None:
    key = (events_path, producer_id)
    with _seq_lock:
        if _seq_counters.get(key, 0) <= seq:
            _seq_counters[key] = seq + 1


def emit(
    run_id: str,
    producer_id: str,
    draft: EventDraft,
    *,
    root: Path | None = None,
) -> str:
    """Create the event file with exclusive create and return the event id.

    Uniqueness holds by construction: the filename embeds the zero-padded ms
    timestamp, the producer id, and a per-producer sequence number; a create
    collision (same ms) retries with the seq bumped. Payload validation happens
    at ``EventDraft`` construction, so invalid payloads are rejected before any
    file is touched.
    """
    if not _PRODUCER_ID_RE.match(producer_id):
        raise EventError(
            f"Invalid producer id {producer_id!r}; expected "
            f"{_PRODUCER_ID_RE.pattern!r} (filename-safe)."
        )
    path = events_dir(run_id, root)
    path.mkdir(parents=True, exist_ok=True)
    ts_ms = f"{_now_ms():0{_TIMESTAMP_WIDTH}d}"
    body = {
        "type": draft.type,
        "ts": utc_now_iso(),
        "lane": draft.lane,
        "payload": dict(draft.payload),
    }
    while True:
        seq = _next_seq(path, producer_id)
        event_id = f"{ts_ms}-{producer_id}-{seq:0{_SEQ_WIDTH}d}"
        file_path = path / event_id
        record = {"id": event_id, **body}
        try:
            with file_path.open("x", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
                handle.write("\n")
            return event_id
        except FileExistsError:
            # Timestamp collision with another emit: retry with the seq bumped.
            _bump_seq_past(path, producer_id, seq)
        except OSError:
            _release_seq(path, producer_id, seq)
            raise


# ----------------------------- consumer API -----------------------------


def next_event(run_id: str, root: Path | None = None) -> Event | None:
    """Return the oldest unacked event (id > cursor), or ``None`` if none."""
    cursor = load_cursor(run_id, root)
    for event in list_events(run_id, root):
        if cursor is None or event.id > cursor:
            return event
    return None


def ack(run_id: str, event_id: str, root: Path | None = None) -> bool:
    """Ack ``event_id``. Returns True when the cursor advanced, False on a no-op.

    Acking is idempotent: an id at or below the persisted cursor is a no-op
    success. Acking anything other than the oldest unacked event raises
    :class:`AckRejected` (the CLI maps it to exit code 5).
    """
    cursor = load_cursor(run_id, root)
    if cursor is not None and event_id <= cursor:
        return False
    pending = next_event(run_id, root)
    if pending is None:
        raise AckRejected(
            f"No pending event to ack (cursor already at {cursor!r}); "
            f"refusing to ack {event_id!r}."
        )
    if event_id != pending.id:
        raise AckRejected(
            f"Refusing to ack {event_id!r}: the oldest unacked event is "
            f"{pending.id!r}. Events must be acked in delivery order."
        )
    advance_cursor(run_id, event_id, root)
    return True


# ----------------------------- reaper (lazy) ----------------------------


@dataclass(frozen=True)
class LaneScanResult:
    """One lane's liveness scan row (produced by the lane-state scanner).

    ``stalled_reason`` is set when the reaper should synthesize a
    ``lane_stalled`` event: the eval pid is dead, the heartbeat is stale, or
    the reflection lease expired. ``lease_epoch`` is the idempotency key — a
    lane that stalls again later reports under a fresh epoch.
    """

    lane: str
    lease_epoch: int
    stalled_reason: str | None = None


@dataclass(frozen=True)
class SelectionDueSignal:
    """Run-scoped straggler-timeout finding from a scan pass."""

    iteration: int
    resolved_lanes: tuple[str, ...] = ()
    straggler_lanes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaneScan:
    """Aggregate scan result handed to the reaper pass."""

    lanes: tuple[LaneScanResult, ...] = ()
    selection_due: SelectionDueSignal | None = None


def scan_lanes(run_id: str, root: Path | None = None) -> LaneScan:
    """Scan lane leases, heartbeat freshness, and recorded pids.

    Stubbed seam: lane state lands with the lane lifecycle
    (pydanticaigepa-task-xcb), so until then every scan reports no lanes and
    no selection pressure. The reaper pass consumes this result object; tests
    and slice 3 inject real scans the same way.
    """
    return LaneScan()


def run_reaper_pass(
    run_id: str,
    scan: LaneScan | None = None,
    *,
    root: Path | None = None,
    producer_id: str = REAPER_PRODUCER_ID,
) -> list[str]:
    """Synthesize ``lane_stalled`` / ``selection_due`` events from a scan result.

    Idempotent: before emitting, existing event files are checked for an event
    with the same key — ``(lane_stalled, lane, payload.lease_epoch)`` or
    ``(selection_due, lane=None, payload.iteration)`` — so no matter how many
    passes run, a (lane, lease epoch) pair never produces a duplicate.
    Returns the ids of events emitted by this pass.
    """
    if scan is None:
        scan = scan_lanes(run_id, root)
    existing = list_events(run_id, root)
    emitted: list[str] = []
    for lane in scan.lanes:
        if lane.stalled_reason is None:
            continue
        duplicate = any(
            event.type == "lane_stalled"
            and event.lane == lane.lane
            and event.payload.get("lease_epoch") == lane.lease_epoch
            for event in existing
        )
        if duplicate:
            continue
        emitted.append(
            emit(
                run_id,
                producer_id,
                EventDraft(
                    type="lane_stalled",
                    lane=lane.lane,
                    payload={
                        "reason": lane.stalled_reason,
                        "lease_epoch": lane.lease_epoch,
                    },
                ),
                root=root,
            )
        )
    if scan.selection_due is not None:
        signal = scan.selection_due
        duplicate = any(
            event.type == "selection_due"
            and event.lane is None
            and event.payload.get("iteration") == signal.iteration
            for event in existing
        )
        if not duplicate:
            emitted.append(
                emit(
                    run_id,
                    producer_id,
                    EventDraft(
                        type="selection_due",
                        lane=None,
                        payload={
                            "iteration": signal.iteration,
                            "resolved_lanes": list(signal.resolved_lanes),
                            "straggler_lanes": list(signal.straggler_lanes),
                        },
                    ),
                    root=root,
                )
            )
    return emitted


# ----------------------------- CLI verbs --------------------------------

_NEXT_HELP = """Deliver the oldest unacked event from the run's event bus.

Runs a reaper pass first (scan lane leases/heartbeats/pids, synthesize
lane_stalled / selection_due), then returns the oldest event past the
persisted consumer cursor. Unacked events are redelivered verbatim, so
crashing between `gepa next` and `gepa ack` loses nothing.

Exit codes: 0 event delivered; 1 run resolution/usage error; 3 no pending
events; 4 --wait timeout elapsed (distinct from "none pending").
"""

_ACK_HELP = """Ack an event, advancing the persisted consumer cursor past it.

Events must be acked in delivery order: acking anything other than the
oldest unacked event is rejected. Acking is idempotent — acking an id at
or below the cursor is a no-op success.

Exit codes: 0 cursor advanced or idempotent no-op; 1 run resolution/usage
error; 5 ack rejected (not the oldest unacked event).
"""


def _resolve_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    latest = latest_run_id()
    if latest is None:
        typer.echo(
            "No runs found under .gepa/runs/. Start one with `gepa run start`.",
            err=True,
        )
        raise typer.Exit(code=1)
    return latest


def _emit_delivered(
    event: Event,
    *,
    as_json: bool,
    cursor_before: str | None,
) -> None:
    if as_json:
        typer.echo(json.dumps(event.to_dict(), indent=2))
        return
    lane = event.lane if event.lane is not None else "-"
    typer.echo(f"{event.id}\t{event.type}\tlane={lane}")
    typer.echo(
        f"unacked; ack with `gepa ack {event.id}` "
        f"(cursor at {cursor_before or 'start'})"
    )


def next_command(
    run_id: str | None = typer.Option(None, "--run-id", help="Defaults to latest run."),
    wait: bool = typer.Option(
        False, "--wait", help="Long-poll until an event arrives or --timeout elapses."
    ),
    timeout: float = typer.Option(
        30.0, "--timeout", help="Seconds to long-poll with --wait."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Print the full event record as JSON."
    ),
) -> None:
    if timeout < 0:
        typer.echo("--timeout must be non-negative.", err=True)
        raise typer.Exit(code=1)
    active_run = _resolve_run_id(run_id)
    run_reaper_pass(active_run)
    deadline = time.monotonic() + timeout
    while True:
        event = next_event(active_run)
        if event is not None:
            _emit_delivered(
                event, as_json=as_json, cursor_before=load_cursor(active_run)
            )
            raise typer.Exit(code=EXIT_DELIVERED)
        if not wait:
            typer.echo("No pending events.", err=True)
            raise typer.Exit(code=EXIT_NONE_PENDING)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            typer.echo(f"No event arrived within {timeout:g}s.", err=True)
            raise typer.Exit(code=EXIT_TIMEOUT)
        time.sleep(min(0.05, remaining))


next_command.__doc__ = _NEXT_HELP


def ack_command(
    event_id: str = typer.Argument(..., help="Event id (the event filename)."),
    run_id: str | None = typer.Option(None, "--run-id", help="Defaults to latest run."),
) -> None:
    active_run = _resolve_run_id(run_id)
    try:
        advanced = ack(active_run, event_id)
    except AckRejected as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=EXIT_ACK_REJECTED) from exc
    if advanced:
        typer.echo(f"Acked {event_id}.")
    else:
        typer.echo(f"{event_id} already acked; cursor unchanged.")


ack_command.__doc__ = _ACK_HELP

"""Tests for the managed-run event bus (`gepa next` / `gepa ack`, spec-fmc)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import pytest
from click.testing import Result
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.events import (
    EXIT_ACK_REJECTED,
    EXIT_NONE_PENDING,
    EXIT_TIMEOUT,
    AckRejected,
    EventDraft,
    EventError,
    EventType,
    LaneScan,
    LaneScanResult,
    SelectionDueSignal,
    ack,
    cursor_path,
    emit,
    events_dir,
    list_events,
    load_cursor,
    next_event,
    run_reaper_pass,
    scan_lanes,
)
from pydantic_ai_gepa.cli.layout import ensure_layout, new_run_id


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.chdir(tmp_path)
    ensure_layout(tmp_path)
    yield tmp_path


@pytest.fixture
def run_id(repo: Path) -> str:
    return new_run_id()


def _run(*argv: str) -> Result:
    return CliRunner().invoke(gepa_app, list(argv))


def _lane_ready_payload() -> dict[str, str]:
    return {
        "packet_path": "runs/x/lanes/lane-a/packet.json",
        "worktree_path": "worktrees/lane-a",
    }


def _dead_lane_scan(
    lane: str = "lane-a", lease_epoch: int = 1, reason: str = "heartbeat_stale"
) -> LaneScan:
    return LaneScan(
        lanes=(
            LaneScanResult(lane=lane, lease_epoch=lease_epoch, stalled_reason=reason),
        )
    )


# ----------------------------- emit / storage ---------------------------


def test_event_files_land_one_per_event_under_events_dir(
    repo: Path, run_id: str
) -> None:
    first = emit(
        run_id, "lane-a", EventDraft("lane_ready", "lane-a", _lane_ready_payload())
    )
    second = emit(
        run_id,
        "lane-b",
        EventDraft(
            "verdict",
            "lane-b",
            {"verdict": "keep", "delta": 0.5, "comparison_path": "p"},
        ),
    )
    files = sorted(p.name for p in events_dir(run_id, repo).iterdir())
    assert files == sorted([first, second])
    record = json.loads((events_dir(run_id, repo) / first).read_text())
    assert record["id"] == first
    assert record["type"] == "lane_ready"
    assert record["lane"] == "lane-a"
    assert record["payload"] == _lane_ready_payload()
    assert record["ts"]


def test_run_scoped_events_require_null_lane(repo: Path, run_id: str) -> None:
    with pytest.raises(EventError, match="lane=None"):
        EventDraft("run_done", "lane-a", {"final_report_path": "report.md"})
    with pytest.raises(EventError, match="requires a lane"):
        EventDraft("lane_stalled", None, {"reason": "x", "lease_epoch": 1})


def test_emit_rejects_inline_non_scalar_payload(repo: Path, run_id: str) -> None:
    with pytest.raises(EventError, match="path or scalar"):
        EventDraft(
            "verdict",
            "lane-a",
            {"verdict": "keep", "delta": 0.1, "comparison_path": {"nested": "inline"}},
        )
    # Validation happens at draft construction — nothing touches the disk.
    assert not events_dir(run_id, repo).exists()


def test_emit_rejects_missing_and_unknown_payload_fields(
    repo: Path, run_id: str
) -> None:
    with pytest.raises(EventError, match="missing payload fields"):
        EventDraft("budget_low", None, {})
    with pytest.raises(EventError, match="unknown payload fields"):
        EventDraft(
            "lane_ready", "lane-a", {**_lane_ready_payload(), "extra_trace": "..."}
        )


def test_per_producer_seq_survives_fresh_state(repo: Path, run_id: str) -> None:
    """A new producer process derives its seq from existing files (no 0-reuse)."""
    from pydantic_ai_gepa.cli import events as events_mod

    first = emit(
        run_id, "lane-a", EventDraft("lane_ready", "lane-a", _lane_ready_payload())
    )
    events_mod._seq_counters.clear()  # simulate a fresh producer process
    second = emit(
        run_id, "lane-a", EventDraft("lane_ready", "lane-a", _lane_ready_payload())
    )
    assert first != second
    assert int(second.rsplit("-", 1)[1]) == int(first.rsplit("-", 1)[1]) + 1


def test_concurrent_producers_create_every_event_exactly_once(
    repo: Path, run_id: str
) -> None:
    """N threaded producers: exclusive create => every event exactly once,
    all ids unique, and replay order is lexicographic."""
    from concurrent.futures import ThreadPoolExecutor

    producers = [f"lane-{chr(ord('a') + i)}" for i in range(8)]
    per_producer = 10

    def produce(producer: str) -> list[str]:
        return [
            emit(
                run_id,
                producer,
                EventDraft("lane_ready", producer, _lane_ready_payload()),
            )
            for _ in range(per_producer)
        ]

    with ThreadPoolExecutor(max_workers=len(producers)) as pool:
        results = list(pool.map(produce, producers))

    emitted = [event_id for ids in results for event_id in ids]
    assert len(emitted) == len(producers) * per_producer
    assert len(set(emitted)) == len(emitted)
    files = [p.name for p in events_dir(run_id, repo).iterdir()]
    assert sorted(files) == sorted(emitted)
    assert len(files) == len(emitted)  # each event created exactly once
    replayed = [event.id for event in list_events(run_id, repo)]
    assert replayed == sorted(emitted)


# ----------------------------- next / redelivery ------------------------


def test_next_redelivers_verbatim_until_acked(repo: Path, run_id: str) -> None:
    """Crash between next and ack => the same event comes back byte-identical."""
    event_id = emit(
        run_id, "lane-a", EventDraft("lane_ready", "lane-a", _lane_ready_payload())
    )
    first = _run("next", "--run-id", run_id, "--json")
    assert first.exit_code == 0, first.output
    second = _run("next", "--run-id", run_id, "--json")  # consumer "restarted"
    assert second.exit_code == 0
    assert second.output == first.output
    record = json.loads(first.output)
    assert record["id"] == event_id

    acked = _run("ack", event_id, "--run-id", run_id)
    assert acked.exit_code == 0, acked.output
    drained = _run("next", "--run-id", run_id, "--json")
    assert drained.exit_code == EXIT_NONE_PENDING


def test_next_delivers_oldest_unacked_in_lexicographic_order(
    repo: Path, run_id: str
) -> None:
    ids = [
        emit(
            run_id,
            f"lane-{c}",
            EventDraft("lane_ready", f"lane-{c}", _lane_ready_payload()),
        )
        for c in "abc"
    ]
    delivered = []
    for expected in sorted(ids):
        result = _run("next", "--run-id", run_id, "--json")
        assert result.exit_code == 0
        delivered.append(json.loads(result.output)["id"])
        assert _run("ack", expected, "--run-id", run_id).exit_code == 0
    assert delivered == sorted(ids)


def test_next_wait_times_out_with_distinct_exit_code(repo: Path, run_id: str) -> None:
    started = time.monotonic()
    result = _run("next", "--run-id", run_id, "--wait", "--timeout", "0.2")
    elapsed = time.monotonic() - started
    assert result.exit_code == EXIT_TIMEOUT
    assert "timeout" in result.output.lower() or "within" in result.output.lower()
    assert elapsed >= 0.2
    assert elapsed < 2.0  # never blocks past the timeout


def test_next_without_wait_reports_none_pending(repo: Path, run_id: str) -> None:
    result = _run("next", "--run-id", run_id)
    assert result.exit_code == EXIT_NONE_PENDING
    assert "No pending events" in result.output


def test_next_wait_delivers_event_arriving_mid_poll(repo: Path, run_id: str) -> None:
    import threading

    def delayed_emit() -> None:
        time.sleep(0.15)
        emit(
            run_id, "lane-a", EventDraft("lane_ready", "lane-a", _lane_ready_payload())
        )

    thread = threading.Thread(target=delayed_emit)
    thread.start()
    result = _run("next", "--run-id", run_id, "--wait", "--timeout", "2", "--json")
    thread.join()
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["type"] == "lane_ready"


def test_next_human_output_names_ack_command(repo: Path, run_id: str) -> None:
    event_id = emit(
        run_id, "lane-a", EventDraft("lane_ready", "lane-a", _lane_ready_payload())
    )
    result = _run("next", "--run-id", run_id)
    assert result.exit_code == 0
    assert event_id in result.output
    assert f"gepa ack {event_id}" in result.output


def test_next_without_run_id_picks_latest(repo: Path) -> None:
    older = "20260101T000000Z-aaaaaaaa"
    newer = "20260102T000000Z-bbbbbbbb"
    emit(
        older,
        "lane-a",
        EventDraft("lane_ready", "lane-a", _lane_ready_payload()),
        root=repo,
    )
    emit(
        newer,
        "lane-b",
        EventDraft("lane_ready", "lane-b", _lane_ready_payload()),
        root=repo,
    )
    result = _run("next", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["lane"] == "lane-b"


def test_next_errors_when_no_runs(repo: Path) -> None:
    result = _run("next")
    assert result.exit_code == 1
    assert "No runs found" in result.output


# ----------------------------- ack semantics ----------------------------


def test_ack_advances_persisted_cursor(repo: Path, run_id: str) -> None:
    event_id = emit(
        run_id, "lane-a", EventDraft("lane_ready", "lane-a", _lane_ready_payload())
    )
    assert load_cursor(run_id, repo) is None
    assert ack(run_id, event_id, root=repo) is True
    assert load_cursor(run_id, repo) == event_id


def test_ack_of_non_oldest_is_rejected(repo: Path, run_id: str) -> None:
    first = emit(
        run_id, "lane-a", EventDraft("lane_ready", "lane-a", _lane_ready_payload())
    )
    second = emit(
        run_id, "lane-b", EventDraft("lane_ready", "lane-b", _lane_ready_payload())
    )
    result = _run("ack", second, "--run-id", run_id)
    assert result.exit_code == EXIT_ACK_REJECTED
    assert first in result.output  # names the oldest unacked event
    assert load_cursor(run_id, repo) is None


def test_second_ack_of_same_id_is_noop_success(repo: Path, run_id: str) -> None:
    event_id = emit(
        run_id, "lane-a", EventDraft("lane_ready", "lane-a", _lane_ready_payload())
    )
    assert _run("ack", event_id, "--run-id", run_id).exit_code == 0
    result = _run("ack", event_id, "--run-id", run_id)
    assert result.exit_code == 0, result.output
    assert "already acked" in result.output
    assert load_cursor(run_id, repo) == event_id


def test_ack_unknown_id_is_rejected(repo: Path, run_id: str) -> None:
    result = _run("ack", "0000000000000-ghost-000000", "--run-id", run_id)
    assert result.exit_code == EXIT_ACK_REJECTED


# ----------------------------- reaper pass ------------------------------


def test_scan_lanes_stub_reports_no_lanes(repo: Path, run_id: str) -> None:
    scan = scan_lanes(run_id, repo)
    assert scan.lanes == ()
    assert scan.selection_due is None
    assert run_reaper_pass(run_id, root=repo) == []


def test_reaper_synthesizes_lane_stalled_exactly_once_per_lease_epoch(
    repo: Path, run_id: str
) -> None:
    scan = _dead_lane_scan()
    first_pass = run_reaper_pass(run_id, scan, root=repo)
    assert len(first_pass) == 1
    second_pass = run_reaper_pass(run_id, scan, root=repo)
    assert second_pass == []
    stalled = [e for e in list_events(run_id, repo) if e.type == "lane_stalled"]
    assert len(stalled) == 1
    assert stalled[0].lane == "lane-a"
    assert stalled[0].payload == {"reason": "heartbeat_stale", "lease_epoch": 1}


def test_reaper_emits_again_for_fresh_lease_epoch(repo: Path, run_id: str) -> None:
    run_reaper_pass(run_id, _dead_lane_scan(lease_epoch=1), root=repo)
    emitted = run_reaper_pass(run_id, _dead_lane_scan(lease_epoch=2), root=repo)
    assert len(emitted) == 1
    epochs = [
        e.payload["lease_epoch"]
        for e in list_events(run_id, repo)
        if e.type == "lane_stalled"
    ]
    assert sorted(epochs) == [1, 2]


def test_reaper_synthesizes_selection_due_with_resolved_and_stragglers(
    repo: Path, run_id: str
) -> None:
    scan = LaneScan(
        selection_due=SelectionDueSignal(
            iteration=3,
            resolved_lanes=("lane-a", "lane-b"),
            straggler_lanes=("lane-c",),
        )
    )
    emitted = run_reaper_pass(run_id, scan, root=repo)
    assert len(emitted) == 1
    assert run_reaper_pass(run_id, scan, root=repo) == []  # idempotent
    due = [e for e in list_events(run_id, repo) if e.type == "selection_due"]
    assert len(due) == 1
    assert due[0].lane is None
    assert due[0].payload == {
        "iteration": 3,
        "resolved_lanes": ["lane-a", "lane-b"],
        "straggler_lanes": ["lane-c"],
    }


def test_next_runs_reaper_pass_before_delivering(
    repo: Path, run_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic_ai_gepa.cli import events as events_mod

    monkeypatch.setattr(events_mod, "scan_lanes", lambda *a, **k: _dead_lane_scan())
    result = _run("next", "--run-id", run_id, "--json")
    assert result.exit_code == 0, result.output
    record = json.loads(result.output)
    assert record["type"] == "lane_stalled"
    assert record["lane"] == "lane-a"


def test_help_documents_exit_codes() -> None:
    for verb in ("next", "ack"):
        result = _run(verb, "--help")
        assert result.exit_code == 0
        assert "Exit codes:" in result.output
    assert "4" in _run("next", "--help").output  # timeout code documented
    assert "5" in _run("ack", "--help").output  # rejected-ack code documented


# ---------- adversarial-review hardening (PR #28 review) ----------


def _draft(
    event_type: EventType = "verdict", lane: str | None = "lane-1"
) -> EventDraft:
    payload = {"verdict": "accepted", "delta": 0.5, "comparison_path": "/tmp/c.json"}
    return EventDraft(type=event_type, lane=lane, payload=payload)


def test_partial_event_file_does_not_wedge_bus(tmp_path: Path) -> None:
    """A producer killed mid-emit leaves an ignored tmpfile / quarantined
    file — next, ack, and the reaper keep working."""
    ensure_layout(tmp_path)
    run = new_run_id()
    events_base = events_dir(run, tmp_path)
    events_base.mkdir(parents=True)

    # Torn event file with a valid event-id name (SIGKILL between create and flush).
    (events_base / "1786000000000-lane-1-000000").write_text('{"type": "verd')
    # Stray non-event files.
    (events_base / ".emit-xyz.tmp").write_text("{}")
    (events_base / "notes.txt").write_text("hello")

    good = emit(run, "lane-1", _draft(), root=tmp_path)
    event = next_event(run, tmp_path)
    assert event is not None and event.id == good
    assert ack(run, good, root=tmp_path) is True
    assert next_event(run, tmp_path) is None


def test_embedded_id_mismatch_loses_to_filename(tmp_path: Path) -> None:
    """The filename is the id (spec-fmc); a mismatched embedded id never
    hijacks delivery order."""
    ensure_layout(tmp_path)
    run = new_run_id()
    events_base = events_dir(run, tmp_path)
    events_base.mkdir(parents=True)
    (events_base / "1786000000000-lane-1-000000").write_text(
        json.dumps(
            {
                "id": "9999999999999-evil-000000",
                "type": "verdict",
                "ts": "2026-01-01T00:00:00+00:00",
                "lane": "lane-1",
                "payload": {
                    "verdict": "accepted",
                    "delta": 1.0,
                    "comparison_path": "/tmp/c.json",
                },
            }
        )
    )
    event = next_event(run, tmp_path)
    assert event is not None
    assert event.id == "1786000000000-lane-1-000000"


def test_corrupt_cursor_resets_to_start(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    run = new_run_id()
    event_id = emit(run, "lane-1", _draft(), root=tmp_path)
    cursor_path(run, tmp_path).write_text("{torn", encoding="utf-8")

    # The event is redelivered instead of the bus wedging.
    event = next_event(run, tmp_path)
    assert event is not None and event.id == event_id


def test_ack_nonexistent_id_below_cursor_rejected(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    run = new_run_id()
    first = emit(run, "lane-1", _draft(), root=tmp_path)
    assert ack(run, first, root=tmp_path) is True

    import pytest as _pytest

    bogus = (
        first[:-6] + "999999"
    )  # same prefix, never existed, sorts at/after cursor...
    # craft an id that sorts below the cursor
    bogus = "1000000000000-lane-1-000000"
    assert bogus < first
    with _pytest.raises(AckRejected):
        ack(run, bogus, root=tmp_path)
    # The real cursor id remains an idempotent no-op.
    assert ack(run, first, root=tmp_path) is False


def test_nonfinite_float_payload_rejected() -> None:
    import pytest as _pytest

    with _pytest.raises(EventError):
        EventDraft(
            type="verdict",
            lane="lane-1",
            payload={
                "verdict": "accepted",
                "delta": float("nan"),
                "comparison_path": "/tmp/c.json",
            },
        )
    with _pytest.raises(EventError):
        EventDraft(
            type="verdict",
            lane="lane-1",
            payload={
                "verdict": "accepted",
                "delta": float("inf"),
                "comparison_path": "/tmp/c.json",
            },
        )


def test_invalid_lane_id_rejected() -> None:
    import pytest as _pytest

    with _pytest.raises(EventError):
        EventDraft(
            type="verdict",
            lane="../../etc",
            payload={
                "verdict": "accepted",
                "delta": 0.1,
                "comparison_path": "/tmp/c.json",
            },
        )


def test_reaper_sentinel_dedupes_concurrent_passes(tmp_path: Path) -> None:
    """Two concurrent reaper passes over the same scan emit exactly one
    lane_stalled — the dedupe key is an exclusive-create sentinel, not a
    read (spec-fmc: no duplicate no matter how many passes run)."""
    import threading

    ensure_layout(tmp_path)
    run = new_run_id()
    scan = LaneScan(
        lanes=(LaneScanResult(lane="lane-1", lease_epoch=3, stalled_reason="pid dead"),)
    )

    emitted: list[list[str]] = []

    def pass_() -> None:
        emitted.append(run_reaper_pass(run, scan, root=tmp_path))

    threads = [threading.Thread(target=pass_) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    total = [event_id for batch in emitted for event_id in batch]
    assert len(total) == 1
    stalled = [
        event for event in list_events(run, tmp_path) if event.type == "lane_stalled"
    ]
    assert len(stalled) == 1
    assert stalled[0].payload["lease_epoch"] == 3

    # A fresh lease epoch re-emits (new sentinel key).
    scan2 = LaneScan(
        lanes=(LaneScanResult(lane="lane-1", lease_epoch=4, stalled_reason="pid dead"),)
    )
    assert len(run_reaper_pass(run, scan2, root=tmp_path)) == 1

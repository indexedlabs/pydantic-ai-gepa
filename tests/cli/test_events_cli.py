"""Tests for the managed-run event bus (`gepa next` / `gepa ack`, spec-fmc)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import pytest
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.events import (
    EXIT_ACK_REJECTED,
    EXIT_NONE_PENDING,
    EXIT_TIMEOUT,
    EventDraft,
    EventError,
    LaneScan,
    LaneScanResult,
    SelectionDueSignal,
    ack,
    emit,
    events_dir,
    list_events,
    load_cursor,
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


def _run(*argv: str) -> object:
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

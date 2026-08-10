"""End-to-end coverage for reflection lanes (spec-1do, task-xcb)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Iterator

import pytest
from click.testing import Result
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.lanes import (
    LaneState,
    lane_state_path,
    load_lane_state,
)
from pydantic_ai_gepa.cli.run import RunState

EVALUATE_MODULE_SOURCE = textwrap.dedent("""
    async def evaluate(case):
        from pathlib import Path

        return Path("score.txt").read_text(encoding="utf-8").strip()
""").lstrip()


def _run(*argv: str) -> Result:
    return CliRunner().invoke(gepa_app, list(argv))


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_payload(output: str) -> dict[str, object]:
    line = next(
        line
        for line in reversed(output.splitlines())
        if line.startswith("{") and '"run"' in line
    )
    return json.loads(line)["run"]


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    # Lane verbs set layout globals via --gepa-dir; save/restore them so the
    # override cannot leak into the next test's init.
    from pydantic_ai_gepa.cli import layout

    saved_dirname = layout._explicit_gepa_dirname
    saved_env = os.environ.get("GEPA_DIR")
    monkeypatch.delenv("GEPA_DIR", raising=False)

    module_dir = tmp_path / "task_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").touch()
    (module_dir / "evaluation.py").write_text(EVALUATE_MODULE_SOURCE, encoding="utf-8")
    (tmp_path / "score.txt").write_text("bad\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(
        "__pycache__/\n.gepa/runs/\n", encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    init_result = _run(
        "init",
        "--candidate-source",
        "git",
        "--evaluate",
        "task_pkg.evaluation:evaluate",
    )
    assert init_result.exit_code == 0, init_result.output
    (tmp_path / ".gepa" / "dataset.jsonl").write_text(
        json.dumps({"name": "case-1", "inputs": "x", "expected_output": "good"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / ".gepa" / "journal.jsonl").write_text(
        json.dumps({"timestamp": "t0", "content": "seed entry", "strategy": "seed"})
        + "\n",
        encoding="utf-8",
    )

    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "tests@example.com")
    _git(tmp_path, "config", "user.name", "GEPA Tests")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "Seed git candidate")

    yield tmp_path

    layout._explicit_gepa_dirname = saved_dirname
    if saved_env is None:
        os.environ.pop("GEPA_DIR", None)
    else:
        os.environ["GEPA_DIR"] = saved_env

    for name in list(sys.modules):
        if name.startswith("task_pkg"):
            sys.modules.pop(name, None)


def _start_lane_run(repo: Path, lanes: int = 2) -> dict[str, object]:
    result = _run(
        "run",
        "start",
        "--lanes",
        str(lanes),
        "--size",
        "1",
        "--acceptance-repetitions",
        "1",
    )
    assert result.exit_code == 0, result.output
    return _run_payload(result.output)


def _lane_state(repo: Path, run_id: str, lane: str) -> LaneState:
    return load_lane_state(repo, run_id, lane)


def test_run_start_lanes_fans_out(git_repo: Path) -> None:
    run = _start_lane_run(git_repo, lanes=2)
    run_id = str(run["run_id"])
    assert run["lanes"] == 2
    assert run["status"] == "running"
    assert run["next_command"] == f"gepa next --wait --run-id {run_id}"

    # Worktrees + branches cut from the frozen baseline commit.
    base_sha = _git(git_repo, "rev-parse", "HEAD")
    for lane in ("lane-1", "lane-2"):
        worktree = git_repo / "worktrees" / run_id / lane
        assert worktree.is_dir()
        assert _git(worktree, "rev-parse", "HEAD") == base_sha
        state = _lane_state(git_repo, run_id, lane)
        assert state.status == "paused_for_reflection"
        assert state.branch == f"gepa/lane/{run_id}/{lane}/{state.iteration}"
        assert Path(str(state.packet_path)).exists()

    # lane_ready events emitted with packet + worktree paths.
    events_dir = git_repo / ".gepa" / "runs" / run_id / "events"
    ids = sorted(p.name for p in events_dir.iterdir() if p.is_file())
    assert len(ids) == 2
    events = [json.loads((events_dir / eid).read_text()) for eid in ids]
    assert {event["type"] for event in events} == {"lane_ready"}
    for event in events:
        assert Path(event["payload"]["packet_path"]).exists()
        assert Path(event["payload"]["worktree_path"]).is_dir()


def test_run_start_lanes_rejects_dirty_primary(git_repo: Path) -> None:
    (git_repo / "score.txt").write_text("worse\n", encoding="utf-8")
    result = _run("run", "start", "--lanes", "2", "--size", "1")
    assert result.exit_code == 1
    assert "clean primary tree" in result.output


def test_run_start_lanes_rejects_component_mode(git_repo: Path) -> None:
    result = _run(
        "run",
        "start",
        "--lanes",
        "2",
        "--size",
        "1",
        "--candidate-source",
        "components",
    )
    assert result.exit_code == 2
    assert "git candidate mode" in result.output


def test_run_continue_errors_in_lane_run(git_repo: Path) -> None:
    run = _start_lane_run(git_repo)
    result = _run("run", "continue", "--run-id", str(run["run_id"]))
    assert result.exit_code == 1
    assert "gepa lane continue" in result.output
    assert "gepa run select" in result.output


def test_single_path_run_untouched(git_repo: Path) -> None:
    """--lanes absent -> existing synchronous path: no worktrees, no events."""
    result = _run("run", "start", "--size", "1", "--acceptance-repetitions", "1")
    assert result.exit_code == 0, result.output
    run = _run_payload(result.output)
    assert run["lanes"] == 0
    assert run["status"] == "paused_for_reflection"
    run_id = str(run["run_id"])
    assert not (git_repo / "worktrees").exists()
    assert not (git_repo / ".gepa" / "runs" / run_id / "events").exists()
    assert run["next_command"] == f"gepa run continue --run-id {run_id}"


def test_pre_lane_state_loads_with_lane_defaults(git_repo: Path) -> None:
    result = _run("run", "start", "--size", "1", "--acceptance-repetitions", "1")
    assert result.exit_code == 0, result.output
    run = _run_payload(result.output)
    state_path = Path(str(run["state_path"]))
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    for key in (
        "lanes",
        "heartbeat_interval_secs",
        "reflection_lease_secs",
        "eval_stall_timeout_secs",
        "straggler_timeout_secs",
        "journal_tail_lines",
    ):
        raw.pop(key, None)
    state = RunState.from_dict(raw)
    assert state.lanes == 0
    assert state.heartbeat_interval_secs == 10.0
    assert state.reflection_lease_secs == 1800.0
    assert state.eval_stall_timeout_secs == 600.0
    assert state.straggler_timeout_secs == 3600.0
    assert state.journal_tail_lines == 20


def test_packet_is_self_contained(git_repo: Path) -> None:
    run = _start_lane_run(git_repo, lanes=1)
    state = _lane_state(git_repo, str(run["run_id"]), "lane-1")
    packet = json.loads(Path(str(state.packet_path)).read_text(encoding="utf-8"))

    assert packet["packet_version"] == 1
    assert packet["lane"] == "lane-1"
    assert Path(packet["worktree_path"]).is_dir()
    baseline = packet["baseline"]
    assert baseline["samples"] == [0.0]
    assert baseline["minibatch_id"]
    assert all(Path(p).exists() for p in baseline["report_paths"])
    assert packet["journal_tail"][0]["content"] == "seed entry"
    invocation = packet["continue_invocation"]
    assert "lane continue lane-1" in invocation
    assert f"--run-id {run['run_id']}" in invocation
    # The invocation carries the absolute workspace explicitly (dec-780).
    assert f"--gepa-dir {git_repo}/.gepa" in invocation


def test_lane_lease_and_double_lease_rejected(git_repo: Path) -> None:
    run = _start_lane_run(git_repo, lanes=1)
    run_id = str(run["run_id"])
    gepa_dir = str(git_repo / ".gepa")

    leased = _run("--gepa-dir", gepa_dir, "lane", "lease", "lane-1", "--run-id", run_id)
    assert leased.exit_code == 0, leased.output
    state = _lane_state(git_repo, run_id, "lane-1")
    assert state.status == "leased"
    assert state.lease_epoch == 1
    assert state.lease_expires_at is not None

    again = _run("--gepa-dir", gepa_dir, "lane", "lease", "lane-1", "--run-id", run_id)
    assert again.exit_code == 1
    assert "already leased" in again.output


def test_lane_continue_evaluates_and_emits_verdict(git_repo: Path) -> None:
    """Reflector flow end-to-end: edit worktree, continue, verdict lands."""
    run = _start_lane_run(git_repo, lanes=1)
    run_id = str(run["run_id"])
    gepa_dir = str(git_repo / ".gepa")
    primary_head = _git(git_repo, "rev-parse", "HEAD")

    # The reflector improves the candidate inside the lane worktree only.
    worktree = git_repo / "worktrees" / run_id / "lane-1"
    (worktree / "score.txt").write_text("good\n", encoding="utf-8")

    # Invoke from inside the worktree using only the packet's invocation
    # fields (explicit absolute gepa-dir + run id + lane id).
    import os

    old_cwd = Path.cwd()
    os.chdir(worktree)
    try:
        result = _run(
            "--gepa-dir",
            gepa_dir,
            "lane",
            "continue",
            "lane-1",
            "--run-id",
            run_id,
            "--foreground",
        )
    finally:
        os.chdir(old_cwd)
    assert result.exit_code == 0, result.output
    assert "accepted" in result.output

    # Auto-commit landed on the lane branch; candidate is a clean commit.
    lane_head = _git(worktree, "rev-parse", "HEAD")
    assert lane_head != primary_head
    assert _git(worktree, "status", "--porcelain") == ""
    assert _git(git_repo, "rev-parse", "HEAD") == primary_head  # primary untouched

    state = _lane_state(git_repo, run_id, "lane-1")
    assert state.status == "awaiting_selection"
    assert state.verdict == "accepted"
    assert state.verdict_delta == pytest.approx(1.0)
    assert state.candidate_sha == lane_head
    assert state.eval_samples == (1.0,)
    assert state.eval_pid is None
    assert Path(str(state.comparison_path)).exists()

    # Verdict event + lane-scoped ledger row with collision-free artifacts.
    from pydantic_ai_gepa.cli.events import list_events

    verdict_events = [
        event.to_dict()
        for event in list_events(run_id, git_repo)
        if event.type == "verdict"
    ]
    assert len(verdict_events) == 1
    assert verdict_events[0]["lane"] == "lane-1"
    assert verdict_events[0]["payload"]["verdict"] == "accepted"

    from pydantic_ai_gepa.cli.runs import ParetoLog

    rows = ParetoLog(run_id, git_repo).iter_rows()
    lane_rows = [row for row in rows if row.lane == "lane-1"]
    assert len(lane_rows) == 1
    assert lane_rows[0].mean_score == pytest.approx(1.0)


def test_two_lanes_evaluate_in_distinct_worktrees(git_repo: Path) -> None:
    """Two lanes -> distinct clean-commit candidate ids; primary untouched."""
    run = _start_lane_run(git_repo, lanes=2)
    run_id = str(run["run_id"])
    gepa_dir = str(git_repo / ".gepa")
    primary_head = _git(git_repo, "rev-parse", "HEAD")

    edits = {"lane-1": "good\n", "lane-2": "good\n\n"}
    import os

    old_cwd = Path.cwd()
    try:
        for lane, content in edits.items():
            worktree = git_repo / "worktrees" / run_id / lane
            (worktree / "score.txt").write_text(content, encoding="utf-8")
            os.chdir(worktree)
            result = _run(
                "--gepa-dir",
                gepa_dir,
                "lane",
                "continue",
                lane,
                "--run-id",
                run_id,
                "--foreground",
            )
            assert result.exit_code == 0, result.output
    finally:
        os.chdir(old_cwd)

    state_1 = _lane_state(git_repo, run_id, "lane-1")
    state_2 = _lane_state(git_repo, run_id, "lane-2")
    assert state_1.candidate_sha != state_2.candidate_sha
    assert state_1.verdict == state_2.verdict == "accepted"
    assert _git(git_repo, "rev-parse", "HEAD") == primary_head
    assert _git(git_repo, "status", "--porcelain") == ""


def test_second_continue_on_evaluating_lane_rejected(git_repo: Path) -> None:
    run = _start_lane_run(git_repo, lanes=1)
    run_id = str(run["run_id"])
    gepa_dir = str(git_repo / ".gepa")

    # Simulate an in-flight eval: status evaluating with a live pid.
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        state = _lane_state(git_repo, run_id, "lane-1")
        state = LaneState(
            **{
                **state.to_dict(),
                "status": "evaluating",
                "eval_pid": sleeper.pid,
                "heartbeat_at": state.updated_at,
            }
        )
        state.save(git_repo, run_id)

        result = _run(
            "--gepa-dir",
            gepa_dir,
            "lane",
            "continue",
            "lane-1",
            "--run-id",
            run_id,
            "--foreground",
        )
        assert result.exit_code == 1
        assert "already evaluating" in result.output
        after = _lane_state(git_repo, run_id, "lane-1")
        assert after.status == "evaluating"
        assert after.eval_pid == sleeper.pid
    finally:
        sleeper.terminate()
        sleeper.wait()


def test_lane_reset_terminates_live_eval_and_preserves_worktree(
    git_repo: Path,
) -> None:
    run = _start_lane_run(git_repo, lanes=1)
    run_id = str(run["run_id"])
    gepa_dir = str(git_repo / ".gepa")

    worktree = git_repo / "worktrees" / run_id / "lane-1"
    (worktree / "notes.txt").write_text("uncommitted work\n", encoding="utf-8")

    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        state = _lane_state(git_repo, run_id, "lane-1")
        LaneState(
            **{
                **state.to_dict(),
                "status": "evaluating",
                "eval_pid": sleeper.pid,
                "heartbeat_at": state.updated_at,
            }
        ).save(git_repo, run_id)

        result = _run(
            "--gepa-dir", gepa_dir, "lane", "reset", "lane-1", "--run-id", run_id
        )
        assert result.exit_code == 0, result.output
        deadline = time.monotonic() + 5.0
        while sleeper.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert sleeper.poll() is not None  # terminated by reset
    finally:
        if sleeper.poll() is None:
            sleeper.kill()
            sleeper.wait()

    state = _lane_state(git_repo, run_id, "lane-1")
    assert state.status == "paused_for_reflection"
    assert state.eval_pid is None
    assert state.lease_epoch == 1
    # Uncommitted worktree content is never auto-deleted.
    assert (worktree / "notes.txt").read_text(encoding="utf-8") == "uncommitted work\n"


def test_run_status_renders_lane_board_and_reaps(git_repo: Path) -> None:
    """run status renders every lane and synthesizes lane_stalled exactly once."""
    run = _start_lane_run(git_repo, lanes=2)
    run_id = str(run["run_id"])

    # Fake a dead eval on lane-1: recorded pid that no longer exists.
    state = _lane_state(git_repo, run_id, "lane-1")
    LaneState(
        **{
            **state.to_dict(),
            "status": "evaluating",
            "eval_pid": 999_999_999,
            "heartbeat_at": state.updated_at,
        }
    ).save(git_repo, run_id)

    first = _run("run", "status", "--run-id", run_id)
    assert first.exit_code == 0, first.output
    payload = json.loads(first.output)
    lanes = {lane["lane"]: lane for lane in payload["lanes"]}
    assert set(lanes) == {"lane-1", "lane-2"}
    # The reaper transitions the dead-pid lane's STATE to stalled (dec-l95),
    # not just an event on the bus.
    assert lanes["lane-1"]["status"] == "stalled"
    assert lanes["lane-1"]["eval_pid"] is None
    assert lanes["lane-2"]["status"] == "paused_for_reflection"

    from pydantic_ai_gepa.cli.events import list_events

    stalled = [
        event for event in list_events(run_id, git_repo) if event.type == "lane_stalled"
    ]
    assert len(stalled) == 1
    assert stalled[0].lane == "lane-1"
    assert "dead" in stalled[0].payload["reason"]

    # A second status pass re-synthesizes nothing for the same lease epoch.
    second = _run("run", "status", "--run-id", run_id)
    assert second.exit_code == 0, second.output
    stalled = [
        event for event in list_events(run_id, git_repo) if event.type == "lane_stalled"
    ]
    assert len(stalled) == 1


def test_lane_verb_requires_explicit_absolute_workspace(git_repo: Path) -> None:
    run = _start_lane_run(git_repo, lanes=1)
    result = _run("lane", "reset", "lane-1", "--run-id", str(run["run_id"]))
    assert result.exit_code != 0
    assert "absolute workspace" in result.output or "GEPA_DIR" in result.output


# ---------- adversarial-review hardening (PR #29 review) ----------


def test_second_lane_run_rejected(git_repo: Path) -> None:
    """One active lane run per workspace (dec-jh6): a second `run start
    --lanes` while the first is active is refused before any fan-out."""
    _start_lane_run(git_repo, lanes=1)
    result = _run("run", "start", "--lanes", "1", "--size", "1")
    assert result.exit_code == 1
    assert "still active" in result.output
    # No second run dir was created with lanes.
    runs = [p for p in (git_repo / ".gepa" / "runs").iterdir() if p.is_dir()]
    assert len(runs) == 1


def test_lane_verbs_reject_done_runs(git_repo: Path) -> None:
    """Lane verbs refuse to run on completed runs (spec-1do: lanes are
    re-fanned at select or removed when the run completes)."""
    result = _run(
        "run",
        "start",
        "--lanes",
        "1",
        "--size",
        "1",
        "--acceptance-repetitions",
        "1",
        "--max-iterations",
        "1",
    )
    assert result.exit_code == 0, result.output
    run = _run_payload(result.output)
    assert run["status"] == "done"  # budget exhausted by the baseline eval
    run_id = str(run["run_id"])
    gepa_dir = str(git_repo / ".gepa")

    # run_done is emitted even though no select ever ran.
    from pydantic_ai_gepa.cli.events import list_events

    done_events = [e for e in list_events(run_id, git_repo) if e.type == "run_done"]
    assert len(done_events) == 1

    lease = _run("--gepa-dir", gepa_dir, "lane", "lease", "lane-1", "--run-id", run_id)
    assert lease.exit_code == 1
    assert "done" in lease.output

    # Once run_done is consumed, terminal lane state must not synthesize a
    # new selection_due event on the next poll.
    first = _run("--gepa-dir", gepa_dir, "next", "--run-id", run_id, "--json")
    assert first.exit_code == 0, first.output
    done_id = json.loads(first.output)["id"]
    ack = _run("--gepa-dir", gepa_dir, "ack", str(done_id), "--run-id", run_id)
    assert ack.exit_code == 0, ack.output
    second = _run("--gepa-dir", gepa_dir, "next", "--run-id", run_id, "--json")
    assert second.exit_code == 3, second.output


def test_dispatch_lease_is_consumed_by_continue(git_repo: Path) -> None:
    """The whole point of `lane lease` is the reflector's continue: a lane
    leased for dispatch ACCEPTS the reflector's continue (consuming the
    lease), while a second lease is rejected."""
    run = _start_lane_run(git_repo, lanes=1)
    run_id = str(run["run_id"])
    gepa_dir = str(git_repo / ".gepa")

    leased = _run("--gepa-dir", gepa_dir, "lane", "lease", "lane-1", "--run-id", run_id)
    assert leased.exit_code == 0, leased.output
    state = _lane_state(git_repo, run_id, "lane-1")
    assert state.lease_purpose == "dispatch"

    # A second dispatch lease is rejected while the first is unexpired.
    again = _run("--gepa-dir", gepa_dir, "lane", "lease", "lane-1", "--run-id", run_id)
    assert again.exit_code == 1
    assert "already leased" in again.output

    # The reflector's continue consumes the dispatch lease and evaluates.
    worktree = git_repo / "worktrees" / run_id / "lane-1"
    (worktree / "score.txt").write_text("good\n", encoding="utf-8")
    import os

    old_cwd = Path.cwd()
    os.chdir(worktree)
    try:
        result = _run(
            "--gepa-dir",
            gepa_dir,
            "lane",
            "continue",
            "lane-1",
            "--run-id",
            run_id,
            "--foreground",
        )
    finally:
        os.chdir(old_cwd)
    assert result.exit_code == 0, result.output
    state = _lane_state(git_repo, run_id, "lane-1")
    assert state.status == "awaiting_selection"
    assert state.lease_purpose is None  # consumed


def test_handoff_lease_rejects_second_continue(git_repo: Path) -> None:
    """A handoff lease (detached eval spawned) rejects a second parent-mode
    continue — the double-spawn race the lease exists to prevent."""
    run = _start_lane_run(git_repo, lanes=1)
    run_id = str(run["run_id"])
    gepa_dir = str(git_repo / ".gepa")

    state = _lane_state(git_repo, run_id, "lane-1")
    LaneState(
        **{
            **state.to_dict(),
            "status": "leased",
            "lease_epoch": 1,
            "lease_purpose": "handoff",
            "lease_expires_at": "2999-01-01T00:00:00+00:00",
        }
    ).save(git_repo, run_id)

    result = _run(
        "--gepa-dir", gepa_dir, "lane", "continue", "lane-1", "--run-id", run_id
    )
    assert result.exit_code == 1
    assert "handoff in flight" in result.output


def test_stale_detached_child_cannot_claim_replaced_handoff(git_repo: Path) -> None:
    """A delayed child is fenced by its original handoff epoch.

    This models the child starting after its lease expired, the reaper stalled
    the lane, and reset/re-dispatch created a replacement lease. It must not
    turn the replacement lane into evaluating or emit an obsolete verdict.
    """
    run = _start_lane_run(git_repo, lanes=1)
    run_id = str(run["run_id"])
    gepa_dir = str(git_repo / ".gepa")
    state = _lane_state(git_repo, run_id, "lane-1")
    replacement = LaneState(
        **{
            **state.to_dict(),
            "status": "leased",
            "lease_epoch": 5,
            "lease_purpose": "dispatch",
            "lease_expires_at": "2999-01-01T00:00:00+00:00",
        }
    )
    replacement.save(git_repo, run_id)

    stale = _run(
        "--gepa-dir",
        gepa_dir,
        "lane",
        "continue",
        "lane-1",
        "--run-id",
        run_id,
        "--foreground",
        "--handoff-lease-epoch",
        "3",
    )
    assert stale.exit_code == 1
    assert "no longer current" in stale.output
    current = _lane_state(git_repo, run_id, "lane-1")
    assert current.status == "leased"
    assert current.lease_epoch == 5
    assert current.lease_purpose == "dispatch"
    assert current.lease_expires_at == "2999-01-01T00:00:00+00:00"

    from pydantic_ai_gepa.cli.events import list_events

    assert not [
        event for event in list_events(run_id, git_repo) if event.type == "verdict"
    ]


def test_lane_reset_refuses_awaiting_selection(git_repo: Path) -> None:
    """A resolved lane's verdict is never destroyed unrecorded."""
    run = _start_lane_run(git_repo, lanes=1)
    run_id = str(run["run_id"])
    gepa_dir = str(git_repo / ".gepa")
    worktree = git_repo / "worktrees" / run_id / "lane-1"
    (worktree / "score.txt").write_text("good\n", encoding="utf-8")

    import os

    old_cwd = Path.cwd()
    os.chdir(worktree)
    try:
        result = _run(
            "--gepa-dir",
            gepa_dir,
            "lane",
            "continue",
            "lane-1",
            "--run-id",
            run_id,
            "--foreground",
        )
    finally:
        os.chdir(old_cwd)
    assert result.exit_code == 0, result.output
    assert _lane_state(git_repo, run_id, "lane-1").status == "awaiting_selection"

    reset = _run("--gepa-dir", gepa_dir, "lane", "reset", "lane-1", "--run-id", run_id)
    assert reset.exit_code == 1
    assert "awaiting_selection" in reset.output
    # Verdict intact.
    state = _lane_state(git_repo, run_id, "lane-1")
    assert state.verdict == "accepted"
    assert state.status == "awaiting_selection"


def test_foreground_continue_evaluates_worktree_not_primary(git_repo: Path) -> None:
    """Foreground continue invoked from the PRIMARY checkout still evaluates
    the lane worktree's tree (the loop chdirs to the candidate root)."""
    run = _start_lane_run(git_repo, lanes=1)
    run_id = str(run["run_id"])
    gepa_dir = str(git_repo / ".gepa")
    worktree = git_repo / "worktrees" / run_id / "lane-1"
    (worktree / "score.txt").write_text("good\n", encoding="utf-8")
    # Primary still scores "bad".
    assert (git_repo / "score.txt").read_text(encoding="utf-8") == "bad\n"

    result = _run(
        "--gepa-dir",
        gepa_dir,
        "lane",
        "continue",
        "lane-1",
        "--run-id",
        run_id,
        "--foreground",
    )
    assert result.exit_code == 0, result.output
    assert "accepted" in result.output
    state = _lane_state(git_repo, run_id, "lane-1")
    assert state.verdict == "accepted"


def test_stale_heartbeat_with_live_pid_is_not_stalled(git_repo: Path) -> None:
    """spec-1do: eval death = stale heartbeat PLUS dead pid. A slow but alive
    eval is never reaped (the documented response kills the pid)."""
    run = _start_lane_run(git_repo, lanes=1)
    run_id = str(run["run_id"])

    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        state = _lane_state(git_repo, run_id, "lane-1")
        LaneState(
            **{
                **state.to_dict(),
                "status": "evaluating",
                "eval_pid": sleeper.pid,
                "heartbeat_at": "2020-01-01T00:00:00+00:00",  # ancient
            }
        ).save(git_repo, run_id)

        result = _run("run", "status", "--run-id", run_id)
        assert result.exit_code == 0, result.output
        from pydantic_ai_gepa.cli.events import list_events

        stalled = [e for e in list_events(run_id, git_repo) if e.type == "lane_stalled"]
        assert stalled == []
        lanes = {lane["lane"]: lane for lane in json.loads(result.output)["lanes"]}
        assert lanes["lane-1"]["status"] == "evaluating"
    finally:
        sleeper.terminate()
        sleeper.wait()


def test_corrupt_lane_state_gives_actionable_error(git_repo: Path) -> None:
    started = _start_lane_run(git_repo, lanes=1)
    run_id = str(started["run_id"])
    path = lane_state_path(git_repo, run_id, "lane-1")
    path.write_text("{torn", encoding="utf-8")
    gepa_dir = str(git_repo / ".gepa")
    result = _run("--gepa-dir", gepa_dir, "lane", "lease", "lane-1", "--run-id", run_id)
    assert result.exit_code != 0
    assert "corrupt" in result.output or "lane reset" in result.output

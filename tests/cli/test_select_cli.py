"""End-to-end coverage for `gepa run select` (spec-er3, task-vso).

Fixtures mirror test_lanes_cli.py: a git-native workspace whose evaluate
callable reads one output file per case, so each lane's score (and diff
footprint) is controlled by which ``out_<case>.txt`` files it writes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest
from click.testing import Result
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.lanes import LaneState, load_lane_state
from pydantic_ai_gepa.cli.run import RunState
from pydantic_ai_gepa.cli.runs import ParetoLog, utc_now_iso

EVALUATE_MODULE_SOURCE = textwrap.dedent("""
    from pathlib import Path

    async def evaluate(case):
        path = Path(f"out_{case.name}.txt")
        return path.read_text(encoding="utf-8").strip() if path.exists() else ""
""").lstrip()

# Three cases, expected outputs a/b/c. The seed tree only answers case-1, so
# the frozen baseline mean is 1/3 and every case file a lane adds moves its
# candidate mean by a known amount.
DATASET = [
    {"name": "case-1", "inputs": "x", "expected_output": "a"},
    {"name": "case-2", "inputs": "x", "expected_output": "b"},
    {"name": "case-3", "inputs": "x", "expected_output": "c"},
]
BASELINE_MEAN = 1.0 / 3.0


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
    (tmp_path / "out_case-1.txt").write_text("a\n", encoding="utf-8")
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
        "".join(json.dumps(row) + "\n" for row in DATASET),
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


# ----------------------------- helpers ----------------------------------


def _gepa_dir(repo: Path) -> str:
    return str(repo / ".gepa")


def _start_lane_run(repo: Path, lanes: int = 3, *extra: str) -> dict[str, object]:
    result = _run(
        "run",
        "start",
        "--lanes",
        str(lanes),
        "--size",
        "3",
        "--acceptance-repetitions",
        "1",
        *extra,
    )
    assert result.exit_code == 0, result.output
    return _run_payload(result.output)


def _drive_lane(repo: Path, run_id: str, lane: str, files: dict[str, str]) -> LaneState:
    """Reflector flow: edit the lane worktree, foreground `lane continue`."""
    worktree = repo / "worktrees" / run_id / lane
    for name, content in files.items():
        (worktree / name).write_text(content, encoding="utf-8")
    old_cwd = Path.cwd()
    os.chdir(worktree)
    try:
        result = _run(
            "--gepa-dir",
            _gepa_dir(repo),
            "lane",
            "continue",
            lane,
            "--run-id",
            run_id,
            "--foreground",
        )
    finally:
        os.chdir(old_cwd)
    assert result.exit_code == 0, result.output
    return load_lane_state(repo, run_id, lane)


def _select(repo: Path, run_id: str) -> Result:
    return _run("--gepa-dir", _gepa_dir(repo), "run", "select", "--run-id", run_id)


def _state(repo: Path, run_id: str) -> RunState:
    raw = json.loads(
        (repo / ".gepa" / "runs" / run_id / "state.json").read_text(encoding="utf-8")
    )
    return RunState.from_dict(raw)


def _write_state(repo: Path, state: RunState) -> None:
    (repo / ".gepa" / "runs" / state.run_id / "state.json").write_text(
        json.dumps(state.to_dict(), indent=2), encoding="utf-8"
    )


def _journal_outcomes(
    repo: Path, run_id: str, *, lane: str | None = None, outcome: str | None = None
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in (
        (repo / ".gepa" / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if row.get("kind") != "lane_outcome" or row.get("run_id") != run_id:
            continue
        if lane is not None and row.get("lane") != lane:
            continue
        if outcome is not None and row.get("outcome") != outcome:
            continue
        rows.append(row)
    return rows


def _events(
    repo: Path, run_id: str, type_: str | None = None
) -> list[dict[str, object]]:
    events_dir = repo / ".gepa" / "runs" / run_id / "events"
    events = [
        json.loads(p.read_text()) for p in sorted(events_dir.iterdir()) if p.is_file()
    ]
    if type_ is not None:
        events = [event for event in events if event["type"] == type_]
    return events


def _lane_branches(repo: Path) -> list[str]:
    out = _git(repo, "branch", "--format=%(refname:short)", "--list", "gepa/lane/*")
    return sorted(line for line in out.splitlines() if line)


def _run_id(run: dict[str, object]) -> str:
    return str(run["run_id"])


# ----------------------------- tests ------------------------------------


def test_select_promotes_winner_journals_losers_and_refans(git_repo: Path) -> None:
    """spec-er3: accepted +0.33 / accepted +0.67 / rejected -> best promoted;
    losers journaled with diff summary, verdict, delta; branches deleted;
    lanes re-branched off the new best; baseline re-measured once."""
    run = _start_lane_run(git_repo, lanes=3)
    run_id = _run_id(run)
    lane_1 = _drive_lane(git_repo, run_id, "lane-1", {"out_case-2.txt": "b\n"})
    lane_2 = _drive_lane(
        git_repo, run_id, "lane-2", {"out_case-2.txt": "b\n", "out_case-3.txt": "c\n"}
    )
    lane_3 = _drive_lane(git_repo, run_id, "lane-3", {"out_case-1.txt": "zzz\n"})
    assert lane_1.verdict == "accepted"
    assert lane_2.verdict == "accepted"
    assert lane_3.verdict == "rejected"
    rows_before = ParetoLog(run_id, git_repo).count_rows()

    result = _select(git_repo, run_id)
    assert result.exit_code == 0, result.output

    # Winner promotion: run state + primary checkout both advanced to lane-2.
    state = _state(git_repo, run_id)
    assert state.select_phase is None
    assert state.status == "running"
    assert state.best_commit_sha == lane_2.candidate_sha
    assert state.best_mean_score == pytest.approx(1.0)
    assert _git(git_repo, "rev-parse", "HEAD") == lane_2.candidate_sha
    assert (git_repo / "out_case-3.txt").read_text(encoding="utf-8") == "c\n"

    # Journal write-back: two loser entries + the promoted entry, each with
    # diff summary, verdict, delta, and confidence.
    losers = _journal_outcomes(git_repo, run_id, outcome="loser")
    assert {row["lane"] for row in losers} == {"lane-1", "lane-3"}
    by_lane = {str(row["lane"]): row for row in losers}
    assert by_lane["lane-1"]["verdict"] == "accepted"
    assert by_lane["lane-1"]["delta"] == pytest.approx(1.0 / 3.0)
    assert by_lane["lane-3"]["verdict"] == "rejected"
    assert by_lane["lane-3"]["delta"] == pytest.approx(-1.0 / 3.0)
    for row in losers:
        assert "out_case" in str(row["diff_summary"])
        assert row["confidence"] is not None
    promoted = _journal_outcomes(git_repo, run_id, outcome="promoted")
    assert [row["lane"] for row in promoted] == ["lane-2"]

    # Loser branches deleted; every lane re-branched off the new best.
    assert _lane_branches(git_repo) == [
        f"gepa/lane/{run_id}/lane-1/2",
        f"gepa/lane/{run_id}/lane-2/2",
        f"gepa/lane/{run_id}/lane-3/2",
    ]
    for lane in ("lane-1", "lane-2", "lane-3"):
        lane_state = load_lane_state(git_repo, run_id, lane)
        assert lane_state.status == "paused_for_reflection"
        assert lane_state.iteration == 2
        assert lane_state.branch == f"gepa/lane/{run_id}/{lane}/2"
        # Fresh lease epoch after re-fan (the detached handoff bumps the
        # epoch once; direct foreground drives start unleased).
        assert lane_state.lease_epoch >= 1
        assert lane_state.candidate_sha is None
        assert lane_state.verdict is None
        worktree = git_repo / "worktrees" / run_id / lane
        assert _git(worktree, "rev-parse", "HEAD") == lane_2.candidate_sha
        assert Path(str(lane_state.packet_path)).exists()
        packet = json.loads(Path(str(lane_state.packet_path)).read_text())
        assert packet["baseline"]["samples"] == [pytest.approx(1.0)]

    # Fresh frozen baseline at the new best, paid once for the iteration.
    assert state.reflection_baseline_commit_sha == lane_2.candidate_sha
    assert list(state.reflection_baseline_samples) == [pytest.approx(1.0)]
    rows_after = ParetoLog(run_id, git_repo).count_rows()
    assert rows_after - rows_before == 1

    # lane_ready re-emitted per lane; overlapping accepted diffs -> no merge
    # opportunity.
    assert len(_events(git_repo, run_id, "lane_ready")) == 6
    assert _events(git_repo, run_id, "merge_opportunity") == []


def test_select_consumes_memoized_verdicts_without_comparing(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """spec-er3: verdicts are computed once per lane eval and memoized; select
    consumes them from lane state and never re-derives them."""
    import pydantic_ai_gepa.acceptance as acceptance_mod
    import pydantic_ai_gepa.cli.lanes as lanes_mod
    import pydantic_ai_gepa.cli.run as run_mod
    import pydantic_ai_gepa.cli.select as select_mod

    run = _start_lane_run(git_repo, lanes=2)
    run_id = _run_id(run)
    _drive_lane(git_repo, run_id, "lane-1", {"out_case-2.txt": "b\n"})
    _drive_lane(git_repo, run_id, "lane-2", {"out_case-3.txt": "c\n"})

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("compare_candidate_samples called during select")

    monkeypatch.setattr(acceptance_mod, "compare_candidate_samples", _boom)
    monkeypatch.setattr(lanes_mod, "compare_candidate_samples", _boom)
    monkeypatch.setattr(run_mod, "compare_candidate_samples", _boom)
    assert not hasattr(select_mod, "compare_candidate_samples")

    result = _select(git_repo, run_id)
    assert result.exit_code == 0, result.output


def test_straggler_terminated_journaled_and_refanned(git_repo: Path) -> None:
    """spec-er3 + dec-4tw: a lane still evaluating when select runs (after the
    straggler timeout) is terminated, its partial result and diff journaled,
    and the lane reset in the re-fan — never compared, never promoted."""
    run = _start_lane_run(git_repo, 2, "--straggler-timeout-secs", "0")
    run_id = _run_id(run)
    winner = _drive_lane(git_repo, run_id, "lane-1", {"out_case-2.txt": "b\n"})

    # Simulate an in-flight eval on lane-2: committed partial work, an
    # uncommitted scratch file, partial samples, live pid, fresh heartbeat.
    worktree = git_repo / "worktrees" / run_id / "lane-2"
    (worktree / "out_case-2.txt").write_text("b\n", encoding="utf-8")
    (worktree / "notes.txt").write_text("scratch\n", encoding="utf-8")
    _git(worktree, "add", "out_case-2.txt")
    _git(
        worktree,
        "-c",
        "user.name=gepa-lane",
        "-c",
        "user.email=gepa-lane@localhost",
        "commit",
        "-m",
        "partial lane work",
    )
    partial_sha = _git(worktree, "rev-parse", "HEAD")

    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        lane_state = load_lane_state(git_repo, run_id, "lane-2")
        LaneState(
            **{
                **lane_state.to_dict(),
                "status": "evaluating",
                "candidate_sha": partial_sha,
                "eval_samples": (2.0 / 3.0,),
                "eval_pid": sleeper.pid,
                "heartbeat_at": utc_now_iso(),
            }
        ).save(git_repo, run_id)

        result = _select(git_repo, run_id)
        assert result.exit_code == 0, result.output
        assert sleeper.poll() is not None  # terminated by select
    finally:
        if sleeper.poll() is None:
            sleeper.kill()
            sleeper.wait()

    # Straggler journaled with partial samples + diff summary, never promoted.
    entries = _journal_outcomes(git_repo, run_id, lane="lane-2", outcome="straggler")
    assert len(entries) == 1
    entry = entries[0]
    assert entry["eval_samples"] == [pytest.approx(2.0 / 3.0)]
    assert "out_case-2.txt" in str(entry["diff_summary"])
    assert "notes.txt" in entry["untracked_paths"]
    assert entry["verdict"] is None

    state = _state(git_repo, run_id)
    assert state.best_commit_sha == winner.candidate_sha

    # Lane reset in the re-fan: paused on the fresh branch, partial work gone.
    lane_state = load_lane_state(git_repo, run_id, "lane-2")
    assert lane_state.status == "paused_for_reflection"
    assert lane_state.iteration == 2
    assert lane_state.candidate_sha is None
    assert lane_state.eval_pid is None
    assert not (worktree / "notes.txt").exists()
    assert _git(worktree, "rev-parse", "HEAD") == winner.candidate_sha


def test_cross_branch_point_lane_invalidated(git_repo: Path) -> None:
    """spec-er3: a lane whose candidate does not descend from the frozen
    baseline commit is invalidated to stalled — never silently compared,
    never promoted, even with the best delta."""
    run = _start_lane_run(git_repo, 2, "--straggler-timeout-secs", "0")
    run_id = _run_id(run)
    winner = _drive_lane(git_repo, run_id, "lane-1", {"out_case-2.txt": "b\n"})

    # lane-2's "candidate" is an orphan root commit (different branch point).
    worktree = git_repo / "worktrees" / run_id / "lane-2"
    _git(worktree, "checkout", "--orphan", "lane-2-orphan")
    _git(
        worktree,
        "-c",
        "user.name=gepa-lane",
        "-c",
        "user.email=gepa-lane@localhost",
        "commit",
        "-m",
        "orphan root",
    )
    orphan_sha = _git(worktree, "rev-parse", "HEAD")
    lane_state = load_lane_state(git_repo, run_id, "lane-2")
    LaneState(
        **{
            **lane_state.to_dict(),
            "status": "awaiting_selection",
            "candidate_sha": orphan_sha,
            "eval_samples": (1.0,),
            "verdict": "accepted",
            "verdict_delta": 99.0,
        }
    ).save(git_repo, run_id)

    result = _select(git_repo, run_id)
    assert result.exit_code == 0, result.output

    entries = _journal_outcomes(git_repo, run_id, lane="lane-2", outcome="invalidated")
    assert len(entries) == 1
    assert "frozen baseline" in str(entries[0]["reason"])

    # lane-1 wins despite the smaller delta; the orphan sha is nowhere.
    state = _state(git_repo, run_id)
    assert state.best_commit_sha == winner.candidate_sha
    lane_state = load_lane_state(git_repo, run_id, "lane-2")
    assert lane_state.status == "paused_for_reflection"
    assert lane_state.iteration == 2


def test_concurrent_select_rejected(git_repo: Path) -> None:
    """spec-er3: a second select invocation while one is in flight is rejected."""
    run = _start_lane_run(git_repo, lanes=1)
    run_id = _run_id(run)
    lane_1 = _drive_lane(git_repo, run_id, "lane-1", {"out_case-2.txt": "b\n"})
    primary_head = _git(git_repo, "rev-parse", "HEAD")

    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        state = _state(git_repo, run_id)
        state = RunState(
            **{
                **state.to_dict(),
                "select_phase": "journal",
                "select_context": {"pid": sleeper.pid, "started_ms": 0},
            }
        )
        _write_state(git_repo, state)

        result = _select(git_repo, run_id)
        assert result.exit_code == 1
        assert "already in flight" in result.output
    finally:
        sleeper.terminate()
        sleeper.wait()

    # Nothing moved.
    state = _state(git_repo, run_id)
    assert state.select_phase == "journal"
    lane_state = load_lane_state(git_repo, run_id, "lane-1")
    assert lane_state.status == "awaiting_selection"
    assert _git(git_repo, "rev-parse", "HEAD") == primary_head
    assert state.best_commit_sha != lane_1.candidate_sha


def test_select_killed_after_promotion_resumes_exactly_once(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """spec-er3: select killed after promotion but before re-fan resumes from
    the recorded phase and completes the remaining phases exactly once."""
    import pydantic_ai_gepa.cli.select as select_mod

    run = _start_lane_run(git_repo, lanes=3)
    run_id = _run_id(run)
    _drive_lane(git_repo, run_id, "lane-1", {"out_case-2.txt": "b\n"})
    lane_2 = _drive_lane(
        git_repo, run_id, "lane-2", {"out_case-2.txt": "b\n", "out_case-3.txt": "c\n"}
    )
    _drive_lane(git_repo, run_id, "lane-3", {"out_case-1.txt": "zzz\n"})

    original = select_mod._journal_lane_outcome
    loser_calls = {"count": 0}

    def _kill_on_second_loser(workspace_root: Path, entry: dict[str, object]) -> None:
        if entry.get("outcome") == "loser":
            loser_calls["count"] += 1
            if loser_calls["count"] == 2:
                raise RuntimeError("simulated kill mid-journal phase")
        original(workspace_root, entry)

    monkeypatch.setattr(select_mod, "_journal_lane_outcome", _kill_on_second_loser)
    first = _select(git_repo, run_id)
    assert first.exit_code == 1
    assert isinstance(first.exception, RuntimeError)

    # Promotion already happened; the kill landed mid-journal.
    state = _state(git_repo, run_id)
    assert state.select_phase == "journal"
    assert state.best_commit_sha == lane_2.candidate_sha
    assert len(_journal_outcomes(git_repo, run_id, outcome="loser")) == 1

    monkeypatch.setattr(select_mod, "_journal_lane_outcome", original)
    second = _select(git_repo, run_id)
    assert second.exit_code == 0, second.output
    assert "Resuming select" in second.output

    # Exactly-once completion: one loser entry per lane, one fresh lane_ready
    # per lane, one set of new branches.
    losers = _journal_outcomes(git_repo, run_id, outcome="loser")
    assert sorted(str(row["lane"]) for row in losers) == ["lane-1", "lane-3"]
    assert len(_events(git_repo, run_id, "lane_ready")) == 6
    assert _lane_branches(git_repo) == [
        f"gepa/lane/{run_id}/lane-1/2",
        f"gepa/lane/{run_id}/lane-2/2",
        f"gepa/lane/{run_id}/lane-3/2",
    ]
    state = _state(git_repo, run_id)
    assert state.select_phase is None
    assert state.select_context is None
    assert state.best_commit_sha == lane_2.candidate_sha
    assert _git(git_repo, "rev-parse", "HEAD") == lane_2.candidate_sha


def test_merge_opportunity_for_disjoint_accepted_lanes(git_repo: Path) -> None:
    """spec-er3: two accepted lanes whose diffs touch disjoint file sets emit
    merge_opportunity; select never auto-merges."""
    run = _start_lane_run(git_repo, lanes=2)
    run_id = _run_id(run)
    lane_1 = _drive_lane(git_repo, run_id, "lane-1", {"out_case-2.txt": "b\n"})
    lane_2 = _drive_lane(git_repo, run_id, "lane-2", {"out_case-3.txt": "c\n"})
    assert lane_1.verdict_delta == pytest.approx(lane_2.verdict_delta or 0.0)

    result = _select(git_repo, run_id)
    assert result.exit_code == 0, result.output

    opportunities = _events(git_repo, run_id, "merge_opportunity")
    assert len(opportunities) == 1
    payload = opportunities[0]["payload"]
    assert payload["lane_a"] == "lane-1"
    assert payload["lane_b"] == "lane-2"
    assert payload["branch_a"] == f"gepa/lane/{run_id}/lane-1/1"
    assert payload["branch_b"] == f"gepa/lane/{run_id}/lane-2/1"
    stat_path = Path(str(payload["diff_stat_path"]))
    assert stat_path.exists()
    stat = stat_path.read_text(encoding="utf-8")
    assert "out_case-2.txt" in stat
    assert "out_case-3.txt" in stat

    # Tie broke to the first lane id; no auto-merge: only lane-1's file is in
    # the promoted tree.
    state = _state(git_repo, run_id)
    assert state.best_commit_sha == lane_1.candidate_sha
    assert _git(git_repo, "rev-parse", "HEAD") == lane_1.candidate_sha
    assert (git_repo / "out_case-2.txt").exists()
    assert not (git_repo / "out_case-3.txt").exists()


def test_budget_exhausted_marks_done_with_overshoot(git_repo: Path) -> None:
    """spec-er3 + dec-msy: the budget stop is enforced at select; overshoot is
    recorded in the final report and run_done is emitted."""
    run = _start_lane_run(git_repo, 2, "--max-iterations", "2")
    run_id = _run_id(run)
    assert ParetoLog(run_id, git_repo).count_rows() == 1
    lane_1 = _drive_lane(git_repo, run_id, "lane-1", {"out_case-2.txt": "b\n"})
    _drive_lane(git_repo, run_id, "lane-2", {"out_case-3.txt": "c\n"})
    assert ParetoLog(run_id, git_repo).count_rows() == 3  # 1 over budget

    result = _select(git_repo, run_id)
    assert result.exit_code == 0, result.output

    state = _state(git_repo, run_id)
    assert state.status == "done"
    assert state.iterations == 3
    assert state.select_phase is None
    assert state.best_commit_sha == lane_1.candidate_sha

    done_events = _events(git_repo, run_id, "run_done")
    assert len(done_events) == 1
    report_path = Path(str(done_events[0]["payload"]["final_report_path"]))
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "budget_overshoot: 1" in report

    # Lanes are removed when the run completes: no re-fan happened.
    assert not (git_repo / "worktrees" / run_id / "lane-1").exists()
    assert not (git_repo / "worktrees" / run_id / "lane-2").exists()
    # Both lanes were accepted with disjoint diffs, so their branches form a
    # merge pair and SURVIVE finalize — the merge_opportunity event names
    # branches for the orchestrator to merge, so deleting them in the same
    # select would make the event unactionable.
    assert _lane_branches(git_repo) == [
        f"gepa/lane/{run_id}/lane-1/1",
        f"gepa/lane/{run_id}/lane-2/1",
    ]
    assert len(_events(git_repo, run_id, "lane_ready")) == 2  # fan-out only

    # A done run has nothing to select.
    again = _select(git_repo, run_id)
    assert again.exit_code == 1
    assert "done" in again.output


def test_select_rejected_when_not_due(git_repo: Path) -> None:
    """Select refuses to run while lanes are in flight and the straggler
    timeout has not elapsed."""
    run = _start_lane_run(git_repo, lanes=2)
    run_id = _run_id(run)
    _drive_lane(git_repo, run_id, "lane-1", {"out_case-2.txt": "b\n"})
    primary_head = _git(git_repo, "rev-parse", "HEAD")

    result = _select(git_repo, run_id)
    assert result.exit_code == 1
    assert "not due" in result.output

    state = _state(git_repo, run_id)
    assert state.select_phase is None
    lane_state = load_lane_state(git_repo, run_id, "lane-2")
    assert lane_state.status == "paused_for_reflection"
    assert _git(git_repo, "rev-parse", "HEAD") == primary_head
    # Only the fan-out lane_ready events exist; select emitted nothing.
    assert len(_events(git_repo, run_id, "lane_ready")) == 2


def test_select_dirty_primary_promotes_in_run_state_only(git_repo: Path) -> None:
    """A dirty primary checkout is never reset: promotion lands in run state,
    a warning is printed, and the winner branch is kept (sole ref)."""
    run = _start_lane_run(git_repo, lanes=1)
    run_id = _run_id(run)
    lane_1 = _drive_lane(git_repo, run_id, "lane-1", {"out_case-2.txt": "b\n"})
    baseline_head = _git(git_repo, "rev-parse", "HEAD")

    (git_repo / "untracked-note.txt").write_text("user work\n", encoding="utf-8")
    result = _select(git_repo, run_id)
    assert result.exit_code == 0, result.output
    assert "primary checkout is dirty" in result.output

    state = _state(git_repo, run_id)
    assert state.best_commit_sha == lane_1.candidate_sha
    # Primary untouched; user work preserved.
    assert _git(git_repo, "rev-parse", "HEAD") == baseline_head
    assert not (git_repo / "out_case-2.txt").exists()
    assert (git_repo / "untracked-note.txt").read_text() == "user work\n"
    # Winner branch kept so the promoted commit stays reachable; the fresh
    # re-fan branch exists alongside it.
    assert _lane_branches(git_repo) == [
        f"gepa/lane/{run_id}/lane-1/1",
        f"gepa/lane/{run_id}/lane-1/2",
    ]
    # The shared baseline was still re-measured at the new best (via the
    # re-fanned lane worktree, which carries the same commit).
    assert list(state.reflection_baseline_samples) == [pytest.approx(2.0 / 3.0)]
    assert state.reflection_baseline_commit_sha == lane_1.candidate_sha


def test_no_accepted_lane_means_no_promotion(git_repo: Path) -> None:
    """spec-er3: with no accepted lane there is no promotion; all lanes are
    journaled as losers and re-fanned onto the unchanged best."""
    run = _start_lane_run(git_repo, lanes=2)
    run_id = _run_id(run)
    baseline_head = _git(git_repo, "rev-parse", "HEAD")
    _drive_lane(git_repo, run_id, "lane-1", {"out_case-1.txt": "zzz\n"})
    _drive_lane(git_repo, run_id, "lane-2", {})  # unchanged -> equivalent

    result = _select(git_repo, run_id)
    assert result.exit_code == 0, result.output
    assert "No lane was accepted" in result.output

    state = _state(git_repo, run_id)
    assert state.best_commit_sha == baseline_head
    assert _git(git_repo, "rev-parse", "HEAD") == baseline_head
    losers = _journal_outcomes(git_repo, run_id, outcome="loser")
    assert {str(row["lane"]) for row in losers} == {"lane-1", "lane-2"}
    assert {str(row["verdict"]) for row in losers} == {"rejected", "equivalent"}
    # Lanes still re-fan (onto the same best) with fresh branches + baseline.
    assert _lane_branches(git_repo) == [
        f"gepa/lane/{run_id}/lane-1/2",
        f"gepa/lane/{run_id}/lane-2/2",
    ]
    assert list(state.reflection_baseline_samples) == [pytest.approx(BASELINE_MEAN)]
    assert len(_events(git_repo, run_id, "lane_ready")) == 4


def test_select_rejects_non_lane_run(git_repo: Path) -> None:
    result = _run("run", "start", "--size", "3", "--acceptance-repetitions", "1")
    assert result.exit_code == 0, result.output
    run_id = _run_id(_run_payload(result.output))
    select = _select(git_repo, run_id)
    assert select.exit_code == 1
    assert "not a lane run" in select.output


def test_pre_select_state_loads_with_select_defaults(git_repo: Path) -> None:
    """Run state files written before select existed load with defaults."""
    run = _start_lane_run(git_repo, lanes=1)
    run_id = _run_id(run)
    state_path = git_repo / ".gepa" / "runs" / run_id / "state.json"
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    raw.pop("select_phase", None)
    raw.pop("select_context", None)
    state = RunState.from_dict(raw)
    assert state.select_phase is None
    assert state.select_context is None


# ---------- adversarial-review hardening (PR #30 review) ----------


def test_select_lock_serializes_concurrent_selects(git_repo: Path) -> None:
    """spec-er3: select never runs concurrently with itself — two fresh
    invocations serialize on the flock, so the second resumes/finishes
    idempotently rather than double-executing phases."""
    import threading

    run = _start_lane_run(git_repo, 2)
    run_id = _run_id(run)
    _drive_lane(git_repo, run_id, "lane-1", {"out_case-2.txt": "b\n"})
    _drive_lane(git_repo, run_id, "lane-2", {"out_case-3.txt": "c\n"})

    results: list = []

    def do_select() -> None:
        results.append(_select(git_repo, run_id))

    threads = [threading.Thread(target=do_select) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # The lock serializes the two selects: the first performs the phases; the
    # second resumes after it and cleanly reports nothing-to-select — with
    # exactly one promotion and one loser journal (no doubled side effects
    # from the fresh-fresh race the pid-only guard used to lose).
    codes = sorted(result.exit_code for result in results)
    assert codes == [0, 1]  # one selects; the other finds nothing left
    promoted = _journal_outcomes(git_repo, run_id, outcome="promoted")
    assert len(promoted) == 1
    losers = _journal_outcomes(git_repo, run_id, outcome="loser")
    assert len(losers) == 1
    state = _state(git_repo, run_id)
    assert state.select_phase is None
    assert len(_events(git_repo, run_id, "lane_ready")) == 4  # fan-out + re-fan


def test_budget_low_emitted_near_budget_floor(git_repo: Path) -> None:
    """dec-d0d: select emits budget_low when remaining evals fall below
    lanes x acceptance max-repetitions."""
    run = _start_lane_run(git_repo, 2, "--max-iterations", "4")
    run_id = _run_id(run)
    _drive_lane(git_repo, run_id, "lane-1", {"out_case-2.txt": "b\n"})
    _drive_lane(git_repo, run_id, "lane-2", {"out_case-3.txt": "c\n"})
    # rows: 1 baseline + 2 lane evals = 3; remaining = 1 < lanes(2) x reps(1) = 2
    result = _select(git_repo, run_id)
    assert result.exit_code == 0, result.output

    low_events = _events(git_repo, run_id, "budget_low")
    assert len(low_events) == 1
    assert low_events[0]["payload"]["remaining_evals"] == 1

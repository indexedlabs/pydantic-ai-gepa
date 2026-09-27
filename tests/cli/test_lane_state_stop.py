"""Broken reflector verdicts terminate with the private incumbent, without leaks."""

import json
from dataclasses import replace

import pytest

from pydantic_ai_gepa.cli import drive, events, harness_record, run, select
from pydantic_ai_gepa.cli.lanes import LaneStateProblem, load_selection_lane_states
from pydantic_ai_gepa.cli.runs import ParetoLog
from pydantic_ai_gepa.cli.validation import harness_environment
from tests.cli import test_drive, test_lane_accounting, test_select_cli

lane_repo = test_select_cli.git_repo
MARKER = "PLANTED-7f3a"
CASES = [
    ("duplicate", "lane state names a duplicate lane"),
    ("unexpected", "lane state names an unexpected lane"),
    ("corrupt", "lane state is unreadable"),
    ("invalid", "lane state is unreadable"),
    ("missing", "lane state is missing"),
]


@pytest.mark.parametrize(
    "data",
    [
        None,
        [MARKER],
        {"lane": "lane-1", "status": MARKER, "iteration": 0},
        {"lane": "lane-1", "status": "awaiting_selection", "iteration": MARKER},
    ],
)
def test_unreadable_lane_diagnostic_does_not_include_content(
    tmp_path, monkeypatch, data
):
    from pydantic_ai_gepa.cli import layout

    monkeypatch.setattr(layout, "_explicit_gepa_dirname", str(tmp_path / ".gepa"))
    path = tmp_path / ".gepa/runs/test/lanes/lane-1/state.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data))
    with pytest.raises(LaneStateProblem, match="^lane state is unreadable$"):
        load_selection_lane_states(tmp_path, "test", 1)


def _plant(repo, monkeypatch, case):
    started = test_lane_accounting._start(repo, monkeypatch, 2, "100")
    run_id = str(started["run_id"])
    for lane in ("lane-1", "lane-2"):
        verdict = test_lane_accounting._drive(repo, run_id, lane)
        assert verdict.verdict == "accepted"
    directory = repo / ".gepa/runs" / run_id
    path = directory / "lanes/lane-2/state.json"
    data = json.loads(path.read_text())
    data["candidate_sha"] = MARKER
    if case == "duplicate":
        data["lane"] = "lane-1"
    elif case == "unexpected":
        data["lane"] = MARKER
    elif case == "invalid":
        data = {"lane": MARKER}
    path.write_text(json.dumps(data) if case != "corrupt" else MARKER + "{")
    if case == "missing":
        path.unlink()
        # The remaining lane is also untrusted and must not nominate a winner.
        other = directory / "lanes/lane-1/state.json"
        data = json.loads(other.read_text())
        data["candidate_sha"] = MARKER
        other.write_text(json.dumps(data))
    # A public incumbent forgery must not become the stop's accepted best.
    public = directory / "state.json"
    data = json.loads(public.read_text())
    data["best_candidate_id"] = MARKER
    data["best_commit_sha"] = MARKER
    public.write_text(json.dumps(data))
    return run_id, started


def _assert_stopped(repo, run_id, started, reason, output):
    directory = repo / ".gepa/runs" / run_id
    comparison = {"reason_code": "lane_state_invalid", "stop_reason": reason}
    state = test_select_cli._state(repo, run_id)
    assert state.status == "done"
    assert state.best_candidate_id == started["best_candidate_id"]
    assert state.best_commit_sha == started["best_commit_sha"]
    assert state.last_comparison == comparison
    report = (directory / "final_report.md").read_text()
    assert f"- best_candidate_id: {started['best_candidate_id']}\n" in report
    assert f"- accepted_best_candidate_id: {started['best_candidate_id']}\n" in report
    assert reason in report
    done = test_select_cli._events(repo, run_id, "run_done")
    assert len(done) == 1
    assert done[0]["payload"] == {
        "final_report_path": str(directory / "final_report.md")
    }
    with harness_environment():
        record = harness_record.for_run(run_id, repo)
        assert record is not None
        private = record.read("state.json")
        assert json.loads(private)["last_comparison"] == comparison
        assert json.loads(private)["best_candidate_id"] == started["best_candidate_id"]
        assert record.read("@lane-training-charges") is None
    visible = (
        (directory / "state.json").read_text()
        + json.dumps(test_select_cli._events(repo, run_id))
        + report
        + output
        + private
    )
    assert MARKER not in visible
    before = {
        path: path.read_bytes() for path in directory.rglob("*") if path.is_file()
    }
    again = test_select_cli._select(repo, run_id)
    assert again.exit_code == 1, (again.output, again.exception)
    assert MARKER not in again.output
    assert before == {path: path.read_bytes() for path in before}
    assert len(test_select_cli._events(repo, run_id, "run_done")) == 1
    with harness_environment():
        assert record.read("state.json") == private


@pytest.mark.parametrize("case,reason", CASES)
@pytest.mark.parametrize("resume", [False, True])
def test_bad_lane_state_finalizes(lane_repo, monkeypatch, case, reason, resume):
    run_id, started = _plant(lane_repo, monkeypatch, case)
    if resume:
        with harness_environment():
            state = run._load_state(run_id)
            select._checkpoint(state, lane_repo, "promote", {})
    result = test_select_cli._select(lane_repo, run_id)
    assert result.exit_code == 70, (result.output, result.exception)
    assert MARKER not in str(result.exception)
    _assert_stopped(lane_repo, run_id, started, reason, result.output)


def test_drive_returns_final_report_for_bad_lane_state(lane_repo, monkeypatch):
    run_id, started = _plant(lane_repo, monkeypatch, "corrupt")
    monkeypatch.setattr(drive, "DarwinProcesses", test_drive.OwnProcesses)
    path = test_drive.config(lane_repo, source="raise AssertionError('no reflection')")
    result = test_drive.invoke(run_id, path)
    assert result.exit_code == 0, (result.output, result.exception)
    assert "final_report.md" in result.output
    assert test_drive.driver_state(lane_repo, run_id)["pause_reason"] is None
    _assert_stopped(
        lane_repo, run_id, started, "lane state is unreadable", result.output
    )


@pytest.mark.parametrize("point", ["state", "report", "event"])
def test_interrupted_stop_repairs_without_duplicate_event(
    lane_repo, monkeypatch, point
):
    run_id, started = _plant(lane_repo, monkeypatch, "duplicate")
    with monkeypatch.context() as crash:
        if point == "state":
            original = run.RunState.save

            def fail(self, *args, **kwargs):
                original(self, *args, **kwargs)
                if self.status == "done":
                    raise RuntimeError("simulated interruption")

            crash.setattr(run.RunState, "save", fail)
        else:
            module, name = (
                (run, "_write_final_report") if point == "report" else (events, "emit")
            )
            original = getattr(module, name)

            def fail(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError("simulated interruption")

            crash.setattr(module, name, fail)
        result = test_select_cli._select(lane_repo, run_id)
        assert isinstance(result.exception, RuntimeError)
    repaired = test_select_cli._select(lane_repo, run_id)
    assert repaired.exit_code == 1, (repaired.output, repaired.exception)
    _assert_stopped(
        lane_repo,
        run_id,
        started,
        "lane state names a duplicate lane",
        result.output + repaired.output,
    )


def test_report_returns_incumbent_even_with_higher_unaccepted_row(
    lane_repo, monkeypatch
):
    run_id, started = _plant(lane_repo, monkeypatch, "unexpected")
    with harness_environment():
        ledger = ParetoLog(run_id, lane_repo)
        row = next(iter(ledger.validation_rows()))
        ledger.append(
            replace(row, candidate_id="unaccepted-finalist", mean_score=999.0)
        )
    result = test_select_cli._select(lane_repo, run_id)
    assert result.exit_code == 70, (result.output, result.exception)
    _assert_stopped(
        lane_repo,
        run_id,
        started,
        "lane state names an unexpected lane",
        result.output,
    )

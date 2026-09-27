"""Held-out lane charges use harness evidence, including candidate costs."""

import json
from dataclasses import replace

import pytest
import typer

from pydantic_ai_gepa.cli import harness_record
from pydantic_ai_gepa.cli.lane_accounting import KIND, consume, refresh
from pydantic_ai_gepa.cli.lanes import load_all_lane_states
from pydantic_ai_gepa.cli.layout import config_path
from pydantic_ai_gepa.cli.runs import ParetoLog
from pydantic_ai_gepa.cli.spend import spend_report
from pydantic_ai_gepa.cli.validation import harness_environment
from tests.cli import test_select_cli

lane_repo = test_select_cli.git_repo


def _start(repo, monkeypatch, lanes=2, cap="5", *extra):
    module = repo / "task_pkg/evaluation.py"
    module.write_text(
        module.read_text()
        + """
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_gepa.spend import current_rollout_capability
_original = evaluate
async def evaluate(case):
    name = "validation" if case.name.startswith("secret-") else "training"
    calls = int(Path("calls.txt").read_text()) if Path("calls.txt").exists() else 1
    for _ in range(calls):
        await Agent(TestModel(custom_output_text="ok", model_name=name)).run(
            "?", capabilities=[current_rollout_capability()])
    return await _original(case)
"""
    )
    (repo / "task_pkg/pricing.py").write_text(
        'def price(response): return 0.10 if response.model_name == "validation" else 0.005\n'
    )
    validation = repo.parent / "validation.jsonl"
    validation.write_text(
        json.dumps({"name": "secret-holdout", "inputs": "x", "expected_output": "v"})
        + "\n"
    )
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(validation))
    config = config_path(repo)
    config.write_text('price_fn = "task_pkg.pricing:price"\n' + config.read_text())
    test_select_cli._git(repo, "add", ".")
    test_select_cli._git(repo, "commit", "-m", "Meter fake student")
    return test_select_cli._start_lane_run(repo, lanes, "--max-token-cost", cap, *extra)


def _drive(repo, run_id, lane="lane-1", calls=1):
    return test_select_cli._drive_lane(
        repo,
        run_id,
        lane,
        {
            "calls.txt": f"{calls}\n",
            "out_case-2.txt": "b\n",
            "out_secret-holdout.txt": "v\n",
        },
    )


def _charges(repo, run_id):
    return [
        row
        for line in (repo / ".gepa/runs" / run_id / "spend.jsonl")
        .read_text()
        .splitlines()
        if (row := json.loads(line))["kind"] == KIND
    ]


@pytest.mark.parametrize("tamper", ["honest", "empty", "deleted", "false"])
def test_multi_lane_charges_ignore_reflector_ledgers(lane_repo, monkeypatch, tamper):
    started = _start(lane_repo, monkeypatch, 2, "2")
    run_id = str(started["run_id"])
    for lane in ("lane-1", "lane-2"):
        _drive(lane_repo, run_id, lane)
        directory = lane_repo / ".gepa/runs" / run_id / "lanes" / lane
        for name in ("spend.jsonl", "pareto.jsonl"):
            path = directory / name
            if tamper == "empty":
                path.write_text("")
            elif tamper == "deleted":
                path.unlink()
            elif tamper == "false":
                path.write_text('{"total_dollars": 0, "iterations": -99999}\n')
    result = test_select_cli._select(lane_repo, run_id)
    assert result.exit_code == 70, (result.output, result.exception)
    state = test_select_cli._state(lane_repo, run_id)
    assert state.status == "done"
    assert state.best_candidate_id == started["best_candidate_id"]
    report = spend_report(run_id, lane_repo)
    assert report["estimated_lane_training_dollars"] == pytest.approx(3.6)
    assert report["total_dollars"] == pytest.approx(3.96)
    assert report["total_dollars"] <= 2 + report["estimated_lane_training_dollars"]
    assert len(_charges(lane_repo, run_id)) == 2
    final = (lane_repo / ".gepa/runs" / run_id / "final_report.md").read_text()
    assert "estimated_lane_training_dollars" in final
    assert "latest_candidate_id: lane-training-" not in final


def test_refan_projects_all_lanes_and_keeps_promoted_best(lane_repo, monkeypatch):
    started = _start(lane_repo, monkeypatch)
    run_id = str(started["run_id"])
    for lane in ("lane-1", "lane-2"):
        _drive(lane_repo, run_id, lane)
    result = test_select_cli._select(lane_repo, run_id)
    assert result.exit_code == 70, (result.output, result.exception)
    state = test_select_cli._state(lane_repo, run_id)
    report = spend_report(run_id, lane_repo)
    assert state.status == "done"
    assert state.best_candidate_id != started["best_candidate_id"]
    assert report["total_dollars"] < 5
    assert report["total_dollars"] + 3.6 > 5
    assert report["estimated_lane_training_dollars"] == pytest.approx(3.6)
    assert all(
        lane.iteration == started["iterations"]
        for lane in load_all_lane_states(lane_repo, run_id)
    )


@pytest.mark.parametrize("cap", ["50", "250"])
def test_candidate_metering_covers_uniform_100_call_counterexample(
    lane_repo, monkeypatch, cap
):
    started = _start(lane_repo, monkeypatch, 1, cap)
    run_id = str(started["run_id"])
    verdict = _drive(lane_repo, run_id, calls=100)
    assert verdict.verdict == "accepted"
    assert len(verdict.eval_samples) == 3
    training = lane_repo / ".gepa/runs" / run_id / "lanes/lane-1/spend.jsonl"
    actual = sum(
        json.loads(line)["total_dollars"] for line in training.read_text().splitlines()
    )
    assert actual == pytest.approx(4.5)
    result = test_select_cli._select(lane_repo, run_id)
    assert result.exit_code == 70, (result.output, result.exception)
    charge = _charges(lane_repo, run_id)[0]
    assert charge["total_dollars"] >= actual
    assert charge["total_dollars"] == pytest.approx(180)
    # Public charge rows contain only an opaque identity, kind and aggregate.
    assert set(charge) == {"eval_id", "kind", "total_dollars"}
    with monkeypatch.context() as public:
        public.delenv("GEPA_HELDOUT_DATASET")
        status = test_select_cli._run("run", "status", "--run-id", run_id)
    assert status.exit_code == 0, (status.output, status.exception)
    assert "secret-holdout" not in status.output
    for private_field in (
        "max_rollout_dollars",
        "per_rollout",
        "rollouts_completed",
        "rollouts_started",
    ):
        assert private_field not in status.output
    assert "estimated_lane_training_dollars" in status.output


def test_training_budget_rows_are_resume_safe(lane_repo, monkeypatch):
    started = _start(lane_repo, monkeypatch, 2, "100")
    run_id = str(started["run_id"])
    for lane in ("lane-1", "lane-2"):
        _drive(lane_repo, run_id, lane)
    with harness_environment():
        state = test_select_cli._state(lane_repo, run_id)
        lanes = load_all_lane_states(lane_repo, run_id)
        before = ParetoLog(run_id, lane_repo).count_budget_rows()
        original = ParetoLog.append
        with monkeypatch.context() as interrupted:

            def crash_after_first_row(self, row):
                original(self, row)
                raise RuntimeError("interrupted budget accounting")

            interrupted.setattr(ParetoLog, "append", crash_after_first_row)
            with pytest.raises(RuntimeError, match="interrupted budget"):
                consume(state, lane_repo, lanes)
        assert ParetoLog(run_id, lane_repo).count_budget_rows() == before + 1
        consume(state, lane_repo, lanes)
        once = _charges(lane_repo, run_id)
        assert ParetoLog(run_id, lane_repo).count_budget_rows() == before + 6
        consume(state, lane_repo, lanes)
        refresh(run_id, lane_repo)
        assert _charges(lane_repo, run_id) == once
        assert ParetoLog(run_id, lane_repo).count_budget_rows() == before + 6
        # The private checkpoint carries the multiplier; public rows do not.
        record = harness_record.for_run(run_id, lane_repo)
        assert record is not None
        assert record.read("@lane-training-charges") is not None


def test_iteration_limit_counts_lane_verdicts(lane_repo, monkeypatch):
    started = _start(lane_repo, monkeypatch, 2, "100", "--max-iterations", "18")
    run_id = str(started["run_id"])
    for lane in ("lane-1", "lane-2"):
        _drive(lane_repo, run_id, lane)
    result = test_select_cli._select(lane_repo, run_id)
    assert result.exit_code == 0, (result.output, result.exception)
    state = test_select_cli._state(lane_repo, run_id)
    assert state.status == "done"
    assert state.iterations == 18
    assert state.best_candidate_id != started["best_candidate_id"]


def test_missing_metered_estimate_stops_and_keeps_best(lane_repo, monkeypatch):
    started = _start(lane_repo, monkeypatch, 1, "100")
    run_id = str(started["run_id"])
    _drive(lane_repo, run_id)
    with harness_environment():
        record = harness_record.for_run(run_id, lane_repo)
        assert record is not None
        # Model a legacy/unmetered harness history. Positive-looking dollar
        # fields without metered rollouts must never authorize lane estimates.
        rows = [json.loads(line) for line in record.read("spend.jsonl").splitlines()]
        for row in rows:
            row["unmetered_rollouts"] = 1
        record.write("spend.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
        state = replace(test_select_cli._state(lane_repo, run_id), max_token_cost=100)
        with pytest.raises(typer.Exit) as error:
            consume(state, lane_repo, load_all_lane_states(lane_repo, run_id))
        assert error.value.exit_code == 70
    final = test_select_cli._state(lane_repo, run_id)
    assert final.status == "done"
    assert final.best_candidate_id == started["best_candidate_id"]
    assert "No harness-metered cost" in final.last_comparison["stop_reason"]

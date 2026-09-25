"""Reflector replacement and crash recovery use durable harness evidence."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pytest
import typer

from pydantic_ai_gepa.cli import run as run_module
from pydantic_ai_gepa.cli import layout
from pydantic_ai_gepa.cli.runs import ParetoLog, ParetoRow
from tests.cli import test_git_candidate_cli, test_run_cli, test_select_cli
from tests.cli.test_git_candidate_cli import (
    _git,
    _run,
    _run_payload,
)


git_repo = test_git_candidate_cli.git_repo
repo = test_run_cli.repo
validation_repo = test_select_cli.git_repo


@pytest.fixture(autouse=True)
def isolated_workspace_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(layout, "_explicit_gepa_dirname", None)
    monkeypatch.delenv("GEPA_DIR", raising=False)


def _start(*extra: str, size: int = 1, budget: int = 8, repetitions: int = 1):
    result = _run(
        "run",
        "start",
        "--size",
        str(size),
        "--max-iterations",
        str(budget),
        "--acceptance-repetitions",
        str(repetitions),
        *extra,
    )
    assert result.exit_code == 0, result.output
    return _run_payload(result.output)


def _resume(run_id: str, *extra: str):
    result = _run("run", "resume", "--run-id", run_id, *extra)
    assert result.exit_code == 0, result.output
    return json.loads(result.output.splitlines()[-1])["packet"]


def _packet(repo: Path, run_id: str):
    return json.loads(
        (repo / ".gepa" / "runs" / run_id / "reflector_packet.json").read_text()
    )


def _commit_score(repo: Path, score: str) -> str:
    (repo / "score.txt").write_text(score + "\n")
    _git(repo, "add", "score.txt")
    _git(repo, "commit", "-m", "Try reflected candidate")
    return _git(repo, "rev-parse", "HEAD")


def test_resume_preserves_unscored_commit_and_issues_complete_packet(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = _start()
    run_id = str(started["run_id"])
    baseline = _git(git_repo, "rev-parse", "HEAD")
    committed = _commit_score(git_repo, "still-bad")
    pareto_path = git_repo / ".gepa" / "runs" / run_id / "pareto.jsonl"
    before_rows = pareto_path.read_bytes()
    before_tree = _git(git_repo, "diff", "HEAD", "--", "score.txt")

    packet = _resume(run_id, "--reason", "usage limit", "--reflector", "fresh-agent")

    assert packet == _packet(git_repo, run_id)
    assert packet["packet_version"] == 1
    assert packet["status"] == "paused_for_reflection"
    assert packet["baseline"]["commit_sha"] == baseline
    assert packet["baseline"]["report_paths"]
    assert all(Path(path).exists() for path in packet["baseline"]["report_paths"])
    assert packet["current_tree"]["commit_sha"] == committed
    assert packet["current_tree"]["differs_from_baseline"] is True
    assert packet["current_tree"]["already_scored"] is False
    assert packet["reflector"]["label"] == "fresh-agent"
    assert packet["reflector"]["state"] == "active"
    assert packet["reflector"]["history"][-1]["state"] == "lost"
    lost = [
        row for row in packet["journal_tail"] if row.get("kind") == "reflector_lost"
    ]
    assert lost[-1]["reason"] == "usage limit"
    assert committed in json.dumps(lost[-1])
    assert "not scored" in packet["instructions"]
    assert pareto_path.read_bytes() == before_rows
    assert _git(git_repo, "rev-parse", "HEAD") == committed
    assert _git(git_repo, "diff", "HEAD", "--", "score.txt") == before_tree
    assert run_module._load_state(run_id).iterations == started["iterations"]

    command = packet["next_command"]
    assert Path(command["cwd"]).is_absolute()
    assert "--reflector-epoch" in command["argv"]
    assert str(git_repo / ".gepa") in command["argv"]
    assert command["shell"]
    original = run_module.run_eval_once
    calls = []

    def count_calls(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(run_module, "run_eval_once", count_calls)
    # argv is paired with an explicit cwd; the shell form supplies its own cd.
    assert "cd " in command["shell"]
    monkeypatch.chdir(command["cwd"])
    continued = _run(*command["argv"][1:])
    assert continued.exit_code == 0, continued.output
    assert len(calls) == 1
    assert _run_payload(continued.output)["reflector_packet_path"]


@pytest.mark.parametrize(
    "score,budget,verdict",
    [
        ("good", 2, "accepted"),
        ("good", 8, "accepted"),
        ("still-bad", 8, "equivalent"),
        ("still-bad", 8, "rejected"),
    ],
)
def test_resume_after_finished_continue_never_rescores(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    score: str,
    budget: int,
    verdict: str,
) -> None:
    if (score == "good" and budget == 8) or verdict == "rejected":
        dataset = git_repo / ".gepa" / "dataset.jsonl"
        dataset.write_text(
            dataset.read_text()
            + json.dumps(
                {"name": "case-2", "inputs": "?", "expected_output": "unmatched"}
            )
            + "\n"
        )
        _git(git_repo, "add", ".gepa/dataset.jsonl")
        _git(git_repo, "commit", "-m", "Add another training case")
    if verdict == "rejected":
        _commit_score(git_repo, "good")
    started = _start(budget=budget, size=2)
    run_id = str(started["run_id"])
    candidate_sha = _commit_score(git_repo, score)
    continued = _run("run", "continue", "--run-id", run_id)
    assert continued.exit_code == 0, continued.output
    assert _packet(git_repo, run_id)["last_comparison"]["verdict"] == verdict
    if score == "good" and budget == 8:
        assert _run_payload(continued.output)["status"] == "paused_for_reflection"
    before = ParetoLog(run_id).count_budget_rows()
    monkeypatch.setattr(
        run_module,
        "run_eval_once",
        lambda **_: pytest.fail("recorded candidate must not run again"),
    )

    packet = _resume(run_id)
    assert packet["current_tree"]["commit_sha"] == candidate_sha
    assert packet["current_tree"]["already_scored"] is True
    assert packet["last_comparison"]["verdict"] == verdict
    repeated = _run("run", "continue", "--run-id", run_id)
    assert repeated.exit_code == 0, repeated.output
    assert _packet(git_repo, run_id)["last_comparison"]["verdict"] == verdict
    assert ParetoLog(run_id).count_budget_rows() == before


@pytest.mark.parametrize("completed_before_death", [1, 2, 3])
def test_interrupted_continue_reuses_persisted_training_samples(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch, completed_before_death: int
) -> None:
    started = _start(repetitions=3, budget=6)
    run_id = str(started["run_id"])
    _commit_score(git_repo, "still-bad")
    original = run_module.run_eval_once
    calls = 0

    def die_after_durable_sample(**kwargs):
        nonlocal calls
        calls += 1
        outcome = original(**kwargs)
        if calls == completed_before_death:
            raise RuntimeError("reflector child died after appending its evaluation")
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", die_after_durable_sample)
    interrupted = _run("run", "continue", "--run-id", run_id)
    assert interrupted.exit_code != 0
    assert ParetoLog(run_id).count_budget_rows() == 3 + completed_before_death
    resumed = _run("run", "continue", "--run-id", run_id)
    assert resumed.exit_code == 0, resumed.output
    comparison = _run_payload(resumed.output)["last_comparison"]
    assert comparison["candidate_sample_count"] == 3
    assert len(comparison["candidate_report_paths"]) == 3
    assert calls == 3
    assert ParetoLog(run_id).count_budget_rows() == 6


def test_resume_fences_stale_epoch_and_keeps_legacy_continue(git_repo: Path) -> None:
    started = _start()
    run_id = str(started["run_id"])
    _commit_score(git_repo, "still-bad")
    packet = _resume(run_id)
    first_epoch = packet["reflector"]["epoch"]
    packet = _resume(run_id)
    assert packet["reflector"]["epoch"] > first_epoch
    before = ParetoLog(run_id).count_budget_rows()
    stale = _run(
        "run", "continue", "--run-id", run_id, "--reflector-epoch", str(first_epoch)
    )
    assert stale.exit_code == 2, stale.output
    assert str(packet["reflector"]["epoch"]) in stale.output
    assert "resume" in stale.output
    assert ParetoLog(run_id).count_budget_rows() == before
    legacy = _run("run", "continue", "--run-id", run_id)
    assert legacy.exit_code == 0, legacy.output
    assert ParetoLog(run_id).count_budget_rows() == before + 1


@pytest.mark.parametrize("command", ["resume", "continue"])
def test_live_run_lock_refuses_commands(git_repo: Path, command: str) -> None:
    started = _start()
    run_id = str(started["run_id"])
    lock_path = git_repo / ".gepa" / "runs" / run_id / "run.lock"
    before = ParetoLog(run_id).count_budget_rows()
    with lock_path.open("w") as holder:
        holder.write(str(os.getpid()))
        holder.flush()
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run("run", command, "--run-id", run_id)
        assert result.exit_code == 1, result.output
    assert ParetoLog(run_id).count_budget_rows() == before
    assert _resume(run_id)["run_id"] == run_id


def test_resume_refuses_lane_runs(git_repo: Path) -> None:
    started = _start("--lanes", "1")
    result = _run("run", "resume", "--run-id", str(started["run_id"]))
    assert result.exit_code == 2, result.output
    assert "lane reset" in result.output
    assert "lane lease" in result.output


def test_resume_latest_legacy_state_preserves_dirty_tree(git_repo: Path) -> None:
    started = _start()
    run_id = str(started["run_id"])
    state_path = git_repo / ".gepa" / "runs" / run_id / "state.json"
    state = json.loads(state_path.read_text())
    state.pop("reflector")
    state_path.write_text(json.dumps(state))
    (git_repo / "score.txt").write_text("uncommitted work\n")
    (state_path.parent / "run.lock").write_text("999999999\n")
    before = ParetoLog(run_id).count_budget_rows()

    resumed = _run("run", "resume", "--reflector", "replacement")

    assert resumed.exit_code == 0, resumed.output
    packet = json.loads(resumed.output.splitlines()[-1])["packet"]
    assert packet["run_id"] == run_id
    assert packet["current_tree"]["dirty"] is True
    assert packet["current_tree"]["differs_from_baseline"] is True
    assert packet["current_tree"]["already_scored"] is False
    assert packet["reflector"]["label"] == "replacement"
    assert (git_repo / "score.txt").read_text() == "uncommitted work\n"
    assert ParetoLog(run_id).count_budget_rows() == before


def test_resume_preserves_infrastructure_failure_phase(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_run_cli._fail_rollout_calls(monkeypatch, {1})
    started = _start(size=2)
    run_id = str(started["run_id"])
    assert started["status"] == "paused_after_infrastructure_error"
    before = ParetoLog(run_id).count_budget_rows()
    packet = _resume(run_id)
    assert packet["status"] == "paused_after_infrastructure_error"
    assert packet["last_comparison"]["outcome"] == "infrastructure_failure"
    assert ParetoLog(run_id).count_budget_rows() == before


def test_component_edits_survive_resume_and_recorded_comparison(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = _start(size=2)
    run_id = str(started["run_id"])
    component = next((repo / ".gepa" / "components").glob("*.md"))
    component.write_text("Revised geography instructions.")
    packet = _resume(run_id)
    assert packet["candidate_source"] == "components"
    assert packet["current_tree"]["differs_from_baseline"] is True
    assert component.read_text() == "Revised geography instructions."
    continued = _run("run", "continue", "--run-id", run_id)
    assert continued.exit_code == 0, continued.output
    monkeypatch.setattr(
        run_module,
        "run_eval_once",
        lambda **_: pytest.fail("component candidate already scored"),
    )
    repeated = _run("run", "continue", "--run-id", run_id)
    assert repeated.exit_code == 0, repeated.output


def test_validation_row_survives_death_without_leaking_evidence(
    validation_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_path = validation_repo.parent / "withheld-validation.jsonl"
    validation_path.write_text(
        json.dumps(
            {
                "name": "secret-resume-case",
                "inputs": "PRIVATE_INPUT",
                "expected_output": "v",
            }
        )
        + "\n"
    )
    config = validation_repo / ".gepa" / "gepa.toml"
    config.write_text(
        config.read_text() + f'validation_dataset = "{validation_path}"\n'
    )
    _git(validation_repo, "add", ".gepa/gepa.toml")
    _git(validation_repo, "commit", "-m", "Configure validation")
    started = _start(size=3, budget=4)
    run_id = str(started["run_id"])
    (validation_repo / "out_case-2.txt").write_text("b\n")
    _git(validation_repo, "add", "out_case-2.txt")
    _git(validation_repo, "commit", "-m", "Improve training candidate")
    original = run_module.run_eval_once
    calls = []

    def die_after_validation(**kwargs):
        calls.append(kwargs.get("dataset_role", "training"))
        outcome = original(**kwargs)
        if kwargs.get("dataset_role") == "validation":
            raise RuntimeError("died after the validation row was persisted")
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", die_after_validation)
    interrupted = _run("run", "continue", "--run-id", run_id)
    assert interrupted.exit_code != 0
    assert calls == ["training", "validation"]
    assert ParetoLog(run_id).count_budget_rows() == 4
    monkeypatch.setattr(
        run_module,
        "run_eval_once",
        lambda **_: pytest.fail("training and validation rows are already durable"),
    )
    resumed = _run("run", "continue", "--run-id", run_id)
    assert resumed.exit_code == 0, resumed.output
    packet = _resume(run_id)
    serialized = json.dumps(packet)
    for secret in (
        str(validation_path),
        "secret-resume-case",
        "PRIVATE_INPUT",
        "validation_dataset_digest",
        "per_case_scores",
    ):
        assert secret not in serialized
    assert packet["last_comparison"]["validation"]["evaluated"] is True
    assert packet["last_comparison"]["validation"]["improved"] is False
    assert len(ParetoLog(run_id).validation_rows()) == 2
    assert ParetoLog(run_id).count_budget_rows() == 4


@pytest.mark.parametrize("baseline_samples_before_death", [1, 2])
def test_post_acceptance_baseline_reuses_partial_paid_samples(
    validation_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    baseline_samples_before_death: int,
) -> None:
    started = _start(size=3, repetitions=3, budget=12)
    run_id = str(started["run_id"])
    (validation_repo / "out_case-2.txt").write_text("b\n")
    _git(validation_repo, "add", "out_case-2.txt")
    _git(validation_repo, "commit", "-m", "Improve one training case")
    original = run_module.run_eval_once
    calls = []

    def die_during_next_baseline(**kwargs):
        calls.append(kwargs)
        outcome = original(**kwargs)
        if len(calls) == 3 + baseline_samples_before_death:
            raise RuntimeError("died while advancing the accepted baseline")
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", die_during_next_baseline)
    interrupted = _run("run", "continue", "--run-id", run_id)
    assert interrupted.exit_code != 0
    assert ParetoLog(run_id).count_budget_rows() == 6 + baseline_samples_before_death
    resumed = _run("run", "continue", "--run-id", run_id)
    assert resumed.exit_code == 0, resumed.output
    payload = _run_payload(resumed.output)
    assert payload["status"] == "paused_for_reflection"
    assert payload["reflection_minibatch_id"] != started["reflection_minibatch_id"]
    assert payload["reflection_baseline_samples"] == [pytest.approx(2 / 3)] * 3
    assert _packet(validation_repo, run_id)["last_comparison"]["verdict"] == "accepted"
    assert len(calls) == 6
    assert ParetoLog(run_id).count_budget_rows() == 9


def test_interrupted_infrastructure_result_remains_failure_on_recovery(
    validation_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = _start(size=3)
    run_id = str(started["run_id"])
    (validation_repo / "out_case-2.txt").write_text("wrong\n")
    _git(validation_repo, "add", "out_case-2.txt")
    _git(validation_repo, "commit", "-m", "Try candidate before provider outage")
    monkeypatch.setenv("GEPA_TEST_REBASELINE_FAILURE", "1")
    original = run_module.run_eval_once

    def die_after_failure_row(**kwargs):
        original(**kwargs)
        raise RuntimeError("died after writing a real failed rollout row")

    monkeypatch.setattr(run_module, "run_eval_once", die_after_failure_row)
    interrupted = _run("run", "continue", "--run-id", run_id)
    assert interrupted.exit_code != 0
    assert ParetoLog(run_id).count_budget_rows() == 2
    monkeypatch.delenv("GEPA_TEST_REBASELINE_FAILURE")
    monkeypatch.setattr(
        run_module,
        "run_eval_once",
        lambda **_: pytest.fail("the recorded failure must be recovered first"),
    )
    recovered = _run("run", "continue", "--run-id", run_id)
    assert recovered.exit_code == 0, recovered.output
    payload = _run_payload(recovered.output)
    assert payload["status"] == "paused_after_infrastructure_error"
    assert payload["last_comparison"]["outcome"] == "infrastructure_failure"
    assert payload["last_comparison"]["verdict"] is None
    assert payload["best_candidate_id"] == started["best_candidate_id"]
    assert ParetoLog(run_id).count_budget_rows() == 2

    monkeypatch.setattr(run_module, "run_eval_once", original)
    retried = _run("run", "continue", "--run-id", run_id)
    assert retried.exit_code == 0, retried.output
    assert _run_payload(retried.output)["last_comparison"]["verdict"] == "equivalent"
    assert ParetoLog(run_id).count_budget_rows() == 3


def test_interrupted_gate_continue_reuses_gate_checkpoint(
    validation_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = _start(size=3)
    run_id = str(started["run_id"])
    (validation_repo / "out_case-3.txt").write_text("wrong\n")
    _git(validation_repo, "add", "out_case-3.txt")
    _git(validation_repo, "commit", "-m", "Try gated candidate")
    original = run_module.run_eval_once
    calls = []

    def die_after_paid_sample(**kwargs):
        calls.append(kwargs)
        outcome = original(**kwargs)
        if kwargs.get("write_pareto", True):
            raise RuntimeError("died after the full training sample")
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", die_after_paid_sample)
    args = ("run", "continue", "--run-id", run_id, "--gate-case", "case-1")
    interrupted = _run(*args)
    assert interrupted.exit_code != 0
    assert len(calls) == 2
    assert calls[0]["write_pareto"] is False
    assert ParetoLog(run_id).count_budget_rows() == 2
    monkeypatch.setattr(
        run_module,
        "run_eval_once",
        lambda **_: pytest.fail("gate and training sample were already evaluated"),
    )
    resumed = _run(*args)
    assert resumed.exit_code == 0, resumed.output
    comparison = _run_payload(resumed.output)["last_comparison"]
    assert comparison["verdict"] == "equivalent"
    assert comparison["gate"]["cases"] == ["case-1"]
    assert ParetoLog(run_id).count_budget_rows() == 2


@pytest.mark.parametrize("refusal", ["invalid_gate", "unpaid_eval_exit"])
def test_unpaid_refusal_does_not_block_corrected_candidate(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch, refusal: str
) -> None:
    started = _start()
    run_id = str(started["run_id"])
    _commit_score(git_repo, "first-attempt")
    original = run_module.run_eval_once

    def refuse_before_evaluation(**kwargs):
        raise typer.Exit(code=2)

    if refusal == "invalid_gate":
        args = ("--gate-case", "unknown-case")
    else:
        args = ()
        monkeypatch.setattr(run_module, "run_eval_once", refuse_before_evaluation)
    refused = _run("run", "continue", "--run-id", run_id, *args)
    assert refused.exit_code == 2, refused.output
    assert run_module._load_state(run_id).continuation is None
    assert ParetoLog(run_id).count_budget_rows() == 1

    monkeypatch.setattr(run_module, "run_eval_once", original)
    _commit_score(git_repo, "corrected-attempt")
    corrected = _run("run", "continue", "--run-id", run_id)
    assert corrected.exit_code == 0, corrected.output
    assert _run_payload(corrected.output)["last_comparison"]["verdict"] == "equivalent"
    assert ParetoLog(run_id).count_budget_rows() == 2


def test_paid_row_is_published_only_after_training_artifacts_exist(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = _start()
    run_id = str(started["run_id"])
    _commit_score(git_repo, "still-bad")
    original_append = ParetoLog.append
    published_artifacts = []

    def die_after_publish(ledger: ParetoLog, row: ParetoRow):
        assert row.extra["dataset_role"] == "training"
        iteration = ledger.count_budget_rows() + 1
        stem = f"{iteration:04d}-{row.extra['eval_id']}-{row.candidate_id}"
        run_root = git_repo / ".gepa" / "runs" / run_id
        report = run_root / "reports" / f"{stem}.md"
        trace = run_root / "traces" / "minibatches" / row.minibatch_id / f"{stem}.jsonl"
        assert report.is_file()
        assert trace.is_file()
        assert report.stat().st_size > 0
        assert trace.stat().st_size > 0
        published_artifacts.extend([report, trace])
        original_append(ledger, row)
        raise RuntimeError("died immediately after publishing the paid row")

    monkeypatch.setattr(ParetoLog, "append", die_after_publish)
    interrupted = _run("run", "continue", "--run-id", run_id)
    assert isinstance(interrupted.exception, RuntimeError), interrupted.output
    assert "after publishing" in str(interrupted.exception)
    assert ParetoLog(run_id).count_budget_rows() == 2
    monkeypatch.setattr(ParetoLog, "append", original_append)
    monkeypatch.setattr(
        run_module,
        "run_eval_once",
        lambda **_: pytest.fail("the paid row and training artifacts already exist"),
    )
    resumed = _run("run", "continue", "--run-id", run_id)
    assert resumed.exit_code == 0, resumed.output
    comparison = _run_payload(resumed.output)["last_comparison"]
    assert comparison["candidate_report_paths"] == [str(published_artifacts[0])]
    assert comparison["candidate_trace_paths"] == [str(published_artifacts[1])]
    assert ParetoLog(run_id).count_budget_rows() == 2


def test_validation_failure_packet_withholds_selection_minibatch_identity(
    validation_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_path = validation_repo.parent / "private-validation.jsonl"
    validation_path.write_text(
        json.dumps(
            {
                "name": "secret-validation-failure",
                "inputs": "private",
                "expected_output": "v",
            }
        )
        + "\n"
    )
    config = validation_repo / ".gepa" / "gepa.toml"
    config.write_text(
        config.read_text() + f'validation_dataset = "{validation_path}"\n'
    )
    _git(validation_repo, "add", ".gepa/gepa.toml")
    _git(validation_repo, "commit", "-m", "Configure private validation")
    started = _start(size=3)
    run_id = str(started["run_id"])
    (validation_repo / "out_case-2.txt").write_text("b\n")
    _git(validation_repo, "add", "out_case-2.txt")
    _git(validation_repo, "commit", "-m", "Improve training candidate")
    monkeypatch.setenv("GEPA_TEST_FAIL_VALIDATION", "1")
    failed = _run("run", "continue", "--run-id", run_id)
    assert failed.exit_code == 0, failed.output
    state = run_module._load_state(run_id)
    assert state.status == "paused_after_infrastructure_error"
    assert state.last_comparison is not None
    selection_minibatch = state.last_comparison["minibatch_id"]
    assert selection_minibatch.startswith("validation-")

    packet = _resume(run_id)

    serialized = json.dumps(packet)
    assert selection_minibatch not in serialized
    assert str(validation_path) not in serialized
    assert "secret-validation-failure" not in serialized
    assert packet["last_comparison"]["outcome"] == "infrastructure_failure"

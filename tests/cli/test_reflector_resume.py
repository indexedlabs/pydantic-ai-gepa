"""Reflector replacement and crash recovery use durable harness evidence."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pytest
from tests.cli.harness_helpers import scored_continue
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
    assert len(calls) == 3
    assert _run_payload(continued.output)["reflector_packet_path"]


@pytest.mark.parametrize(
    "score,budget,verdict",
    [
        ("good", 7, "accepted"),
        ("good", 14, "accepted"),
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
    if (score == "good" and budget == 14) or verdict == "rejected":
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
    if score == "good" and budget == 14:
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
    started = _start(repetitions=3, budget=7)
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
    assert ParetoLog(run_id).count_budget_rows() == 4 + completed_before_death
    resumed = _run("run", "continue", "--run-id", run_id)
    assert resumed.exit_code == 0, resumed.output
    comparison = _run_payload(resumed.output)["last_comparison"]
    assert comparison["candidate_sample_count"] == 3
    assert len(comparison["candidate_report_paths"]) == 3
    assert calls == 3
    assert ParetoLog(run_id).count_budget_rows() == 7


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
    assert ParetoLog(run_id).count_budget_rows() == before + 3


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
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(validation_path))
    _git(validation_repo, "add", ".gepa/gepa.toml")
    _git(validation_repo, "commit", "--allow-empty", "-m", "Configure validation")
    started = _start(size=3, budget=13)
    run_id = str(started["run_id"])
    (validation_repo / "out_case-2.txt").write_text("b\n")
    _git(validation_repo, "add", "out_case-2.txt")
    _git(validation_repo, "commit", "-m", "Improve training candidate")
    original = run_module.run_eval_once
    calls = []

    def die_after_validation(**kwargs):
        calls.append(kwargs.get("dataset_role", "training"))
        outcome = original(**kwargs)
        if calls.count("validation") == 3:
            raise RuntimeError("died after the validation row was persisted")
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", die_after_validation)
    interrupted = scored_continue("run", "continue", "--run-id", run_id)
    assert interrupted.exit_code != 0
    assert calls == ["training"] * 3 + ["validation"] * 3
    assert ParetoLog(run_id).count_budget_rows() == 13
    monkeypatch.setattr(
        run_module,
        "run_eval_once",
        lambda **_: pytest.fail("training and validation rows are already durable"),
    )
    resumed = scored_continue("run", "continue", "--run-id", run_id)
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
    assert len(ParetoLog(run_id).validation_rows()) == 6
    assert ParetoLog(run_id).count_budget_rows() == 13


@pytest.mark.parametrize("baseline_samples_before_death", [0, 1, 2, 3])
def test_post_acceptance_baseline_reuses_partial_paid_samples(
    validation_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    baseline_samples_before_death: int,
) -> None:
    started = _start(size=3, repetitions=3, budget=14)
    run_id = str(started["run_id"])
    (validation_repo / "out_case-2.txt").write_text("b\n")
    _git(validation_repo, "add", "out_case-2.txt")
    _git(validation_repo, "commit", "-m", "Improve one training case")
    original = run_module.run_eval_once
    calls = []

    def die_during_next_baseline(**kwargs):
        calls.append(kwargs)
        outcome = original(**kwargs)
        if len(calls) == 4 + baseline_samples_before_death:
            raise RuntimeError("died while advancing the accepted baseline")
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", die_during_next_baseline)
    interrupted = _run("run", "continue", "--run-id", run_id)
    assert interrupted.exit_code != 0
    assert ParetoLog(run_id).count_budget_rows() == 8 + baseline_samples_before_death
    resumed = _run("run", "continue", "--run-id", run_id)
    assert resumed.exit_code == 0, resumed.output
    payload = _run_payload(resumed.output)
    assert payload["status"] == "paused_for_reflection"
    assert payload["reflection_minibatch_id"] != started["reflection_minibatch_id"]
    assert payload["reflection_baseline_samples"] == [pytest.approx(2 / 3)] * 3
    assert _packet(validation_repo, run_id)["last_comparison"]["verdict"] == "accepted"
    assert len(calls) == 7
    assert ParetoLog(run_id).count_budget_rows() == 11


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
    assert ParetoLog(run_id).count_budget_rows() == 5
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
    assert ParetoLog(run_id).count_budget_rows() == 5

    monkeypatch.setattr(run_module, "run_eval_once", original)
    retried = _run("run", "continue", "--run-id", run_id)
    assert retried.exit_code == 0, retried.output
    assert _run_payload(retried.output)["last_comparison"]["verdict"] == "equivalent"
    assert ParetoLog(run_id).count_budget_rows() == 8


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
        if sum(call.get("write_pareto", True) for call in calls) == 3:
            raise RuntimeError("died after the full training sample")
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", die_after_paid_sample)
    args = ("run", "continue", "--run-id", run_id, "--gate-case", "case-1")
    interrupted = _run(*args)
    assert interrupted.exit_code != 0
    assert len(calls) == 6
    assert calls[0]["write_pareto"] is False
    assert ParetoLog(run_id).count_budget_rows() == 7
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
    assert ParetoLog(run_id).count_budget_rows() == 7


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
    assert ParetoLog(run_id).count_budget_rows() == 4

    monkeypatch.setattr(run_module, "run_eval_once", original)
    _commit_score(git_repo, "corrected-attempt")
    corrected = _run("run", "continue", "--run-id", run_id)
    assert corrected.exit_code == 0, corrected.output
    assert _run_payload(corrected.output)["last_comparison"]["verdict"] == "equivalent"
    assert ParetoLog(run_id).count_budget_rows() == 7


def test_paid_row_is_published_only_after_training_artifacts_exist(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = _start()
    run_id = str(started["run_id"])
    _commit_score(git_repo, "still-bad")
    original_append = ParetoLog.append
    published_artifacts = []
    original_evaluate = run_module.run_eval_once

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
    assert ParetoLog(run_id).count_budget_rows() == 5
    monkeypatch.setattr(ParetoLog, "append", original_append)
    remaining_calls = []

    def evaluate_remaining(**kwargs):
        remaining_calls.append(kwargs)
        return original_evaluate(**kwargs)

    monkeypatch.setattr(run_module, "run_eval_once", evaluate_remaining)
    resumed = _run("run", "continue", "--run-id", run_id)
    assert resumed.exit_code == 0, resumed.output
    comparison = _run_payload(resumed.output)["last_comparison"]
    assert comparison["candidate_report_paths"][0] == str(published_artifacts[0])
    assert comparison["candidate_trace_paths"][0] == str(published_artifacts[1])
    assert len(remaining_calls) == 2
    assert len(comparison["candidate_report_paths"]) == 3
    assert ParetoLog(run_id).count_budget_rows() == 7


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
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(validation_path))
    _git(validation_repo, "add", ".gepa/gepa.toml")
    _git(
        validation_repo, "commit", "--allow-empty", "-m", "Configure private validation"
    )
    started = _start(size=3, budget=14)
    run_id = str(started["run_id"])
    (validation_repo / "out_case-2.txt").write_text("b\n")
    _git(validation_repo, "add", "out_case-2.txt")
    _git(validation_repo, "commit", "-m", "Improve training candidate")
    monkeypatch.setenv("GEPA_TEST_FAIL_VALIDATION", "1")
    failed = scored_continue("run", "continue", "--run-id", run_id)
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


def _configure_private_validation(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    directory = project.parent / f"{project.name}-heldout"
    directory.mkdir()
    path = directory / "recovery-heldout.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "name": f"secret-paired-{i}",
                    "inputs": "PRIVATE",
                    "expected_output": "v",
                }
            )
            + "\n"
            for i in range(2)
        )
    )
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(path))
    _git(project, "add", ".gepa/gepa.toml")
    _git(
        project, "commit", "--allow-empty", "-m", "Configure withheld recovery evidence"
    )
    return path


@pytest.mark.parametrize("death_sample", [4, 5])
def test_extra_noise_training_samples_replay_without_new_paid_rows(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch, death_sample: int
) -> None:
    from pydantic_ai_gepa.cli import eval as eval_module
    from pydantic_ai_gepa.evaluation import EvaluationRecord

    calls = []
    baseline_samples = [0.4, 0.6, 0.5, 0.4, 0.6]
    samples = iter([0.5, *baseline_samples, *baseline_samples])

    async def noisy_training(**kwargs):
        score = next(samples)
        calls.append(score)
        return [
            EvaluationRecord(case.name, score, None, {}) for case in kwargs["dataset"]
        ]

    monkeypatch.setattr(eval_module, "evaluate_callable_dataset", noisy_training)
    started = _start("--acceptance-max-repetitions", "5", repetitions=3, budget=11)
    run_id = str(started["run_id"])
    assert started["reflection_baseline_samples"] == baseline_samples
    assert len(calls) == 6  # Selected failure plus five fresh baselines.
    _commit_score(git_repo, "noisy-proposal")
    original = run_module.run_eval_once
    candidate_calls = []

    def die_in_extra_look(**kwargs):
        outcome = original(**kwargs)
        candidate_calls.append(outcome.summary["eval_id"])
        if len(candidate_calls) == death_sample:
            raise RuntimeError("died in an extra noise acceptance look")
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", die_in_extra_look)
    interrupted = _run("run", "continue", "--run-id", run_id)
    assert isinstance(interrupted.exception, RuntimeError)
    before = list(ParetoLog(run_id).iter_rows())
    resumed = _run("run", "continue", "--run-id", run_id)
    assert resumed.exit_code == 0, resumed.output
    comparison = _run_payload(resumed.output)["last_comparison"]
    assert comparison["verdict"] == "inconclusive"
    assert comparison["candidate_samples"] == baseline_samples
    assert comparison["max_looks"] == 3
    assert len(candidate_calls) == 5
    assert len(calls) == 11
    after = ParetoLog(run_id).iter_rows()
    assert after[: len(before)] == before
    assert len(after) == 11


@pytest.mark.parametrize("death_sample", [1, 3, 4, 5])
def test_validation_confirmation_replays_initial_and_extra_samples(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch, death_sample: int
) -> None:
    from pydantic_ai_gepa.cli import eval as eval_module
    from pydantic_ai_gepa.evaluation import EvaluationRecord

    validation_path = _configure_private_validation(git_repo, monkeypatch)
    candidate_samples = [0.2, 0.8, 0.5, 0.2, 0.8]
    calls = {"training": [], "validation": []}
    proposed = False

    async def noisy_evaluator(**kwargs):
        role = (
            "validation"
            if kwargs["dataset"][0].name.startswith("secret-")
            else "training"
        )
        if role == "training":
            score = 0.8 if proposed else 0.1
        else:
            score = candidate_samples[len(calls[role]) - 3] if proposed else 0.5
        calls[role].append(score)
        return [
            EvaluationRecord(case.name, score, None, {}) for case in kwargs["dataset"]
        ]

    monkeypatch.setattr(eval_module, "evaluate_callable_dataset", noisy_evaluator)
    started = _start("--acceptance-max-repetitions", "5", repetitions=3, budget=17)
    run_id = str(started["run_id"])
    assert started["iterations"] == 9
    _commit_score(git_repo, "noisy-validation-proposal")
    proposed = True
    original = run_module.run_eval_once
    validation_calls = []

    def die_during_confirmation(**kwargs):
        outcome = original(**kwargs)
        if kwargs.get("dataset_role") == "validation":
            validation_calls.append(outcome.summary["eval_id"])
            if len(validation_calls) == death_sample:
                raise RuntimeError("died during validation confirmation")
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", die_during_confirmation)
    interrupted = scored_continue("run", "continue", "--run-id", run_id)
    assert interrupted.exit_code == 1
    before = ParetoLog(run_id).iter_rows()
    packet = _resume(run_id)
    resumed = scored_continue(*packet["next_command"]["argv"][1:])
    assert resumed.exit_code == 0, resumed.output
    comparison = _run_payload(resumed.output)["last_comparison"]
    assert comparison["training_verdict"] == "accepted"
    assert comparison["validation_comparison"]["verdict"] == "inconclusive"
    assert comparison["validation_comparison"]["candidate_samples"] == candidate_samples
    assert comparison["validation_comparison"]["max_looks"] == 3
    assert len(validation_calls) == 5
    assert calls["validation"] == [0.5] * 3 + candidate_samples
    assert calls["training"] == [0.1] * 6 + [0.8] * 3
    after = ParetoLog(run_id).iter_rows()
    assert after[: len(before)] == before
    assert len(after) == 17
    packet = _packet(git_repo, run_id)
    assert packet["last_comparison"]["candidate_samples"] == [0.8] * 3
    assert packet["last_comparison"]["baseline_samples"] == [0.1] * 3
    assert packet["last_comparison"]["validation"]["mean"] == pytest.approx(0.5)
    assert str(validation_path) not in json.dumps(packet)
    assert "secret-paired" not in json.dumps(packet)


@pytest.mark.parametrize("death_phase", ["training", "validation", "final_save"])
def test_paired_validation_recovers_private_evidence_and_incumbent(
    validation_repo: Path, monkeypatch: pytest.MonkeyPatch, death_phase: str
) -> None:
    from pydantic_ai_gepa.cli import eval as eval_module
    from pydantic_ai_gepa.evaluation import EvaluationRecord

    validation_path = _configure_private_validation(validation_repo, monkeypatch)
    proposed = False
    calls = []

    async def paired_evaluator(**kwargs):
        private = kwargs["dataset"][0].name.startswith("secret-")
        calls.append("validation" if private else "training")
        return [
            EvaluationRecord(
                case.name,
                (0.7 + index * 0.11 if proposed else 0.2 + index * 0.1),
                None,
                {},
            )
            for index, case in enumerate(kwargs["dataset"])
        ]

    monkeypatch.setattr(eval_module, "evaluate_callable_dataset", paired_evaluator)
    started = _start("--acceptance-paired-min-cases", "2", size=3, budget=5)
    run_id = str(started["run_id"])
    assert calls == ["validation", "training", "training"]
    incumbent = dict(
        run_module._load_state(run_id)
        .restore_validation_evidence()
        .best_validation_per_case_scores
    )
    assert len(incumbent) == 2
    candidate = _commit_score(validation_repo, "paired-proposal")
    proposed = True
    original = run_module.run_eval_once
    original_replace = os.replace
    killed = False

    def die_after_row(**kwargs):
        nonlocal killed
        outcome = original(**kwargs)
        if not killed and kwargs.get("dataset_role", "training") == death_phase:
            killed = True
            raise RuntimeError("died after paid paired evaluation")
        return outcome

    def die_before_final_state(source, destination, **kwargs):
        nonlocal killed
        if (
            not killed
            and death_phase == "final_save"
            and Path(destination).name.endswith(".record.json")
            and json.loads(
                json.loads(Path(source).read_text())["files"].get("state.json", "{}")
            ).get("status")
            == "done"
        ):
            killed = True
            raise RuntimeError("died after replacing private incumbent evidence")
        return original_replace(source, destination, **kwargs)

    monkeypatch.setattr(run_module, "run_eval_once", die_after_row)
    monkeypatch.setattr(os, "replace", die_before_final_state)
    interrupted = scored_continue("run", "continue", "--run-id", run_id)
    assert interrupted.exit_code == 1, interrupted.output
    assert (
        run_module._load_state(run_id)
        .restore_validation_evidence()
        .best_validation_per_case_scores
        == incumbent
    )
    before = ParetoLog(run_id).iter_rows()
    packet = _resume(run_id)
    resumed = scored_continue(*packet["next_command"]["argv"][1:])
    assert resumed.exit_code == 0, resumed.output
    payload = _run_payload(resumed.output)
    assert payload["last_comparison"]["validation_comparison"]["method"] == "paired_t"
    assert payload["last_comparison"]["verdict"] == "accepted"
    assert payload["best_candidate_id"] == candidate[:12]
    assert calls == ["validation", "training", "training", "training", "validation"]
    after = ParetoLog(run_id).iter_rows()
    assert after[: len(before)] == before
    assert len(after) == 5
    state = run_module._load_state(run_id).restore_validation_evidence()
    assert state.best_validation_samples == (pytest.approx(0.755),)
    assert state.best_validation_per_case_scores == {
        "secret-paired-0": pytest.approx(0.7),
        "secret-paired-1": pytest.approx(0.81),
    }
    packet = _packet(validation_repo, run_id)
    assert "secret-paired" not in json.dumps(packet)
    assert str(validation_path) not in json.dumps(packet)
    public_files = validation_repo / ".gepa" / "runs" / run_id
    assert all(
        "secret-paired" not in path.read_text()
        for path in public_files.rglob("*")
        if path.is_file()
    )


def test_interrupted_validation_seed_retry_reuses_paid_incumbent_samples(
    validation_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_private_validation(validation_repo, monkeypatch)
    monkeypatch.setenv("GEPA_TEST_FAIL_VALIDATION", "1")
    started = _start(size=3, budget=14)
    run_id = str(started["run_id"])
    assert started["status"] == "paused_after_infrastructure_error"
    assert started["iterations"] == 1
    monkeypatch.delenv("GEPA_TEST_FAIL_VALIDATION")
    original = run_module.run_eval_once
    validation_calls = []

    def die_during_incumbent_sampling(**kwargs):
        outcome = original(**kwargs)
        if kwargs.get("dataset_role") == "validation":
            validation_calls.append(outcome.summary["eval_id"])
            if len(validation_calls) == 2:
                raise RuntimeError("died while collecting the incumbent confirmation")
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", die_during_incumbent_sampling)
    interrupted = scored_continue("run", "continue", "--run-id", run_id)
    assert interrupted.exit_code == 1
    before = ParetoLog(run_id).iter_rows()
    assert len(before) == 3
    resumed = scored_continue("run", "continue", "--run-id", run_id)
    assert resumed.exit_code == 0, resumed.output
    payload = _run_payload(resumed.output)
    assert payload["status"] == "paused_for_reflection"
    assert (
        payload["validation_evaluations"] == 4
    )  # Failed seed plus three fresh samples.
    assert len(payload["best_validation_samples"]) == 3
    assert len(validation_calls) == 3
    after = ParetoLog(run_id).iter_rows()
    assert after[: len(before)] == before
    assert len(after) == 8  # Four validation rows, selected failure, three baselines.


@pytest.mark.parametrize("restore_original", [False, True])
@pytest.mark.parametrize("budget", [7, 14])
def test_candidate_change_releases_checkpoint_and_keeps_budget(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch, restore_original: bool, budget: int
) -> None:
    started = _start(repetitions=3, budget=budget)
    run_id = str(started["run_id"])
    original_candidate = _commit_score(git_repo, "candidate-x")
    original_eval = run_module.run_eval_once
    calls = []

    def change_candidate_after_first_sample(**kwargs):
        outcome = original_eval(**kwargs)
        calls.append(outcome.summary["candidate_id"])
        if len(calls) == 1:
            _commit_score(git_repo, "candidate-y")
        return outcome

    monkeypatch.setattr(
        run_module, "run_eval_once", change_candidate_after_first_sample
    )
    failed = _run("run", "continue", "--run-id", run_id)
    assert failed.exit_code == 1, failed.output
    assert len(calls) == 2 and calls[0] != calls[1]
    before = ParetoLog(run_id).iter_rows()
    assert len(before) == 6
    assert run_module._load_state(run_id).continuation is None
    assert run_module._load_state(run_id).iterations == 6
    if restore_original:
        _git(git_repo, "reset", "--hard", original_candidate)
    packet = _resume(run_id)
    assert "pending_continuation" not in packet
    assert packet["budget"]["used"] == 6
    monkeypatch.setattr(run_module, "run_eval_once", original_eval)
    retried = _run("run", "continue", "--run-id", run_id)
    # The review's exact budget=7 leaves only one row, below the three-sample
    # minimum. It must report the budget cap, not a wedged checkpoint.
    assert retried.exit_code == (70 if budget == 7 else 0), retried.output
    assert "Restore candidate" not in retried.output
    assert "another candidate" not in retried.output
    assert run_module._load_state(run_id).continuation is None
    after = ParetoLog(run_id).iter_rows()
    assert after[:6] == before
    assert len(after) == (6 if budget == 7 else 9)


def test_foreign_paid_row_ends_prefix_and_repeated_recovery_stays_reusable(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = _start(repetitions=3, budget=14)
    run_id = str(started["run_id"])
    candidate_x = _commit_score(git_repo, "candidate-x")
    original_eval = run_module.run_eval_once
    calls = []

    def die_before_candidate_change_is_detected(**kwargs):
        outcome = original_eval(**kwargs)
        calls.append(outcome.summary["candidate_id"])
        if len(calls) == 1:
            _commit_score(git_repo, "candidate-y")
        else:
            raise RuntimeError("died before controller observed the foreign row")
        return outcome

    monkeypatch.setattr(
        run_module, "run_eval_once", die_before_candidate_change_is_detected
    )
    failed = _run("run", "continue", "--run-id", run_id)
    assert isinstance(failed.exception, RuntimeError)
    assert ParetoLog(run_id).count_budget_rows() == 6
    _git(git_repo, "reset", "--hard", candidate_x)
    before = ParetoLog(run_id).iter_rows()
    new_calls = []

    def die_once_more(**kwargs):
        outcome = original_eval(**kwargs)
        new_calls.append(outcome.summary["eval_id"])
        if len(new_calls) == 1:
            raise RuntimeError("died after extending the reusable prefix")
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", die_once_more)
    assert isinstance(
        _run("run", "continue", "--run-id", run_id).exception, RuntimeError
    )
    resumed = _run("run", "continue", "--run-id", run_id)
    assert resumed.exit_code == 0, resumed.output
    comparison = _run_payload(resumed.output)["last_comparison"]
    assert comparison["candidate_sample_count"] == 3
    assert comparison["candidate_iteration"] == 6  # Includes the discarded row's cost.
    assert len(comparison["candidate_report_paths"]) == 3
    assert len(comparison["candidate_trace_paths"]) == 3
    assert all(Path(path).exists() for path in comparison["candidate_report_paths"])
    assert len(new_calls) == 2
    assert ParetoLog(run_id).iter_rows()[:6] == before
    assert ParetoLog(run_id).count_budget_rows() == 8
    assert run_module._load_state(run_id).iterations == 8


def test_components_can_abandon_an_unrestorable_continuation(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = _start(size=2, budget=11)
    run_id = str(started["run_id"])
    component = next((repo / ".gepa" / "components").glob("*.md"))
    component.write_text("Candidate X instructions")
    original_eval = run_module.run_eval_once

    def die_after_sample(**kwargs):
        original_eval(**kwargs)
        raise RuntimeError("lost reflector")

    monkeypatch.setattr(run_module, "run_eval_once", die_after_sample)
    assert isinstance(
        _run("run", "continue", "--run-id", run_id).exception, RuntimeError
    )
    component.write_text("Candidate Y overwrote X without a backup")
    packet = _resume(run_id)
    assert "--abandon-continuation" in packet["instructions"]
    assert _run("run", "continue", "--run-id", run_id).exit_code == 2
    before = ParetoLog(run_id).iter_rows()
    packet = _resume(
        run_id, "--abandon-continuation", "--reason", "components overwritten"
    )
    assert "pending_continuation" not in packet
    assert packet["budget"]["used"] == 5
    assert any(
        row.get("kind") == "continuation_abandoned"
        and row["reason"] == "components overwritten"
        for row in packet["journal_tail"]
    )
    assert component.read_text() == "Candidate Y overwrote X without a backup"
    assert ParetoLog(run_id).iter_rows() == before
    monkeypatch.setattr(run_module, "run_eval_once", original_eval)
    continued = _run("run", "continue", "--run-id", run_id)
    assert continued.exit_code == 0, continued.output
    assert ParetoLog(run_id).count_budget_rows() == 8


@pytest.mark.parametrize("paired_threshold", [None, 100])
def test_nonpaired_validation_requires_writable_private_front(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch, paired_threshold: int | None
) -> None:
    from pydantic_ai_gepa.cli import front
    from pydantic_ai_gepa.cli import eval as eval_module

    validation_path = _configure_private_validation(git_repo, monkeypatch)
    calls = []

    async def evaluate(**kwargs):
        calls.append(kwargs)
        pytest.fail("Storage must be checked before any rollout is paid")

    monkeypatch.setattr(eval_module, "evaluate_callable_dataset", evaluate)

    def readonly_evidence(*args, **kwargs):
        raise PermissionError(f"{validation_path} is read-only")

    monkeypatch.setattr(front, "_write_front", readonly_evidence)
    options = (
        ()
        if paired_threshold is None
        else ("--acceptance-paired-min-cases", str(paired_threshold))
    )
    started = _run("run", "start", *options, "--max-iterations", "13")
    assert started.exit_code == 2, started.output
    assert "Private parent front storage must be writable" in started.output
    assert str(validation_path) not in started.output
    assert not calls
    assert not list((git_repo / ".gepa/runs").glob("*/pareto.jsonl"))
    assert not (validation_path.parent / ".gepa-validation-evidence").exists()


def test_lane_validation_does_not_create_continuation_snapshots(
    validation_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic_ai_gepa.cli import validation

    validation_path = _configure_private_validation(validation_repo, monkeypatch)
    started = _start("--acceptance-paired-min-cases", "2", size=3, budget=5)
    run_id = str(started["run_id"])
    state = run_module._load_state(run_id)
    directory = validation_path.parent / ".gepa-validation-evidence"
    before = list(directory.glob("*.json"))
    monkeypatch.setattr(
        validation,
        "write_validation_evidence",
        lambda *args, **kwargs: pytest.fail(
            "lane evals do not use continuation replay"
        ),
    )
    _, outcome = run_module._evaluate_validation_candidate(state, lane="lane-1")
    assert outcome.summary["lane"] == "lane-1"
    assert list(directory.glob("*.json")) == before


@pytest.mark.parametrize("abandon", [False, True])
def test_private_replay_snapshots_removed_after_completion_or_abandonment(
    validation_repo: Path, monkeypatch: pytest.MonkeyPatch, abandon: bool
) -> None:
    from pydantic_ai_gepa.cli import eval as eval_module
    from pydantic_ai_gepa.evaluation import EvaluationRecord

    validation_path = _configure_private_validation(validation_repo, monkeypatch)
    proposed = False

    async def paired_evaluator(**kwargs):
        return [
            EvaluationRecord(
                case.name, 0.7 + index * 0.01 if proposed else 0.2, None, {}
            )
            for index, case in enumerate(kwargs["dataset"])
        ]

    monkeypatch.setattr(eval_module, "evaluate_callable_dataset", paired_evaluator)
    started = _start("--acceptance-paired-min-cases", "2", size=3, budget=5)
    run_id = str(started["run_id"])
    directory = validation_path.parent / ".gepa-validation-evidence"
    incumbent_files = set(directory.glob("*.json"))
    assert len(incumbent_files) == 2  # Incumbent and persistent parent front.
    proposed = True
    candidate = _commit_score(validation_repo, "paired proposal")
    original_eval = run_module.run_eval_once

    def die_after_validation(**kwargs):
        outcome = original_eval(**kwargs)
        if kwargs.get("dataset_role") == "validation":
            raise RuntimeError("died after saving paired evidence")
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", die_after_validation)
    assert scored_continue("run", "continue", "--run-id", run_id).exit_code == 1
    assert (
        len(list(directory.glob("*.json"))) == 4
    )  # Incumbent, persistent front, backup, candidate eval.
    before = ParetoLog(run_id).iter_rows()
    monkeypatch.setattr(
        run_module, "run_eval_once", lambda **kwargs: pytest.fail("all samples paid")
    )
    if abandon:
        packet = _resume(run_id, "--abandon-continuation")
        assert packet["budget"]["used"] == 5
        assert _git(validation_repo, "rev-parse", "HEAD") == candidate
    else:
        completed = scored_continue("run", "continue", "--run-id", run_id)
        assert completed.exit_code == 0, completed.output
        assert (
            _run_payload(completed.output)["last_comparison"]["verdict"] == "accepted"
        )
    assert ParetoLog(run_id).iter_rows() == before
    assert set(directory.glob("*.json")) == incumbent_files
    assert run_module._load_state(run_id).continuation is None


def test_packet_and_continue_agree_when_rejected_candidate_returns_on_new_batch(
    validation_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = _start(size=3, budget=17)
    run_id = str(started["run_id"])
    baseline = _git(validation_repo, "rev-parse", "HEAD")
    (validation_repo / "out_case-1.txt").write_text("wrong")
    _git(validation_repo, "add", "out_case-1.txt")
    _git(validation_repo, "commit", "-m", "Rejected candidate B")
    candidate = _git(validation_repo, "rev-parse", "HEAD")
    rejected = _run("run", "continue", "--run-id", run_id)
    assert _run_payload(rejected.output)["last_comparison"]["verdict"] == "rejected"
    _git(validation_repo, "reset", "--hard", baseline)
    advanced = _run("run", "continue", "--run-id", run_id)
    assert advanced.exit_code == 0, advanced.output
    minibatch = _run_payload(advanced.output)["reflection_minibatch_id"]
    assert minibatch != started["reflection_minibatch_id"]
    _git(validation_repo, "reset", "--hard", candidate)
    packet = _resume(run_id)
    assert packet["current_tree"]["already_scored"] is False
    assert "not scored" in packet["instructions"]
    assert packet["journal_tail"][-1]["unscored_candidate_commit"] == candidate
    before = ParetoLog(run_id).count_budget_rows()
    continued = _run("run", "continue", "--run-id", run_id)
    assert continued.exit_code == 0, continued.output
    assert (
        _run_payload(continued.output)["last_comparison"]["minibatch_id"] == minibatch
    )
    assert ParetoLog(run_id).count_budget_rows() == before + 3


def test_torn_journal_does_not_break_pause_or_resume(git_repo: Path) -> None:
    started = _start(budget=7)
    run_id = str(started["run_id"])
    _commit_score(git_repo, "good")
    journal = git_repo / ".gepa" / "journal.jsonl"
    journal.write_text('{"content": "valid note"}\nnot-json\n42\n{"torn":')
    continued = _run("run", "continue", "--run-id", run_id)
    assert continued.exit_code == 0, continued.output
    assert _packet(git_repo, run_id)["journal_tail"] == [{"content": "valid note"}]
    packet = _resume(run_id)
    assert packet["status"] == "done"
    assert packet["journal_tail"][0] == {"content": "valid note"}
    assert packet["journal_tail"][-1]["kind"] == "reflector_lost"


@pytest.mark.parametrize("failure", ["write", "notes"])
def test_packet_failure_warns_after_committed_state_and_resume_regenerates(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from pydantic_ai_gepa.cli import reflector

    started = _start(budget=7)
    run_id = str(started["run_id"])
    _commit_score(git_repo, "good")
    owner, name = (
        (run_module, "write_packet")
        if failure == "write"
        else (reflector, "notes_index")
    )
    original = getattr(owner, name)

    def broken_packet(*args, **kwargs):
        raise OSError("packet unavailable")

    monkeypatch.setattr(owner, name, broken_packet)
    continued = _run("run", "continue", "--run-id", run_id)
    assert continued.exit_code == 0, continued.output
    assert "Warning: state saved" in continued.output
    state = run_module._load_state(run_id)
    assert state.status == "done" and state.continuation is None
    monkeypatch.setattr(owner, name, original)
    before = ParetoLog(run_id).iter_rows()
    packet = _resume(run_id)
    assert packet["status"] == "done"
    assert packet["last_comparison"]["verdict"] == "accepted"
    assert ParetoLog(run_id).iter_rows() == before

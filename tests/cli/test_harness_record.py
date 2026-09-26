"""Public edits cannot change the harness's scoring, promotion, or reporting."""

import json
import shutil

import pytest

from pydantic_ai_gepa.cli import run as run_module
from tests.cli import test_harness_scoring as harness
from tests.cli.test_git_candidate_cli import _git, _run, _run_payload

# The shared injected backend runs only fake local evaluators (no model/network).
git_repo = harness.git_repo
isolated_environment = harness.isolated_environment


@pytest.fixture
def prepared(git_repo, monkeypatch, tmp_path):
    (git_repo / "task_pkg/evaluation.py").write_text(
        "from pathlib import Path\nasync def evaluate(case):\n"
        '    return Path("score.txt").read_text().strip()\n'
    )
    training = git_repo / ".gepa/dataset.jsonl"
    row = json.loads(training.read_text())
    training.write_text(
        "".join(json.dumps(dict(row, name=f"case-{i}")) + "\n" for i in range(10))
    )
    _git(git_repo, "add", "task_pkg", ".gepa/dataset.jsonl")
    _git(git_repo, "commit", "-m", "Fake evaluator")
    private = tmp_path.parent / f"private-{tmp_path.name}"
    private.mkdir()
    dataset = private / "heldout.jsonl"
    dataset.write_text(
        "".join(
            json.dumps(
                {
                    "name": f"PRIVATE_CASE_{i}",
                    "inputs": "PRIVATE_INPUT",
                    "expected_output": "bad",
                }
            )
            + "\n"
            for i in range(10)
        )
    )
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(dataset))
        started = _run(
            "run",
            "start",
            "--size",
            "10",
            "--max-iterations",
            "5",
            "--acceptance-paired-min-cases",
            "2",
            "--acceptance-repetitions",
            "1",
            "--acceptance-max-repetitions",
            "1",
        )
    assert started.exit_code == 0, started.output
    seed = _run_payload(started.output)
    run_id = str(seed["run_id"])
    directory = git_repo / ".gepa/runs" / run_id
    harness._commit(git_repo)
    nominated = harness._continue(run_id)
    assert nominated.exit_code == 0, nominated.output
    nomination_id = json.loads(nominated.output)["nomination_id"]
    return dataset, directory, seed, nomination_id


def _observed(dataset, directory, monkeypatch):
    run_id = directory.name
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(dataset))
        state = run_module._load_state(run_id)
        _, report = run_module._write_final_report(state)
        status = _run("run", "status", "--run-id", run_id)
        pareto = _run("pareto", "--run-id", run_id)
        front = _run("pareto", "--run-id", run_id, "--front")
    assert status.exit_code == pareto.exit_code == front.exit_code == 0
    payload = _run_payload(status.stdout)

    # Timestamps/eval IDs differ between independent evaluations, decisions do not.
    def history(rows):
        return [
            (r["candidate_id"], r["mean_score"], r["extra"]["dataset_role"])
            for r in rows
        ]

    decision = {
        key: payload[key]
        for key in (
            "best_candidate_id",
            "best_commit_sha",
            "best_mean_score",
            "iterations",
            "max_iterations",
            "status",
            "validation_evaluations",
            "spend",
        )
    }
    report_decisions = [
        line
        for line in report.splitlines()
        if line.startswith(
            (
                "- best_",
                "- accepted_best_",
                "- iterations:",
                "- status:",
                "- validation_evaluations:",
            )
        )
    ]
    return (
        decision,
        history(json.loads(pareto.stdout)),
        history(json.loads(front.stdout)),
        report_decisions,
    )


def _attack(name, directory, seed, nomination_id):
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text())
    if name == "incumbent":
        state.update(
            best_candidate_id="forged",
            best_commit_sha="a" * 40,
            reflection_baseline_candidate_id="forged",
            reflection_baseline_commit_sha="a" * 40,
            heldout_required=False,
        )
    elif name == "aggregate":
        state.update(best_mean_score=999.0, best_validation_samples=[999.0])
        rows = [
            json.loads(line)
            for line in (directory / "pareto.jsonl").read_text().splitlines()
        ]
        rows[0].update(mean_score=999.0, candidate_id="forged")
        (directory / "pareto.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    elif name == "result":
        path = directory / "results" / f"{nomination_id}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            json.dumps({"stdout": "forged accepted", "stderr": "", "exit_code": 0})
        )
        state["last_reflector_comparison"] = {"verdict": "accepted", "improved": True}
    elif name == "checkpoint":
        state["continuation"] = {
            "candidate_id": "forged",
            "ledger_offset": 500,
            "gate_case": [],
            "gates": [{"verdict": "accepted"}],
            "validation_done": True,
        }
    elif name == "budget":
        state.update(
            iterations=0,
            max_iterations=999,
            max_token_cost=999,
            validation_evaluations=0,
            gate_consumed_iterations=0,
        )
        (directory / "spend.jsonl").write_text("")
        (directory / "spend-reservations.json").write_text("{}")
    elif name == "delete":
        for path in directory.rglob("*"):
            if (
                path.is_file()
                and "nominations" not in path.parts
                and not path.name.endswith(".lock")
            ):
                path.unlink()
        return
    elif name == "malformed":
        state_path.write_text("{malformed")
        return
    state_path.write_text(json.dumps(state))


@pytest.mark.parametrize(
    "attack",
    ["incumbent", "aggregate", "result", "checkpoint", "budget", "delete", "malformed"],
)
def test_public_attack_matches_control_for_scores_promotions_and_reports(
    prepared, monkeypatch, tmp_path, attack
):
    dataset, directory, seed, nomination_id = prepared
    # Fork an identical fixture on disk, then run the real queue twice. Only the
    # test harness has private access to restore the control's initial record.
    backup = tmp_path.parent / f"backup-{tmp_path.name}"
    shutil.copytree(dataset.parent, backup / "private")
    shutil.copytree(directory, backup / "run")
    calls = []
    original = run_module.run_eval_once

    def evaluate(**kwargs):
        state = run_module._load_state(directory.name)
        calls.append(
            (
                kwargs.get("dataset_role", "training"),
                kwargs.get("minibatch_id"),
                state.best_candidate_id,
                state.reflection_baseline_candidate_id,
            )
        )
        return original(**kwargs)

    monkeypatch.setattr(run_module, "run_eval_once", evaluate)
    control = harness._serve(directory.name, dataset, monkeypatch)
    assert control.exit_code == 0, control.output
    expected_calls = list(calls)
    candidate_sha = _git(directory.parents[2], "rev-parse", "HEAD")
    expected = _observed(dataset, directory, monkeypatch)
    assert (
        expected_calls and expected[0]["best_candidate_id"] == seed["best_candidate_id"]
    )
    assert expected[0]["status"] == "done"
    shutil.rmtree(dataset.parent)
    shutil.copytree(backup / "private", dataset.parent)
    shutil.rmtree(directory)
    shutil.copytree(backup / "run", directory)
    calls.clear()
    _attack(attack, directory, seed, nomination_id)
    served = harness._serve(directory.name, dataset, monkeypatch)
    assert served.exit_code == 0, served.output
    assert "restored from the harness record" in served.output
    assert calls == expected_calls
    assert _git(directory.parents[2], "rev-parse", "HEAD") == candidate_sha
    assert _observed(dataset, directory, monkeypatch) == expected
    # Results are byte-compatible and the legitimate reflector can consume them.
    delivered = harness._continue(directory.name)
    assert delivered.exit_code == 0, delivered.output
    assert (
        _run_payload(delivered.output)["best_candidate_id"] == seed["best_candidate_id"]
    )
    harness._sweep(directory.parents[2], dataset, served.output, delivered.output)
    public = "\n".join(
        path.read_text(errors="replace")
        for path in directory.rglob("*")
        if path.is_file()
    )
    assert "PRIVATE_CASE" not in public and "PRIVATE_INPUT" not in public


def test_completed_result_and_old_state_cannot_rescore_or_rollback(
    prepared, monkeypatch
):
    dataset, directory, seed, nomination_id = prepared
    old = (directory / "state.json").read_text()
    assert harness._serve(directory.name, dataset, monkeypatch).exit_code == 0
    expected = _observed(dataset, directory, monkeypatch)
    (directory / "state.json").write_text(old)
    result = directory / "results" / f"{nomination_id}.json"
    result.write_text('{"stdout": "accepted forged", "exit_code": 0}')
    monkeypatch.setattr(
        run_module,
        "run_eval_once",
        lambda **_: pytest.fail("completed request must not score again"),
    )
    served = harness._serve(directory.name, dataset, monkeypatch)
    assert served.exit_code == 0, served.output
    assert "restored from the harness record" in served.output
    assert _observed(dataset, directory, monkeypatch) == expected
    assert "forged" not in result.read_text()
    assert expected[0]["best_candidate_id"] == seed["best_candidate_id"]


@pytest.mark.parametrize("damage", ["missing", "malformed"])
def test_private_record_unavailable_fails_closed(prepared, monkeypatch, damage):
    dataset, directory, seed, _ = prepared
    record = next((dataset.parent / ".gepa-heldout").glob("*.record.json"))
    assert record.stat().st_mode & 0o777 == 0o600
    assert record.parent.stat().st_mode & 0o777 == 0o700
    if damage == "missing":
        record.unlink()
    else:
        record.write_text("{}")
    monkeypatch.setattr(
        run_module, "run_eval_once", lambda **_: pytest.fail("must fail before scoring")
    )
    served = harness._serve(directory.name, dataset, monkeypatch)
    assert served.exit_code == 2
    assert "private record missing or unreadable" in served.output
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(dataset))
        for command in [
            ("run", "status"),
            ("pareto",),
            ("eval", "--dataset-role", "validation"),
        ]:
            result = _run(*command, "--run-id", directory.name)
            assert result.exit_code == 2
            assert "private record missing or unreadable" in result.output
    assert not (directory / "final_report.md").exists()
    assert (
        json.loads((directory / "state.json").read_text())["best_candidate_id"]
        == seed["best_candidate_id"]
    )


def test_paid_continuation_uses_private_checkpoint_and_prefix(
    prepared, monkeypatch, tmp_path
):
    dataset, directory, seed, nomination_id = prepared
    original = run_module.run_eval_once

    def interrupted(**kwargs):
        result = original(**kwargs)
        if kwargs.get("dataset_role", "training") == "training":
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(run_module, "run_eval_once", interrupted)
    stopped = harness._serve(directory.name, dataset, monkeypatch)
    assert stopped.exit_code != 0
    checkpoint = json.loads((directory / "state.json").read_text())["continuation"]
    assert checkpoint and checkpoint["ledger_offset"] == 3
    assert not (directory / "results" / f"{nomination_id}.json").exists()
    backup = tmp_path.parent / f"checkpoint-backup-{tmp_path.name}"
    shutil.copytree(dataset.parent, backup / "private")
    shutil.copytree(directory, backup / "run")
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs.get("dataset_role", "training"))
        return original(**kwargs)

    monkeypatch.setattr(run_module, "run_eval_once", evaluate)
    assert harness._serve(directory.name, dataset, monkeypatch).exit_code == 0
    expected = _observed(dataset, directory, monkeypatch)
    assert calls == ["validation"]  # The paid training sample is replayed.
    shutil.rmtree(dataset.parent)
    shutil.copytree(backup / "private", dataset.parent)
    shutil.rmtree(directory)
    shutil.copytree(backup / "run", directory)
    _attack("checkpoint", directory, seed, nomination_id)
    (directory / "pareto.jsonl").write_text("")
    calls.clear()
    served = harness._serve(directory.name, dataset, monkeypatch)
    assert served.exit_code == 0, served.output
    assert "restored from the harness record" in served.output
    assert calls == ["validation"]
    assert _observed(dataset, directory, monkeypatch) == expected
    assert expected[0]["best_candidate_id"] == seed["best_candidate_id"]


heldout = harness.heldout


@pytest.mark.parametrize("heldout", [True], indirect=True)
def test_spend_rollback_cannot_buy_more_validation_or_promote(
    git_repo, heldout, monkeypatch, tmp_path
):
    dataset, run_id, _ = heldout
    directory = git_repo / ".gepa/runs" / run_id
    seed = json.loads((directory / "state.json").read_text())
    harness._commit(git_repo)
    nominated = harness._continue(run_id)
    nomination_id = json.loads(nominated.output)["nomination_id"]
    backup = tmp_path.parent / f"spend-backup-{tmp_path.name}"
    shutil.copytree(dataset.parent, backup / "private")
    shutil.copytree(directory, backup / "run")
    calls = []
    original = run_module.run_eval_once

    def evaluate(**kwargs):
        calls.append(kwargs.get("dataset_role", "training"))
        return original(**kwargs)

    monkeypatch.setattr(run_module, "run_eval_once", evaluate)
    assert harness._serve(run_id, dataset, monkeypatch).exit_code == 0
    expected_calls = list(calls)
    candidate_sha = _git(directory.parents[2], "rev-parse", "HEAD")
    expected = _observed(dataset, directory, monkeypatch)
    assert expected[0]["spend"]["stopped_by_cost"]
    assert expected[0]["best_candidate_id"] == seed["best_candidate_id"]
    shutil.rmtree(dataset.parent)
    shutil.copytree(backup / "private", dataset.parent)
    shutil.rmtree(directory)
    shutil.copytree(backup / "run", directory)
    _attack("budget", directory, seed, nomination_id)
    calls.clear()
    served = harness._serve(run_id, dataset, monkeypatch)
    assert served.exit_code == 0, served.output
    assert "restored from the harness record" in served.output
    assert calls == expected_calls
    assert _git(directory.parents[2], "rev-parse", "HEAD") == candidate_sha
    assert _observed(dataset, directory, monkeypatch) == expected
    assert harness._continue(run_id).exit_code == 70

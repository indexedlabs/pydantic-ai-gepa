"""Private per-case parent selection through the real nomination protocol."""

import json

import pytest

from pydantic_ai_gepa.cli import run as run_module
from pydantic_ai_gepa.cli.front import ValidationFront
from pydantic_ai_gepa.cli.validation import harness_environment
from tests.cli import test_harness_scoring as harness
from tests.cli import test_run_cli
from tests.cli.test_git_candidate_cli import _git, _run, _run_payload


git_repo = harness.git_repo
isolated_environment = harness.isolated_environment
repo = test_run_cli.repo


@pytest.fixture
def front_run(git_repo, monkeypatch):
    # The evaluator reads only candidate text; all private scores live in the
    # held-out rows, so no test secret is embedded in reflector-readable code.
    (git_repo / "task_pkg/evaluation.py").write_text("""from pathlib import Path
from pydantic_ai_gepa.types import MetricResult
async def evaluate(case):
    return Path("score.txt").read_text().strip()
async def metric(case, output):
    return MetricResult(score=case.inputs[output])
""")
    config = git_repo / ".gepa/gepa.toml"
    config.write_text('metric = "task_pkg.evaluation:metric"\n' + config.read_text())
    (git_repo / ".gepa/dataset.jsonl").write_text(
        json.dumps(
            {
                "name": "training",
                "inputs": {"bad": 0.1, "B": 0.4, "C": 0.7},
            }
        )
        + "\n"
    )
    _git(git_repo, "add", ".")
    _git(git_repo, "commit", "-m", "Configure fake scoring")
    private = git_repo.parent / f"private-{git_repo.name}"
    private.mkdir()
    path = private / "secret-evaluation-set.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"name": name, "inputs": values})
            for name, values in [
                (
                    "WITHHELD_ALPHA",
                    {"bad": 0.723456789, "B": 0.876543211, "C": 0.123456789},
                ),
                (
                    "WITHHELD_BETA",
                    {"bad": 0.923456789, "B": 0.223456789, "C": 0.023456789},
                ),
            ]
        )
        + "\n"
    )
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(path))
    started = _run(
        "run",
        "start",
        "--size",
        "1",
        "--max-iterations",
        "80",
        "--acceptance-repetitions",
        "3",
        "--acceptance-max-repetitions",
        "3",
        "--seed",
        "0",
    )
    assert started.exit_code == 0, started.output
    payload = _run_payload(started.output)
    assert payload["status"] == "paused_for_reflection", started.output
    monkeypatch.delenv("GEPA_HELDOUT_DATASET")
    return path, str(payload["run_id"]), payload, started.output


def _private_front(repo, path, run_id, monkeypatch):
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(path))
        with harness_environment():
            return ValidationFront(repo, run_id)


def test_non_mean_winner_becomes_parent_without_leaking(
    git_repo, front_run, monkeypatch
):
    path, run_id, seed_state, start_output = front_run
    harness._commit(git_repo, "B")
    candidate_b = _git(git_repo, "rev-parse", "HEAD")[:12]
    assert harness._continue(run_id).exit_code == 0
    served = harness._serve(run_id, path, monkeypatch)
    assert served.exit_code == 0, served.output
    result = harness._continue(run_id)
    assert result.exit_code == 0, result.output
    state = _run_payload(result.output)
    assert state["best_candidate_id"] == seed_state["best_candidate_id"]
    assert state["best_mean_score"] == seed_state["best_mean_score"]
    assert state["last_reflector_comparison"]["validation_improved"] is False
    front = _private_front(git_repo, path, run_id, monkeypatch)
    assert set(front.weights()) == {seed_state["best_candidate_id"], candidate_b}
    # Seed 0's second draw chooses B, despite its losing validation mean.
    selected = (
        state["next_parent_candidate_id"] or state["reflection_baseline_candidate_id"]
    )
    assert selected == candidate_b
    if state["next_parent_candidate_id"]:
        _git(git_repo, "checkout", str(state["next_parent_commit_sha"]))
        assert harness._continue(run_id).exit_code == 0
        assert harness._serve(run_id, path, monkeypatch).exit_code == 0
        result = harness._continue(run_id)
        state = _run_payload(result.output)
    assert state["reflection_baseline_candidate_id"] == selected
    assert state["best_candidate_id"] == seed_state["best_candidate_id"]

    # A dominated, training-improving child is validated but never sampled.
    harness._commit(git_repo, "C")
    candidate_c = _git(git_repo, "rev-parse", "HEAD")[:12]
    assert harness._continue(run_id).exit_code == 0
    assert harness._serve(run_id, path, monkeypatch).exit_code == 0
    rejected = harness._continue(run_id)
    assert rejected.exit_code == 0, rejected.output
    front = _private_front(git_repo, path, run_id, monkeypatch)
    assert candidate_c not in front.weights()
    assert candidate_b in front.weights()
    pending = _run_payload(rejected.output)
    assert pending["next_parent_candidate_id"] == seed_state["best_candidate_id"]
    resumed_output = _run(
        "run", "resume", "--run-id", run_id, "--reflector", "replacement"
    )
    assert resumed_output.exit_code == 0, resumed_output.output
    assert (
        _run_payload(resumed_output.output)["next_parent_candidate_id"]
        == pending["next_parent_candidate_id"]
    )
    packet_path = git_repo / ".gepa/runs" / run_id / "reflector_packet.json"
    packet = json.loads(packet_path.read_text())
    assert packet["next_parent"]["candidate_id"] == pending["next_parent_candidate_id"]
    assert "Restore next parent" in packet["instructions"]
    assert harness._continue(run_id).exit_code == 0
    assert harness._serve(run_id, path, monkeypatch).exit_code == 0
    waiting = harness._continue(run_id)
    assert waiting.exit_code == 0, waiting.output
    waiting_state = _run_payload(waiting.output)
    assert waiting_state["iterations"] == pending["iterations"]
    assert waiting_state["continuation"] is None
    _git(git_repo, "checkout", str(pending["next_parent_commit_sha"]))
    assert harness._continue(run_id).exit_code == 0
    assert harness._serve(run_id, path, monkeypatch).exit_code == 0
    restored = harness._continue(run_id)
    assert restored.exit_code == 0, restored.output
    restored_state = _run_payload(restored.output)
    assert (
        restored_state["reflection_baseline_candidate_id"]
        == pending["next_parent_candidate_id"]
    )
    assert restored_state["next_parent_candidate_id"] is None
    final_path, final_text = run_module._write_final_report(
        run_module.RunState.from_dict(restored_state),
        root=git_repo,
    )
    assert seed_state["best_candidate_id"] in final_text
    assert final_path.is_file()
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(path))
        with harness_environment():
            parents = [front.select(seed=0, round_id=100 + i) for i in range(20)]
    assert candidate_b in {parent["candidate_id"] for parent in parents}
    assert candidate_c not in {parent["candidate_id"] for parent in parents}

    outputs = [
        start_output,
        served.output,
        result.output,
        rejected.output,
        resumed_output.output,
        restored.output,
        final_text,
    ]
    for command in [("run", "status"), ("run", "resume")]:
        output = _run(*command, "--run-id", run_id)
        assert output.exit_code == 0, output.output
        outputs.append(output.output)
    for format_ in ("json", "tsv"):
        for mode in ("--front", "--all"):
            for frontier in ("instance", "objective", "hybrid", "cartesian"):
                output = _run(
                    "pareto",
                    "--run-id",
                    run_id,
                    "--format",
                    format_,
                    mode,
                    "--frontier",
                    frontier,
                )
                assert output.exit_code == (
                    2 if mode == "--front" and frontier != "instance" else 0
                ), output.output
                outputs.append(output.output)
    outputs.append(_git(git_repo, "branch", "--list"))
    harness._sweep(git_repo, path, *outputs)
    public = "\n".join(
        p.read_text(errors="replace")
        for p in (git_repo / ".gepa").rglob("*")
        if p.is_file()
    )
    for secret in (
        "0.923456789",
        "0.223456789",
        "0.123456789",
        "0.023456789",
        "case_win",
        '"weights"',
        '"selections"',
    ):
        assert secret not in public

    # A reflector switch neither redraws an already selected parent nor changes
    # the next draw. Re-open private storage to simulate a new harness process.
    resumed = _private_front(git_repo, path, run_id, monkeypatch)
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(path))
        with harness_environment():
            assert [
                resumed.select(seed=0, round_id=100 + i) for i in range(20)
            ] == parents
            assert resumed.select(seed=0, round_id=200) == front.select(
                seed=0, round_id=200
            )


def test_front_ignores_invalid_evidence_and_deduplicates(
    git_repo, front_run, monkeypatch
):
    path, run_id, _, _ = front_run
    front = _private_front(git_repo, path, run_id, monkeypatch)
    before = json.dumps(front.data, sort_keys=True)
    from pydantic_ai_gepa.cli.eval import EvalOutcome

    invalid = EvalOutcome([], {"selectable": False}, None, None)
    front.record(invalid)
    from pydantic_ai_gepa.evaluation import EvaluationRecord

    eval_id, saved = next(iter(front.data["evaluations"].items()))
    duplicate = EvalOutcome(
        [EvaluationRecord(case, -1.0, None, {}) for case in saved["scores"]],
        {"eval_id": eval_id, "candidate_id": saved["candidate_id"]},
        None,
        None,
    )
    front.record(duplicate)
    incomplete = EvalOutcome(
        [EvaluationRecord("incomplete", 1.0, None, {})],
        {"eval_id": "incomplete", "candidate_id": "incomplete"},
        None,
        None,
    )
    front.record(incomplete)
    assert json.dumps(front.data, sort_keys=True) == before


def test_component_parent_snapshot_is_restorable(repo, monkeypatch):
    from dataclasses import replace
    import shlex
    from pydantic_ai_gepa.cli.front import parent_restore_command
    from pydantic_ai_gepa.cli.store import ComponentStore

    path = repo.parent / f"{repo.name}-heldout.jsonl"
    path.write_text(
        json.dumps({"name": "private-case", "inputs": "?", "expected_output": "Paris"})
        + "\n"
    )
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(path))
    started = _run("run", "start", "--max-iterations", "8", "--size", "2")
    assert started.exit_code == 0, started.output
    state = _run_payload(started.output)
    parent = state["best_candidate_id"]
    snapshot = repo / ".gepa/runs" / str(state["run_id"]) / "parents" / f"{parent}.json"
    saved = json.loads(snapshot.read_text())
    slot = next(iter(saved["components"]))
    ComponentStore().write(slot, "Changed instructions")
    pending = replace(
        run_module.RunState.from_dict(state), next_parent_candidate_id=parent
    )
    restore = shlex.split(parent_restore_command(pending, repo))
    assert restore[1:3] == ["--gepa-dir", str(repo / ".gepa")]
    applied = _run(*restore[1:])
    assert applied.exit_code == 0, applied.output
    assert run_module._current_baseline_candidate_id() == parent
    assert "private-case" not in snapshot.read_text()

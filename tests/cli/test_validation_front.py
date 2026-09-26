"""Private per-case parent selection through the real nomination protocol."""

import json
from dataclasses import replace
import shlex

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
def front_run(git_repo, monkeypatch, request):
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
    seed_alpha = (
        0.323456789 if getattr(request, "param", None) == "improving" else 0.723456789
    )
    child_beta = (
        0.823456789 if getattr(request, "param", None) == "improving" else 0.223456789
    )
    path.write_text(
        "\n".join(
            json.dumps({"name": name, "inputs": values})
            for name, values in [
                (
                    "WITHHELD_ALPHA",
                    {"bad": seed_alpha, "B": 0.876543211, "C": 0.123456789},
                ),
                (
                    "WITHHELD_BETA",
                    {"bad": 0.923456789, "B": child_beta, "C": 0.023456789},
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
    assert state["last_reflector_comparison"]["recommendation"] == "select_next_parent"
    first_packet = json.loads(
        (git_repo / ".gepa/runs" / run_id / "reflector_packet.json").read_text()
    )
    assert first_packet["last_comparison"]["recommendation"] == "select_next_parent"
    front = _private_front(git_repo, path, run_id, monkeypatch)
    assert set(front.weights()) == {seed_state["best_candidate_id"], candidate_b}
    # Seed 0's second draw chooses B, despite its losing validation mean.
    selected = (
        state["next_parent_candidate_id"] or state["reflection_baseline_candidate_id"]
    )
    assert selected == candidate_b
    if state["next_parent_candidate_id"]:
        _git(git_repo, *shlex.split(state["parent_restore_command"])[1:])
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
    _git(git_repo, *shlex.split(pending["parent_restore_command"])[1:])
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
    outputs.append(_git(git_repo, "for-each-ref", "--format=%(refname)"))
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


@pytest.mark.parametrize("tracked", [False, True])
def test_component_parent_snapshot_is_restorable(repo, monkeypatch, tracked):
    from dataclasses import replace
    import shlex
    from pydantic_ai_gepa.cli.front import parent_restore_command
    from pydantic_ai_gepa.cli.store import ComponentStore

    if tracked:
        (repo / ".gitignore").write_text("__pycache__/\n.gepa/runs/\n")
        _git(repo, "init")
        _git(repo, "config", "user.name", "GEPA Tests")
        _git(repo, "config", "user.email", "tests@example.com")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "Seed component candidate")
    path = repo.parent / f"{repo.name}-heldout.jsonl"
    path.write_text(
        json.dumps({"name": "private-case", "inputs": "?", "expected_output": "Paris"})
        + "\n"
    )
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(path))
    started = _run("run", "start", "--max-iterations", "32", "--size", "2")
    assert started.exit_code == 0, started.output
    state = _run_payload(started.output)
    parent = state["best_candidate_id"]
    snapshot = repo / ".gepa/runs" / str(state["run_id"]) / "parents" / f"{parent}.json"
    saved = json.loads(snapshot.read_text())
    slot = next(iter(saved["components"]))
    ComponentStore().write(slot, "Changed instructions")
    if tracked:
        _git(repo, "add", ".gepa/components")
        _git(repo, "commit", "-m", "Change component candidate")
    pending = replace(
        run_module.RunState.from_dict(state),
        next_parent_candidate_id=parent,
        status="paused_after_candidate_eval",
    )
    pending.save()
    restore = shlex.split(parent_restore_command(pending, repo))
    assert restore[1:3] == ["--gepa-dir", str(repo / ".gepa")]
    assert ("--commit" in restore) == tracked
    applied = _run(*restore[1:])
    assert applied.exit_code == 0, applied.output
    assert run_module._current_baseline_candidate_id() == parent
    assert "private-case" not in snapshot.read_text()
    if tracked:
        assert not _git(repo, "status", "--porcelain")
    monkeypatch.delenv("GEPA_HELDOUT_DATASET")
    run_id = str(state["run_id"])
    nominated = harness._continue(run_id)
    assert nominated.exit_code == 0, nominated.output
    assert json.loads(nominated.output)["status"] == "pending"
    assert harness._serve(run_id, path, monkeypatch).exit_code == 0
    continued = harness._continue(run_id)
    assert continued.exit_code == 0, continued.output
    assert _run_payload(continued.output)["reflection_baseline_candidate_id"] == parent


@pytest.mark.parametrize("front_run", ["improving"], indirect=True)
def test_best_commit_survives_restoring_another_parent(
    git_repo, front_run, monkeypatch
):
    path, run_id, seed_state, _ = front_run
    branch = _git(git_repo, "symbolic-ref", "HEAD")
    harness._commit(git_repo, "B")
    best_sha = _git(git_repo, "rev-parse", "HEAD")
    assert harness._continue(run_id).exit_code == 0
    assert harness._serve(run_id, path, monkeypatch).exit_code == 0
    accepted = harness._continue(run_id)
    assert accepted.exit_code == 0, accepted.output
    best = _run_payload(accepted.output)
    assert best["best_commit_sha"] == best_sha
    assert best["last_reflector_comparison"]["validation_improved"] is True

    harness._commit(git_repo, "C")
    assert harness._continue(run_id).exit_code == 0
    assert harness._serve(run_id, path, monkeypatch).exit_code == 0
    rejected = harness._continue(run_id)
    assert rejected.exit_code == 0, rejected.output
    pending = _run_payload(rejected.output)
    assert pending["next_parent_candidate_id"] == seed_state["best_candidate_id"]
    restore = shlex.split(pending["parent_restore_command"])
    assert restore[:3] == ["git", "reset", "--hard"]
    _git(git_repo, *restore[1:])
    assert _git(git_repo, "symbolic-ref", "HEAD") == branch
    ref = f"refs/gepa/{run_id}/{best_sha[:12]}"
    assert _git(git_repo, "show-ref", "--verify", ref).split()[0] == best_sha
    assert ref in _git(
        git_repo, "for-each-ref", "--contains", best_sha, "--format=%(refname)"
    )
    # Reflogs must not be the last owner of a front or best commit.
    _git(git_repo, "reflog", "expire", "--expire=now", "--all")
    _git(git_repo, "gc", "--prune=now")
    _, report = run_module._write_final_report(
        run_module.RunState.from_dict(pending), root=git_repo
    )
    restore_best = next(
        line.split(": ", 1)[1]
        for line in report.splitlines()
        if line.startswith("- best_restore_command:")
    )
    _git(git_repo, *shlex.split(restore_best)[1:])
    assert _git(git_repo, "rev-parse", "HEAD") == best_sha
    assert _git(git_repo, "symbolic-ref", "HEAD") == branch


@pytest.mark.parametrize(
    "invalid", ["", "../private", "other/ref", "a.lock", "--option"]
)
def test_retention_ref_rejects_invalid_public_ids(git_repo, invalid):
    import typer
    from pydantic_ai_gepa.cli.front import _pin_git_candidate

    sha = _git(git_repo, "rev-parse", "HEAD")
    for run_id, candidate in [(invalid, sha[:12]), ("public-run", invalid)]:
        with pytest.raises(
            typer.BadParameter, match="Invalid public candidate identity"
        ):
            _pin_git_candidate(git_repo, run_id, candidate, sha)
    assert not _git(git_repo, "for-each-ref", "refs/gepa")


def test_infrastructure_pause_keeps_parent_but_shows_recovery(
    git_repo, front_run, monkeypatch
):
    path, run_id, seed_state, _ = front_run
    state = replace(
        run_module.RunState.from_dict(seed_state),
        status="paused_after_candidate_eval",
        next_parent_candidate_id=str(seed_state["best_candidate_id"]),
        next_parent_commit_sha=str(seed_state["best_commit_sha"]),
        best_validation_samples=(),
    )
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(path))
        state.save()
    original = run_module.run_eval_once
    test_run_cli._fail_rollout_calls(monkeypatch, {1})
    assert harness._continue(run_id).exit_code == 0
    assert harness._serve(run_id, path, monkeypatch).exit_code == 0
    paused = harness._continue(run_id)
    assert paused.exit_code == 0, paused.output
    payload = _run_payload(paused.output)
    assert payload["status"] == "paused_after_infrastructure_error"
    assert payload["next_parent_candidate_id"] == state.next_parent_candidate_id
    assert "parent_restore_command" not in payload
    assert "Next parent:" not in paused.output
    assert "required evaluation rollout failed" in paused.output
    packet = json.loads(
        (git_repo / ".gepa/runs" / run_id / "reflector_packet.json").read_text()
    )
    assert "next_parent" not in packet
    assert "required evaluation rollout failed" in packet["instructions"]
    monkeypatch.setattr(run_module, "run_eval_once", original)
    assert harness._continue(run_id).exit_code == 0
    assert harness._serve(run_id, path, monkeypatch).exit_code == 0
    resumed = harness._continue(run_id)
    assert resumed.exit_code == 0, resumed.output
    recovered = _run_payload(resumed.output)
    assert recovered["status"] == "paused_for_reflection"
    assert (
        recovered["reflection_baseline_candidate_id"] == state.next_parent_candidate_id
    )
    assert recovered["next_parent_candidate_id"] is None


def test_front_has_private_schema_and_migrates_legacy_store(
    git_repo, front_run, monkeypatch
):
    import stat

    path, run_id, _, _ = front_run
    front = _private_front(git_repo, path, run_id, monkeypatch)
    data = json.loads(front.path.read_text())
    assert set(data) == {"digest", "evaluations", "selections"}
    assert data == front.data
    assert stat.S_IMODE(front.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(front.path.parent.stat().st_mode) == 0o700
    front.path.write_text(json.dumps({"identity": data, "scores": {}}))
    reopened = _private_front(git_repo, path, run_id, monkeypatch)
    assert reopened.data == data
    reopened.preflight()
    assert json.loads(front.path.read_text()) == data


def test_reseeding_checks_front_storage_before_rollout(
    git_repo, front_run, monkeypatch
):
    import typer
    from pydantic_ai_gepa.cli import front as front_module

    path, run_id, seed_state, _ = front_run

    def readonly(*args):
        raise PermissionError(str(path))

    monkeypatch.setattr(front_module, "_write_front", readonly)
    monkeypatch.setattr(
        run_module,
        "run_eval_once",
        lambda **_: pytest.fail("No paid seed before storage preflight"),
    )
    state = replace(
        run_module.RunState.from_dict(seed_state), best_validation_samples=()
    )
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(path))
        with harness_environment():
            with pytest.raises(
                typer.BadParameter,
                match="Private parent front storage must be writable",
            ) as error:
                run_module._ensure_validation_seed(state)
    assert str(path) not in str(error.value)

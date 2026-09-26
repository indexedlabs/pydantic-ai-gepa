"""Held-out scoring crosses the nomination queue, never the reflector process."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from pydantic_ai_gepa.cli import run as run_module
from pydantic_ai_gepa.cli import layout
from tests.cli import test_git_candidate_cli
from tests.cli.test_git_candidate_cli import _git, _run, _run_payload

git_repo = test_git_candidate_cli.git_repo


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(layout, "_explicit_gepa_dirname", None)
    monkeypatch.delenv("GEPA_DIR", raising=False)
    monkeypatch.delenv("GEPA_HELDOUT_DATASET", raising=False)


@pytest.fixture
def heldout(git_repo, monkeypatch, request):
    (git_repo / "task_pkg" / "evaluation.py").write_text(
        'from pathlib import Path\nasync def evaluate(case):\n    return Path("score.txt").read_text().strip()\n'
    )
    metered = getattr(request, "param", False)
    if metered:
        (git_repo / "task_pkg/evaluation.py").write_text(
            """from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_gepa.spend import current_rollout_capability
async def evaluate(case):
    score = Path("score.txt").read_text().strip()
    name = "jump" if score == "good" and case.name == "WITHHELD_BETA" else "test"
    await Agent(TestModel(custom_output_text="ok", model_name=name)).run(
        "?", capabilities=[current_rollout_capability()])
    if case.name.startswith("WITHHELD"):
        print("WITHHELD_INPUT diagnostic must stay private")
    return score
"""
        )
        (git_repo / "task_pkg/pricing.py").write_text(
            'def price(response): return 0.30 if response.model_name == "jump" else 0.01\n'
        )
        config = git_repo / ".gepa/gepa.toml"
        config.write_text('price_fn = "task_pkg.pricing:price"\n' + config.read_text())
    _git(git_repo, "add", "task_pkg", ".gepa/gepa.toml")
    _git(git_repo, "commit", "-m", "Use trace-independent fake evaluator")
    private = git_repo.parent / f"private-{git_repo.name}"
    private.mkdir()
    path = private / "secret-evaluation-set.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {"name": name, "inputs": "WITHHELD_INPUT", "expected_output": "good"}
            )
            for name in ("WITHHELD_ALPHA", "WITHHELD_BETA")
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
        "16",
        "--acceptance-repetitions",
        "1",
        "--acceptance-paired-min-cases",
        "2",
        *(["--max-token-cost", "0.20"] if metered else []),
    )
    assert started.exit_code == 0, started.output
    run_id = str(_run_payload(started.output)["run_id"])
    monkeypatch.delenv("GEPA_HELDOUT_DATASET")
    return path, run_id, started.output


def _commit(repo, score="good"):
    (repo / "score.txt").write_text(score + "\n")
    _git(repo, "add", "score.txt")
    _git(repo, "commit", "-m", "Reflector candidate")


def _continue(run_id, *extra):
    return _run("run", "continue", "--run-id", run_id, "--wait-secs", "0", *extra)


def _serve(run_id, path, monkeypatch):
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(path))
        return _run("harness", "serve", "--run-id", run_id, "--once")


def _sweep(repo, path, *outputs):
    public = "\n".join(
        p.read_text(errors="replace")
        for p in (repo / ".gepa").rglob("*")
        if p.is_file()
    ) + "\n".join(outputs)
    for secret in (
        str(path),
        path.name,
        str(path.parent),
        "WITHHELD_ALPHA",
        "WITHHELD_BETA",
        "WITHHELD_INPUT",
        "0.723456789",
        "0.876543211",
    ):
        assert secret not in public


@pytest.mark.parametrize("verdict", ["training_rejected", "improved", "not_improved"])
def test_only_harness_scores_and_results_preserve_continue_contract(
    git_repo, heldout, monkeypatch, verdict
):
    path, run_id, start_output = heldout
    _commit(git_repo, "bad-again" if verdict == "training_rejected" else "good")
    original = run_module.run_eval_once
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs.get("dataset_role", "training"))
        outcome = original(**kwargs)
        if kwargs.get("dataset_role") == "validation":
            print(f"Private evaluator diagnostic: {path} WITHHELD_ALPHA 0.723456789")
            scores = [0.723456789, 0.876543211] if verdict == "improved" else [0.0, 0.0]
            outcome.records[:] = [
                replace(record, score=score)
                for record, score in zip(outcome.records, scores)
            ]
            outcome.summary["mean_score"] = sum(scores) / 2
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", evaluate)
    first = _continue(run_id, "--reflector-epoch", "1")
    assert first.exit_code == 0, first.output
    assert json.loads(first.output)["status"] == "pending"
    assert calls == []
    # Accidentally inheriting the env does not turn continue into a scorer.
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(path))
        repeated = _continue(run_id, "--reflector-epoch", "1")
    assert repeated.output == first.output
    assert calls == []
    served = _serve(run_id, path, monkeypatch)
    assert served.exit_code == 0, served.output
    assert calls and ("validation" in calls) == (verdict != "training_rejected")
    monkeypatch.setattr(
        run_module, "run_eval_once", lambda **_: pytest.fail("reflector must not score")
    )
    result = _continue(run_id, "--reflector-epoch", "1")
    assert result.exit_code == 0, result.output
    comparison = _run_payload(result.output)["last_reflector_comparison"]
    assert comparison["improved"] == (verdict == "improved")
    if verdict != "training_rejected":
        assert comparison["training_verdict"] == "accepted"
        assert comparison["validation_improved"] == (verdict == "improved")
    root = git_repo / ".gepa" / "runs" / run_id
    assert len(list((root / "nominations").glob("*.json"))) == 1
    saved = json.loads(next((root / "results").glob("*.json")).read_text())
    assert saved["stdout"] == result.stdout
    assert saved["stderr"] == result.stderr
    assert saved["exit_code"] == result.exit_code
    status = _run("run", "status", "--run-id", run_id)
    resume = _run("run", "resume", "--run-id", run_id)
    assert status.exit_code == 0
    assert resume.exit_code == 2
    assert "GEPA_HELDOUT_DATASET" in resume.output
    _sweep(
        git_repo,
        path,
        start_output,
        first.output,
        repeated.output,
        result.output,
        status.output,
        resume.output,
    )


@pytest.mark.parametrize("change", ["head", "dirty", "epoch"])
def test_stale_nomination_is_rejected_without_scoring(
    git_repo, heldout, monkeypatch, change
):
    path, run_id, _ = heldout
    _commit(git_repo)
    assert _continue(run_id).exit_code == 0
    if change == "head":
        _commit(git_repo, "different")
    elif change == "dirty":
        (git_repo / "score.txt").write_text("dirty")
    else:
        # Epoch changes are harness-owned; public resume cannot revoke a nomination.
        with monkeypatch.context() as env:
            env.setenv("GEPA_HELDOUT_DATASET", str(path))
            assert _run("run", "resume", "--run-id", run_id).exit_code == 0
    monkeypatch.setattr(
        run_module,
        "run_eval_once",
        lambda **_: pytest.fail("stale candidate must not be scored"),
    )
    served = _serve(run_id, path, monkeypatch)
    assert served.exit_code == 0, served.output
    result = json.loads(
        next(
            (git_repo / ".gepa" / "runs" / run_id / "results").glob("*.json")
        ).read_text()
    )
    assert result["exit_code"] == 2
    assert "Stale nomination" in result["stderr"]
    _sweep(git_repo, path)


def test_pending_identity_timeout_and_epoch(git_repo, heldout):
    _, run_id, _ = heldout
    _commit(git_repo)
    first = _continue(run_id)
    assert _continue(run_id).output == first.output
    timed = _run("run", "continue", "--run-id", run_id, "--wait-secs", "0.01")
    assert timed.exit_code == 75
    assert "same continue command" in timed.output
    _commit(git_repo, "different")
    refused = _continue(run_id)
    assert (
        refused.exit_code == 2 and "different nomination is pending" in refused.output
    )
    stale = _continue(run_id, "--reflector-epoch", "0")
    assert stale.exit_code == 2 and "Stale reflector epoch" in stale.output


def test_changed_dataset_and_missing_environment_fail_closed(
    git_repo, heldout, monkeypatch
):
    path, run_id, _ = heldout
    _commit(git_repo)
    assert _continue(run_id).exit_code == 0
    for argv in (
        ("harness", "serve", "--run-id", run_id, "--once"),
        ("eval", "--dataset-role", "validation"),
        ("run", "start", "--heldout-required"),
    ):
        result = _run(*argv)
        assert result.exit_code == 2, result.output
        assert "GEPA_HELDOUT_DATASET" in result.output
        _sweep(git_repo, path, result.output)
    path.write_text(path.read_text() + "\n")
    monkeypatch.setattr(
        run_module,
        "run_eval_once",
        lambda **_: pytest.fail("changed dataset must not be scored"),
    )
    changed = _serve(run_id, path, monkeypatch)
    assert changed.exit_code == 2
    assert "changed after run start" in changed.output
    _sweep(git_repo, path, changed.output)


def test_legacy_state_is_refused_without_disclosing_path(git_repo, heldout):
    path, run_id, _ = heldout
    state_path = git_repo / ".gepa" / "runs" / run_id / "state.json"
    raw = json.loads(state_path.read_text())
    raw["validation_dataset_path"] = str(path)
    state_path.write_text(json.dumps(raw))
    result = _run("run", "status", "--run-id", run_id)
    assert result.exit_code == 2
    assert "Legacy held-out state" in result.output
    assert str(path) not in result.output


def test_scoring_tree_change_cannot_publish_a_promotion(git_repo, heldout, monkeypatch):
    path, run_id, _ = heldout
    _commit(git_repo)
    before = run_module._load_state(run_id)
    assert _continue(run_id).exit_code == 0
    original = run_module.run_eval_once

    def move_during_evaluation(**kwargs):
        outcome = original(**kwargs)
        (git_repo / "score.txt").write_text("changed while scoring")
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", move_during_evaluation)
    result = _serve(run_id, path, monkeypatch)
    assert result.exit_code == 0, result.output
    state = run_module._load_state(run_id)
    assert state.best_candidate_id == before.best_candidate_id
    assert state.best_mean_score == before.best_mean_score
    saved = json.loads(
        next(
            (git_repo / ".gepa" / "runs" / run_id / "results").glob("*.json")
        ).read_text()
    )
    assert saved["exit_code"] == 2, saved["stderr"]
    assert saved["stale"]
    assert state.continuation is None
    from pydantic_ai_gepa.cli.runs import ParetoLog

    paid = [row.extra["eval_id"] for row in ParetoLog(run_id).iter_rows()]
    assert state.iterations == ParetoLog(run_id).count_budget_rows()
    monkeypatch.setattr(run_module, "run_eval_once", original)
    _commit(git_repo, "fixed candidate")
    assert _continue(run_id).exit_code == 0
    assert _serve(run_id, path, monkeypatch).exit_code == 0
    delivered = _continue(run_id)
    assert delivered.exit_code in (0, 70), delivered.output
    rows = ParetoLog(run_id).iter_rows()
    assert [row.extra["eval_id"] for row in rows[: len(paid)]] == paid
    assert len({row.extra["eval_id"] for row in rows}) == len(rows)
    assert run_module._load_state(run_id).continuation is None
    _sweep(git_repo, path, delivered.output)


def test_private_helpers_require_harness_environment_even_with_cached_evidence(heldout):
    import typer
    from pydantic_ai_gepa.cli.reflector_recovery import _row_outcome
    from pydantic_ai_gepa.cli.runs import ParetoLog
    from pydantic_ai_gepa.cli.select import _run_select_locked

    _, run_id, _ = heldout
    state = run_module._load_state(run_id)
    assert state.heldout_required and state.best_validation_samples
    row = ParetoLog(run_id).validation_rows()[0]
    for call in (
        lambda: run_module._ensure_validation_seed(state),
        lambda: run_module._confirm_validation_candidate(state),
        lambda: state.restore_validation_evidence(),
        lambda: _row_outcome(state, row, 1, state.threshold),
        lambda: _run_select_locked(Path(state.project_root), state),
    ):
        with pytest.raises(typer.BadParameter, match="GEPA_HELDOUT_DATASET"):
            call()


def test_invalid_heldout_file_never_appears_in_error_output(
    git_repo, heldout, monkeypatch
):
    path, _, _ = heldout
    path.write_text("invalid WITHHELD_ALPHA data")
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(path))
        result = _run("run", "start")
    assert result.exit_code == 2, result.output
    _sweep(git_repo, path, result.output)


def test_harness_replays_budget_exit_code(git_repo, heldout, monkeypatch):
    path, run_id, _ = heldout
    state = run_module._load_state(run_id)
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(path))
        replace(state, max_iterations=state.iterations + 1).save()
    _commit(git_repo)
    assert _continue(run_id).exit_code == 0
    served = _serve(run_id, path, monkeypatch)
    assert served.exit_code == 0, served.output
    result = _continue(run_id)
    assert result.exit_code == 70, result.output
    _sweep(git_repo, path, result.output)


@pytest.mark.parametrize("heldout", [True], indirect=True)
def test_validation_spend_stop_crosses_harness_without_promoting_partial_candidate(
    git_repo, heldout, monkeypatch
):
    from pydantic_ai_gepa.cli.spend import spend_report

    path, run_id, start_output = heldout
    initial = _run_payload(start_output)
    incumbent = initial["best_candidate_id"]
    _commit(git_repo)
    queued = _continue(run_id)
    assert queued.exit_code == 0, queued.output
    assert json.loads(queued.output)["status"] == "pending"
    served = _serve(run_id, path, monkeypatch)
    assert served.exit_code == 0, served.output
    delivered = _continue(run_id)
    assert delivered.exit_code == 70, delivered.output
    result = _run_payload(delivered.output)
    assert result["status"] == "done"
    assert result["last_comparison"]["reason_code"] == "cost_budget_exhausted"
    assert result["best_candidate_id"] == incumbent
    assert result["best_mean_score"] == initial["best_mean_score"]
    assert result["spend"]["validation_dollars"] == pytest.approx(0.33)
    assert result["spend"]["stopped_by_cost"]
    # The reflector receives the exact trusted output/code captured by the harness.
    results = list((git_repo / ".gepa/runs" / run_id / "results").glob("*.json"))
    stored = json.loads(results[0].read_text())
    assert delivered.stdout == stored["stdout"]
    assert delivered.stderr == stored["stderr"]
    assert delivered.exit_code == stored["exit_code"]
    assert spend_report(run_id)["total_dollars"] == result["spend"]["total_dollars"]
    _sweep(git_repo, path, start_output, queued.output, served.output, delivered.output)

"""Regression coverage for harness output, recovery, locking and private access."""

from dataclasses import replace
import importlib
import json
import logging
import sys
import threading

import pytest

from pydantic_ai_gepa.cli import harness, run as run_module
from pydantic_ai_gepa.cli.reflector import run_lock
from pydantic_ai_gepa.cli.runs import ParetoLog
from tests.cli import test_harness_scoring as fixtures, test_run_cli
from tests.cli.test_harness_scoring import (
    _commit,
    _continue,
    _git,
    _run,
    _run_payload,
    _serve,
    _sweep,
)

git_repo = fixtures.git_repo
heldout = fixtures.heldout
isolated_environment = fixtures.isolated_environment
repo = test_run_cli.repo

EVALUATOR = """from pathlib import Path
async def evaluate(case):
    return Path("score.txt").read_text().strip()
"""


def _edit_evaluator(repo, source):
    (repo / "task_pkg/evaluation.py").write_text(source)
    _git(repo, "add", "task_pkg/evaluation.py")
    _git(repo, "commit", "-m", "Change evaluator")


def _result(repo, run_id, queued):
    nomination = json.loads(queued.output)["nomination_id"]
    return json.loads(
        (repo / ".gepa/runs" / run_id / "results" / f"{nomination}.json").read_text()
    )


@pytest.mark.parametrize("import_in_tree", [False, True])
def test_logging_and_evaluator_stdio_never_enter_result(
    git_repo, heldout, monkeypatch, import_in_tree
):
    path, run_id, _ = heldout
    _edit_evaluator(
        git_repo,
        """from pathlib import Path
import logging
import sys
import typer
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("candidate")
print("IMPORT_STDOUT")
print("IMPORT_STDERR", file=sys.stderr)
async def evaluate(case):
    log.info("evaluating %s %s", case.name, case.inputs)
    print("EVALUATOR_STDOUT", case.name, case.inputs)
    print("EVALUATOR_STDERR", case.name, case.inputs, file=sys.stderr)
    typer.echo("EVALUATOR_ECHO " + case.name)
    return Path("score.txt").read_text().strip()
""",
    )
    _commit(git_repo)
    queued = _continue(run_id)
    assert queued.exit_code == 0, queued.output
    original_tree = harness._current_tree

    def tree(state, project):
        # Component identity resolution imports the agent before training starts.
        module = importlib.import_module("task_pkg.evaluation")
        assert hasattr(module, "log")  # The changed module really was imported here.
        return original_tree(state, project)

    with monkeypatch.context() as patch:
        patch.setattr(logging.root, "handlers", [])
        patch.setattr(logging.root, "level", logging.WARNING)
        if import_in_tree:
            patch.delitem(sys.modules, "task_pkg.evaluation", raising=False)
            patch.setattr(harness, "_current_tree", tree)
        served = _serve(run_id, path, patch)
        assert served.exit_code == 0, served.output
    delivered = _continue(run_id)
    assert delivered.exit_code == 0, delivered.output
    stored = _result(git_repo, run_id, queued)
    assert _run_payload(delivered.output)["last_reflector_comparison"][
        "validation_improved"
    ]
    for output in (delivered.output, stored["stdout"], stored["stderr"]):
        for sentinel in ("evaluating ", "IMPORT_STD", "EVALUATOR_", "WITHHELD_"):
            assert sentinel not in output
    _sweep(git_repo, path, delivered.output)


@pytest.mark.parametrize(
    "failure,phase",
    [("syntax", "training"), ("runtime", "training"), ("runtime", "held-out")],
)
def test_unexpected_scoring_error_returns_result_and_fixed_commit_recovers(
    git_repo, heldout, monkeypatch, failure, phase
):
    path, run_id, _ = heldout
    source = (
        "def evaluate(:\n"
        if failure == "syntax"
        else (
            "from pydantic_ai_gepa.cli.validation import evaluation_phase\n"
            f"if evaluation_phase() == {phase!r}:\n"
            "    raise RuntimeError('WITHHELD_INPUT must not enter the diagnostic')\n"
            + EVALUATOR
        )
    )
    _edit_evaluator(git_repo, source)
    _commit(git_repo)
    queued = _continue(run_id)
    assert queued.exit_code == 0, queued.output
    served = _serve(run_id, path, monkeypatch)
    assert served.exit_code == 0, served.output
    result = _result(git_repo, run_id, queued)
    assert result["exit_code"] == 1
    assert result["retryable"]
    assert ("SyntaxError" if failure == "syntax" else "RuntimeError") in result[
        "stderr"
    ]
    assert phase in result["stderr"]
    delivered = _continue(run_id)
    assert delivered.exit_code == 1
    _sweep(git_repo, path, delivered.output)
    failed_state = run_module._load_state(run_id)
    if phase == "training":
        assert failed_state.continuation is None
    else:
        assert failed_state.continuation is not None  # Paid training is retained.
    paid_ids = [row.extra["eval_id"] for row in ParetoLog(run_id).iter_rows()]
    _edit_evaluator(git_repo, EVALUATOR)
    assert _continue(run_id).exit_code == 0
    fixed = _serve(run_id, path, monkeypatch)
    assert fixed.exit_code == 0, fixed.output
    verdict = _continue(run_id)
    assert verdict.exit_code == 0, verdict.output
    assert _run_payload(verdict.output)["last_reflector_comparison"][
        "validation_improved"
    ]
    rows = ParetoLog(run_id).iter_rows()
    assert [row.extra["eval_id"] for row in rows[: len(paid_ids)]] == paid_ids
    assert len({row.extra["eval_id"] for row in rows}) == len(rows)
    assert run_module._load_state(run_id).continuation is None
    _sweep(git_repo, path, verdict.output)


def test_runtime_evaluator_failure_can_be_fixed_without_operator(
    git_repo, heldout, monkeypatch
):
    path, run_id, _ = heldout
    _edit_evaluator(
        git_repo,
        "async def evaluate(case):\n    raise RuntimeError('fake evaluator failed')\n",
    )
    _commit(git_repo)
    assert _continue(run_id).exit_code == 0
    assert _serve(run_id, path, monkeypatch).exit_code == 0
    failed = _continue(run_id)
    assert failed.exit_code == 0, failed.output
    assert _run_payload(failed.output)["status"] == "paused_after_infrastructure_error"
    assert run_module._load_state(run_id).continuation is None
    _edit_evaluator(git_repo, EVALUATOR)
    assert _continue(run_id).exit_code == 0
    assert _serve(run_id, path, monkeypatch).exit_code == 0
    assert _run_payload(_continue(run_id).output)["last_reflector_comparison"][
        "validation_improved"
    ]


@pytest.mark.parametrize("timeout", [False, True])
def test_nomination_waits_for_lock_or_returns_75(
    git_repo, heldout, monkeypatch, timeout
):
    _, run_id, _ = heldout
    _commit(git_repo)
    ready, release = threading.Event(), threading.Event()

    def hold():
        with run_lock(run_id):
            ready.set()
            release.wait(5 if timeout else 0.1)

    thread = threading.Thread(target=hold)
    thread.start()
    assert ready.wait(5)
    if timeout:
        monkeypatch.setattr(harness, "NOMINATION_LOCK_TIMEOUT", 0.01)
    try:
        nominated = _continue(run_id)
        assert nominated.exit_code == (75 if timeout else 0), nominated.output
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert _continue(run_id).exit_code == 0
    assert (
        len(list((git_repo / ".gepa/runs" / run_id / "nominations").glob("*.json")))
        == 1
    )


def test_idle_serve_does_not_take_lock_or_check_pin(heldout, monkeypatch):
    path, run_id, _ = heldout
    monkeypatch.setattr(harness, "run_lock", lambda *a, **kw: pytest.fail("idle lock"))
    monkeypatch.setattr(
        harness, "check_heldout_pin", lambda *a: pytest.fail("idle pin read")
    )
    result = _serve(run_id, path, monkeypatch)
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("heldout", [True], indirect=True)
def test_harness_spend_paths_use_project_root_from_subdirectory(
    git_repo, heldout, monkeypatch
):
    path, run_id, _ = heldout
    _commit(git_repo)
    assert _continue(run_id).exit_code == 0
    child = git_repo / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    served = _serve(run_id, path, monkeypatch)
    assert served.exit_code == 0, served.output
    result = _continue(run_id)
    assert result.exit_code == 70, result.output
    spend = _run_payload(result.output)["spend"]
    assert not spend.get("validation_checkpoint_missing")
    assert spend["validation_dollars"] == pytest.approx(0.33)


def test_first_dirty_nomination_requests_a_commit(git_repo, heldout):
    _, run_id, _ = heldout
    (git_repo / "score.txt").write_text("dirty")
    result = _continue(run_id)
    assert result.exit_code == 2
    assert "tree is dirty" in result.output and "commit your candidate" in result.output
    assert "Stale nomination" not in result.output


def test_dirty_component_nomination_requests_a_commit(repo, monkeypatch):
    from pydantic_ai_gepa.cli.store import ComponentStore

    private = repo.parent / f"{repo.name}-heldout.jsonl"
    private.write_text(
        json.dumps({"name": "private-case", "inputs": "?", "expected_output": "Paris"})
        + "\n"
    )
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(private))
    started = _run("run", "start", "--size", "2", "--max-iterations", "30")
    assert started.exit_code == 0, started.output
    run_id = _run_payload(started.output)["run_id"]
    monkeypatch.delenv("GEPA_HELDOUT_DATASET")
    (repo / ".gitignore").write_text("__pycache__/\n.gepa/runs/\n")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Tests")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Seed component candidate")
    ComponentStore().write("instructions", "Uncommitted prompt edit")
    result = _continue(run_id)
    assert result.exit_code == 2, result.output
    assert "tree is dirty" in result.output and "commit your candidate" in result.output


def test_done_continue_returns_public_status_without_harness(
    git_repo, heldout, monkeypatch
):
    path, run_id, _ = heldout
    state = run_module._load_state(run_id)
    replace(state, status="done").save()
    _commit(git_repo)
    monkeypatch.setattr(
        harness, "_tree", lambda *a, **kw: pytest.fail("done tree check")
    )
    result = _continue(run_id)
    assert result.exit_code == 0, result.output
    assert _run_payload(result.output)["status"] == "done"
    assert not list((git_repo / ".gepa/runs" / run_id / "nominations").glob("*.json"))
    _sweep(git_repo, path, result.output)


def test_lane_continue_refuses_without_harness(git_repo, heldout, monkeypatch):
    _, run_id, _ = heldout
    replace(run_module._load_state(run_id), lanes=2).save()
    monkeypatch.setattr(
        harness, "_tree", lambda *a, **kw: pytest.fail("lane tree check")
    )
    result = _continue(run_id)
    assert result.exit_code == 1
    assert "does not drive lane runs" in result.output
    assert not list((git_repo / ".gepa/runs" / run_id / "nominations").glob("*.json"))


def test_harness_hides_environment_during_import_identity_and_evaluation(
    git_repo, heldout, monkeypatch
):
    path, run_id, _ = heldout
    source = """import os
from pathlib import Path
assert "GEPA_HELDOUT_DATASET" not in os.environ
async def evaluate(case):
    assert "GEPA_HELDOUT_DATASET" not in os.environ
    return Path("score.txt").read_text().strip()
"""
    _edit_evaluator(git_repo, source)
    _commit(git_repo)
    assert _continue(run_id).exit_code == 0
    original = harness._current_tree

    def identity(state, root):
        import os

        assert "GEPA_HELDOUT_DATASET" not in os.environ
        return original(state, root)

    with monkeypatch.context() as patch:
        patch.setattr(harness, "_current_tree", identity)
        result = _serve(run_id, path, patch)
        assert result.exit_code == 0, result.output
    delivered = _continue(run_id)
    assert delivered.exit_code == 0, delivered.output
    assert _run_payload(delivered.output)["last_reflector_comparison"][
        "validation_improved"
    ]
    # Startup also hides it while seeding a new held-out run.
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(path))
    started = _run("run", "start", "--size", "1", "--max-iterations", "10")
    assert started.exit_code == 0, started.output
    assert _run_payload(started.output)["validation_seeded"]

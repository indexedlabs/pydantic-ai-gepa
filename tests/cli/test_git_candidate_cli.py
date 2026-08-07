"""End-to-end coverage for git-native CLI candidates."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest
from click.testing import Result
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.layout import GepaConfig, config_path


EVALUATE_MODULE_SOURCE = textwrap.dedent("""
    import json
    import os
    from pathlib import Path

    async def evaluate(case):
        trace_path = Path(os.environ["GEPA_TRACE_FILE"])
        span = {
            "name": "pipeline stage",
            "context": {
                "trace_id": f"trace-{case.name}",
                "span_id": f"span-{case.name}",
            },
            "attributes": {
                "stage": "classify",
                "case_id": case.name,
            },
        }
        with trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(span) + "\\n")
        return Path("score.txt").read_text(encoding="utf-8").strip()
""").lstrip()


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
    module_dir = tmp_path / "task_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").touch()
    (module_dir / "evaluation.py").write_text(EVALUATE_MODULE_SOURCE, encoding="utf-8")
    (tmp_path / "score.txt").write_text("bad\n", encoding="utf-8")
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
        json.dumps(
            {
                "name": "case-1",
                "inputs": {"household": "one@example.com"},
                "expected_output": "good",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "tests@example.com")
    _git(tmp_path, "config", "user.name", "GEPA Tests")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "Seed git candidate")

    yield tmp_path

    for name in list(sys.modules):
        if name.startswith("task_pkg"):
            sys.modules.pop(name, None)


def test_git_eval_uses_plain_callable_and_exposes_trace_path(
    git_repo: Path,
) -> None:
    result = _run("eval", "--size", "1", "--capture-traces")

    assert result.exit_code == 0, result.output
    summary = next(
        json.loads(line)["summary"]
        for line in result.output.splitlines()
        if line.startswith("{") and '"summary"' in line
    )
    head = _git(git_repo, "rev-parse", "HEAD")
    assert summary["candidate_source"] == "git"
    assert summary["candidate_id"] == head[:12]
    assert summary["commit_sha"] == head
    assert summary["dirty_tree"] is False

    trace_path = Path(summary["trace_path"])
    assert (
        trace_path
        == git_repo
        / ".gepa"
        / "runs"
        / summary["run_id"]
        / "traces"
        / "minibatches"
        / summary["minibatch_id"]
        / f"0001-{summary['eval_id']}-{summary['candidate_id']}.jsonl"
    )
    trace = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert trace["attributes"]["stage"] == "classify"


def test_git_managed_run_advances_best_commit_on_improvement(
    git_repo: Path,
) -> None:
    start = _run(
        "run",
        "start",
        "--candidate-source",
        "git",
        "--size",
        "1",
        "--max-iterations",
        "2",
    )
    start_payload = _run_payload(start.output)
    baseline_sha = _git(git_repo, "rev-parse", "HEAD")
    assert start_payload["reflection_baseline_commit_sha"] == baseline_sha

    (git_repo / "score.txt").write_text("good\n", encoding="utf-8")
    _git(git_repo, "add", "score.txt")
    _git(git_repo, "commit", "-m", "Improve pipeline")
    improved_sha = _git(git_repo, "rev-parse", "HEAD")

    continued = _run("run", "continue", "--run-id", str(start_payload["run_id"]))

    assert continued.exit_code == 0, continued.output
    payload = _run_payload(continued.output)
    assert payload["status"] == "done"
    assert payload["best_commit_sha"] == improved_sha
    assert payload["best_candidate_id"] == improved_sha[:12]
    comparison = payload["last_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["improved"] is True
    assert comparison["candidate_commit_sha"] == improved_sha


def test_git_managed_run_reports_reset_then_detects_equivalent_discard(
    git_repo: Path,
) -> None:
    start = _run("run", "start", "--size", "1", "--max-iterations", "3")
    start_payload = _run_payload(start.output)
    run_id = str(start_payload["run_id"])
    baseline_sha = _git(git_repo, "rev-parse", "HEAD")

    (git_repo / "score.txt").write_text("still-bad\n", encoding="utf-8")
    _git(git_repo, "add", "score.txt")
    _git(git_repo, "commit", "-m", "Try pipeline change")

    losing = _run("run", "continue", "--run-id", run_id)

    assert losing.exit_code == 0, losing.output
    losing_payload = _run_payload(losing.output)
    comparison = losing_payload["last_comparison"]
    assert isinstance(comparison, dict)
    reset_command = f"git reset --hard {baseline_sha}"
    assert comparison["verdict"] == "equivalent"
    assert comparison["recommendation"] == "discard_no_material_change"
    assert comparison["discard_command"] == reset_command
    assert reset_command in losing.output

    _git(git_repo, "reset", "--hard", baseline_sha)
    discarded = _run("run", "continue", "--run-id", run_id)

    assert discarded.exit_code == 0, discarded.output
    assert "discarding the losing candidate and advancing" in discarded.output
    assert _run_payload(discarded.output)["status"] == "done"


def test_git_init_writes_plain_evaluate_config_without_components(
    git_repo: Path,
) -> None:
    cfg = GepaConfig.load(config_path(git_repo))

    assert cfg.candidate_source == "git"
    assert cfg.agent is None
    assert cfg.evaluate == "task_pkg.evaluation:evaluate"
    assert list((git_repo / ".gepa" / "components").iterdir()) == []

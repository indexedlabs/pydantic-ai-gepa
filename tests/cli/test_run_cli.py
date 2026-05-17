"""End-to-end tests for the managed `gepa run` controller."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest
from click.testing import Result
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.layout import final_report_path, run_state_path


AGENT_MODULE_SOURCE = textwrap.dedent("""
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(
        TestModel(custom_output_text="Paris"),
        instructions="You are a geography assistant.",
        name="geo",
    )
""").lstrip()


DATASET = [
    {"name": "case-paris", "inputs": "?", "expected_output": "Paris"},
    {"name": "case-berlin", "inputs": "?", "expected_output": "Berlin"},
]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    module_dir = tmp_path / "agent_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").touch()
    (module_dir / "agents.py").write_text(AGENT_MODULE_SOURCE, encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    init_result = _run("init", "--agent", "agent_pkg.agents:agent")
    assert init_result.exit_code == 0, init_result.output
    (tmp_path / ".gepa" / "dataset.jsonl").write_text(
        "\n".join(json.dumps(row) for row in DATASET) + "\n", encoding="utf-8"
    )

    yield tmp_path

    for name in list(sys.modules):
        if name.startswith("agent_pkg"):
            sys.modules.pop(name, None)


def _run(*argv: str) -> Result:
    return CliRunner().invoke(gepa_app, list(argv))


def _run_payload(output: str) -> dict[str, object]:
    line = next(
        line
        for line in reversed(output.splitlines())
        if line.startswith("{") and '"run"' in line
    )
    payload = json.loads(line)
    return payload["run"]


def test_managed_run_pauses_for_reflection_and_writes_trace_paths(repo: Path) -> None:
    result = _run("run", "start", "--size", "2", "--max-iterations", "3")

    assert result.exit_code == 0, result.output
    payload = _run_payload(result.output)
    assert payload["status"] == "paused_for_reflection"
    assert payload["iterations"] == 1
    assert payload["next_command"] == f"gepa run continue --run-id {payload['run_id']}"
    assert Path(str(payload["reflection_baseline_report_path"])).exists()
    assert Path(str(payload["reflection_baseline_trace_path"])).exists()
    assert run_state_path(str(payload["run_id"]), repo).exists()


def test_continue_recommends_discard_when_candidate_does_not_improve(
    repo: Path,
) -> None:
    start = _run("run", "start", "--size", "2", "--max-iterations", "3")
    run_id = str(_run_payload(start.output)["run_id"])

    result = _run("run", "continue", "--run-id", run_id)

    assert result.exit_code == 0, result.output
    assert "discard or revise" in result.output
    payload = _run_payload(result.output)
    assert payload["status"] == "paused_after_candidate_eval"
    assert payload["iterations"] == 2
    comparison = payload["last_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["recommendation"] == "discard_or_revise"
    assert comparison["delta"] == pytest.approx(0.0)
    assert Path(str(comparison["candidate_trace_path"])).exists()


def test_continue_after_revert_discards_candidate_and_advances(repo: Path) -> None:
    start = _run("run", "start", "--size", "2", "--max-iterations", "3")
    run_id = str(_run_payload(start.output)["run_id"])

    first_continue = _run("run", "continue", "--run-id", run_id)
    assert first_continue.exit_code == 0, first_continue.output
    assert _run_payload(first_continue.output)["status"] == "paused_after_candidate_eval"

    second_continue = _run("run", "continue", "--run-id", run_id)

    assert second_continue.exit_code == 0, second_continue.output
    assert "discarding the losing candidate and advancing" in second_continue.output
    payload = _run_payload(second_continue.output)
    assert payload["status"] == "done"
    assert payload["iterations"] == 3


def test_managed_run_prints_final_report_at_max_iterations(repo: Path) -> None:
    start = _run("run", "start", "--size", "2", "--max-iterations", "2")
    run_id = str(_run_payload(start.output)["run_id"])

    done = _run("run", "continue", "--run-id", run_id)

    assert done.exit_code == 0, done.output
    payload = _run_payload(done.output)
    assert payload["status"] == "done"
    assert payload["final_report_path"] == str(final_report_path(run_id, repo))
    assert Path(str(payload["final_report_path"])).exists()
    assert "GEPA Run Final Report" in done.output

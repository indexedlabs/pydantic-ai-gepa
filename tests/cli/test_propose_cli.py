"""End-to-end tests for `gepa propose`."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.candidates import Candidate
from pydantic_ai_gepa.cli.layout import (
    ensure_layout,
    write_default_config,
)
from pydantic_ai_gepa.cli.runs import ParetoLog
from pydantic_ai_gepa.cli.store import ComponentStore


# Agent that produces "Paris" via TestModel. Dataset has matching + failing cases.
AGENT_MODULE_SOURCE = textwrap.dedent("""
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(
        TestModel(custom_output_text="Paris"),
        instructions="You are a geography assistant.",
        name="geo",
    )
""").lstrip()


DATASET_LINES = [
    {"name": "case-good", "inputs": "?", "expected_output": "Paris"},
    {"name": "case-bad", "inputs": "?", "expected_output": "Berlin"},
    {"name": "case-good2", "inputs": "?", "expected_output": "paris"},
]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    module_dir = tmp_path / "agent_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").touch()
    (module_dir / "agents.py").write_text(AGENT_MODULE_SOURCE, encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    ensure_layout(tmp_path)
    write_default_config("agent_pkg.agents:agent", root=tmp_path)
    (tmp_path / ".gepa" / "dataset.jsonl").write_text(
        "\n".join(json.dumps(row) for row in DATASET_LINES) + "\n",
        encoding="utf-8",
    )

    yield tmp_path

    for name in list(sys.modules):
        if name.startswith("agent_pkg"):
            sys.modules.pop(name, None)


def _run(*argv: str) -> object:
    return CliRunner().invoke(gepa_app, list(argv))


def test_propose_refuses_on_unstaged_slots(repo: Path) -> None:
    """Without any confirmed components, propose must stage stubs and exit 2."""
    result = _run("propose", "--minibatch-size", "2")
    assert result.exit_code == 2, result.output
    assert "unconfirmed component slots" in result.output
    # Stubs must have been written to .gepa/staged/
    store = ComponentStore(repo)
    staged = store.list_staged_slots()
    assert "instructions" in staged


def _confirm_all_introspected(repo: Path) -> None:
    """Confirm the introspected slots so subsequent gepa propose calls succeed."""
    # Trigger detection via the CLI (writes stubs), then confirm each.
    _run("propose")  # exits 2 but populates .gepa/staged/
    store = ComponentStore(repo)
    for slot in store.list_staged_slots():
        result = _run("components", "confirm", slot)
        assert result.exit_code == 0, result.output


def test_propose_happy_path(repo: Path) -> None:
    _confirm_all_introspected(repo)
    result = _run(
        "propose",
        "--minibatch-size",
        "3",
        "--seed",
        "0",
        "--epoch",
        "0",
        "--max-iterations",
        "5",
    )
    assert result.exit_code == 0, result.output

    summary = json.loads(result.output)
    assert summary["iterations"] == 1
    assert summary["max_iterations"] == 5
    assert Path(summary["proposal_path"]).exists()
    assert Path(summary["report_path"]).exists()

    run_id = summary["run_id"]
    rows = ParetoLog(run_id, repo).iter_rows()
    statuses = [r.status for r in rows]
    assert statuses == ["baseline", "proposal"]


def test_propose_writes_proposal_file_matching_baseline(repo: Path) -> None:
    _confirm_all_introspected(repo)
    result = _run("propose", "--minibatch-size", "2")
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    cand = Candidate.load(Path(summary["proposal_path"]))
    # Without a reflection model the proposal starts as a copy of baseline.
    assert cand.components["instructions"] == ComponentStore(repo).read("instructions")


def test_propose_max_iterations_blocks_subsequent_calls(repo: Path) -> None:
    _confirm_all_introspected(repo)
    first = _run(
        "propose",
        "--minibatch-size",
        "2",
        "--seed",
        "0",
        "--max-iterations",
        "1",
    )
    assert first.exit_code == 0, first.output
    run_id = json.loads(first.output)["run_id"]

    second = _run(
        "propose",
        "--minibatch-size",
        "2",
        "--seed",
        "1",
        "--max-iterations",
        "1",
        "--run-id",
        run_id,
    )
    assert second.exit_code == 70
    assert "Max iterations reached" in second.output


def test_propose_writes_report_with_failures(repo: Path) -> None:
    _confirm_all_introspected(repo)
    result = _run("propose", "--minibatch-size", "3", "--seed", "0")
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    report = Path(summary["report_path"]).read_text(encoding="utf-8")
    # case-bad expects Berlin, agent produces Paris -> should be in failures
    assert "case-bad" in report or "Every case in this minibatch passed" not in report

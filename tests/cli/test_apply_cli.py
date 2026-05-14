"""End-to-end tests for `gepa apply`."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.candidates import Candidate
from pydantic_ai_gepa.cli.layout import ensure_layout, write_default_config
from pydantic_ai_gepa.cli.store import ComponentStore


AGENT_MODULE_SOURCE = textwrap.dedent('''
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(
        TestModel(),
        instructions="Initial instructions.",
        name="test-agent",
    )

    @agent.tool_plain
    def format_text(text: str) -> str:
        """Format text."""
        return text
''').lstrip()


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

    yield tmp_path

    for name in list(sys.modules):
        if name.startswith("agent_pkg"):
            sys.modules.pop(name, None)


def _run(*argv: str) -> object:
    return CliRunner().invoke(gepa_app, list(argv))


def _candidate(
    tmp_path: Path, components: dict[str, str], id_: str = "cand-apply"
) -> Path:
    cand = Candidate(id=id_, components=components)
    path = tmp_path / f"{id_}.json"
    cand.write(path)
    return path


def test_apply_writes_components(repo: Path, tmp_path: Path) -> None:
    cand_path = _candidate(tmp_path, {"instructions": "New override text."})
    result = _run("apply", "--candidate-file", str(cand_path))
    assert result.exit_code == 0, result.output

    store = ComponentStore(repo)
    assert store.read("instructions") == "New override text."


def test_apply_rejects_orphan_slot(repo: Path, tmp_path: Path) -> None:
    cand_path = _candidate(tmp_path, {"tool:nope:description": "Not on the agent."})
    result = _run("apply", "--candidate-file", str(cand_path))
    assert result.exit_code == 1
    assert "tool:nope:description" in result.output


def test_apply_with_commit(repo: Path, tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
    )

    cand_path = _candidate(tmp_path, {"instructions": "Committed text."})
    result = _run(
        "apply",
        "--candidate-file",
        str(cand_path),
        "--commit",
        "--message",
        "test: apply candidate",
    )
    assert result.exit_code == 0, result.output
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "test: apply candidate" in log.stdout

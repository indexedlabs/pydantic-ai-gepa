"""End-to-end tests for `gepa init`."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.layout import GepaConfig, config_path
from pydantic_ai_gepa.cli.store import ComponentStore


AGENT_MODULE_SOURCE = textwrap.dedent('''
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(
        TestModel(),
        instructions="Hello from init test.",
        name="init-test",
    )

    @agent.tool_plain
    def echo(text: str) -> str:
        """Echo back the input."""
        return text
''').lstrip()


@pytest.fixture
def empty_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    module_dir = tmp_path / "agent_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").touch()
    (module_dir / "agents.py").write_text(AGENT_MODULE_SOURCE, encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    yield tmp_path

    for name in list(sys.modules):
        if name.startswith("agent_pkg"):
            sys.modules.pop(name, None)


def _run(*argv: str) -> object:
    return CliRunner().invoke(gepa_app, list(argv))


def test_init_scaffolds_and_seeds(empty_repo: Path) -> None:
    result = _run("init", "--agent", "agent_pkg.agents:agent")
    assert result.exit_code == 0, result.output

    cfg = GepaConfig.load(config_path(empty_repo))
    assert cfg.agent == "agent_pkg.agents:agent"

    store = ComponentStore(empty_repo)
    # Instructions slot must be seeded with the introspected text.
    assert store.read("instructions") == "Hello from init test."
    # Tool description slot must be seeded.
    assert any(
        slot.startswith("tool:echo:description")
        for slot in store.list_confirmed_slots()
    )


def test_init_refuses_when_already_initialized(empty_repo: Path) -> None:
    _run("init", "--agent", "agent_pkg.agents:agent")
    result = _run("init", "--agent", "agent_pkg.agents:agent")
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_init_force_overwrites(empty_repo: Path) -> None:
    _run("init", "--agent", "agent_pkg.agents:agent")
    result = _run("init", "--agent", "agent_pkg.agents:agent", "--force")
    assert result.exit_code == 0, result.output


def test_init_rejects_invalid_agent_ref(empty_repo: Path) -> None:
    result = _run("init", "--agent", "no_such_module:agent")
    assert result.exit_code == 1
    assert "Could not import" in result.output

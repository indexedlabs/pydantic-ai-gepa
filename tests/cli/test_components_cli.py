"""Tests for the `gepa components` Typer verb group."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.layout import ensure_layout, write_default_config
from pydantic_ai_gepa.cli.store import ComponentStore


# Minimal agent module on disk so the CLI can resolve agent refs the same way
# end users would. We write a small Python file to tmp_path and add it to
# sys.path during the test.

AGENT_MODULE_SOURCE = textwrap.dedent('''
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(
        TestModel(),
        instructions="Initial instructions.",
        name="test-agent",
    )

    @agent.tool_plain
    def format_text(text: str, style: str = "plain") -> str:
        """Format the given text.

        Args:
            text: The text to format.
            style: Formatting style.
        """
        return f"{style}: {text}"
''').lstrip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Set up an isolated repo with a registered agent module."""
    module_dir = tmp_path / "agent_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").touch()
    (module_dir / "agents.py").write_text(AGENT_MODULE_SOURCE, encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    ensure_layout(tmp_path)
    write_default_config("agent_pkg.agents:agent", root=tmp_path)

    yield tmp_path

    # Clean module imports so subsequent tests pick up fresh source.
    for name in list(sys.modules):
        if name.startswith("agent_pkg"):
            sys.modules.pop(name, None)


def _run(*argv: str, input_: str | None = None) -> object:
    # click 8.3 merged stderr into stdout by default; we read result.output for both.
    return CliRunner().invoke(gepa_app, list(argv), input=input_)


def test_list_table_format(repo: Path) -> None:
    result = _run("components", "list")
    assert result.exit_code == 0, result.output
    assert "instructions" in result.output
    assert "SLOT" in result.output


def test_list_json_format(repo: Path) -> None:
    result = _run("components", "list", "--format", "json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    names = {row["name"] for row in data}
    assert "instructions" in names
    # All start as introspected_only with no confirmed/staged.
    instructions = next(row for row in data if row["name"] == "instructions")
    assert instructions["status"] == "introspected_only"
    assert instructions["has_confirmed"] is False
    assert instructions["has_seed"] is True


def test_list_tsv_format(repo: Path) -> None:
    result = _run("components", "list", "--format", "tsv")
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0].split("\t") == [
        "slot",
        "status",
        "has_confirmed",
        "has_staged",
        "has_seed",
    ]
    rows = [line.split("\t") for line in lines[1:]]
    names = [row[0] for row in rows]
    assert "instructions" in names


def test_list_rejects_unknown_format(repo: Path) -> None:
    result = _run("components", "list", "--format", "yaml")
    assert result.exit_code == 2
    assert "Unknown --format" in result.output


def test_show_auto_falls_back_to_seed(repo: Path) -> None:
    result = _run("components", "show", "instructions")
    assert result.exit_code == 0, result.output
    assert "Initial instructions" in result.output


def test_show_writes_to_output_file(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.md"
    result = _run("components", "show", "instructions", "--output-file", str(out))
    assert result.exit_code == 0, result.output
    assert "Initial instructions" in out.read_text(encoding="utf-8")


def test_show_returns_error_for_missing_slot(repo: Path) -> None:
    result = _run("components", "show", "tool:does-not-exist:description")
    assert result.exit_code == 1


def test_set_writes_from_content_file(repo: Path, tmp_path: Path) -> None:
    content = tmp_path / "new_instructions.md"
    content.write_text("Refined instructions.", encoding="utf-8")
    result = _run("components", "set", "instructions", "--content-file", str(content))
    assert result.exit_code == 0, result.output
    store = ComponentStore(repo)
    assert store.read("instructions") == "Refined instructions."


def test_set_reads_from_stdin(repo: Path) -> None:
    result = _run(
        "components",
        "set",
        "instructions",
        "--content-file",
        "-",
        input_="From stdin.\n",
    )
    assert result.exit_code == 0, result.output
    store = ComponentStore(repo)
    assert store.read("instructions").startswith("From stdin.")


def test_set_rejects_inline_content_flag(repo: Path) -> None:
    """The content-file rule (dec-dmk) means there is no inline --content flag."""
    result = _run(
        "components",
        "set",
        "instructions",
        "--content",
        "inline text",
    )
    # Typer reports unknown option / unexpected argument with exit code 2.
    # The exact wrapped error text varies with terminal width; the durable
    # contract is exit code 2 + no mutation to the slot.
    assert result.exit_code == 2
    store = ComponentStore(repo)
    assert store.read("instructions") is None


def test_confirm_promotes_staged(repo: Path, tmp_path: Path) -> None:
    store = ComponentStore(repo)
    store.stage("tool:format_text:description", "Seed description.")
    result = _run("components", "confirm", "tool:format_text:description")
    assert result.exit_code == 0, result.output
    assert store.read("tool:format_text:description") == "Seed description."
    assert store.read_staged("tool:format_text:description") is None


def test_confirm_with_override(repo: Path, tmp_path: Path) -> None:
    store = ComponentStore(repo)
    store.stage("tool:format_text:description", "Seed.")
    override = tmp_path / "override.md"
    override.write_text("Better description.", encoding="utf-8")
    result = _run(
        "components",
        "confirm",
        "tool:format_text:description",
        "--content-file",
        str(override),
    )
    assert result.exit_code == 0, result.output
    assert store.read("tool:format_text:description") == "Better description."


def test_confirm_missing_staged_errors(repo: Path) -> None:
    result = _run("components", "confirm", "tool:nope:description")
    assert result.exit_code == 1
    assert "No staged stub" in result.output

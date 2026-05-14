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


def test_init_force_reseeds_confirmed_slots(empty_repo: Path) -> None:
    """--force must restore every confirmed slot to its introspected seed."""
    _run("init", "--agent", "agent_pkg.agents:agent")
    store = ComponentStore(empty_repo)
    # Mutate a confirmed slot — simulate a user edit we want --force to revert.
    store.write("instructions", "user-edited text")
    assert store.read("instructions") == "user-edited text"

    result = _run("init", "--agent", "agent_pkg.agents:agent", "--force")
    assert result.exit_code == 0, result.output
    assert store.read("instructions") == "Hello from init test."


def test_init_force_preserves_staged_stubs(empty_repo: Path) -> None:
    """--force must NOT silently destroy a user's staged stub.

    A staged stub represents in-progress confirmation work. If the user adds a
    new tool, runs `gepa eval` (which stages a stub), then edits that stub,
    they expect their edit to survive a follow-up `gepa init --force`.
    """
    _run("init", "--agent", "agent_pkg.agents:agent")
    store = ComponentStore(empty_repo)
    # Hand-stage a slot the agent doesn't have — simulates the user's
    # in-progress edits on the staged file.
    store.stage("tool:future_tool:description", "custom staged description")
    assert store.read_staged("tool:future_tool:description") == (
        "custom staged description"
    )

    result = _run("init", "--agent", "agent_pkg.agents:agent", "--force")
    assert result.exit_code == 0, result.output
    # The staged file survives the re-seed.
    assert store.read_staged("tool:future_tool:description") == (
        "custom staged description"
    )


def test_init_force_cleans_orphan_slot_files(empty_repo: Path) -> None:
    """--force removes confirmed slot files that no longer correspond to an agent slot."""
    _run("init", "--agent", "agent_pkg.agents:agent")
    store = ComponentStore(empty_repo)
    # Plant an orphan slot file (no tool of this name exists on the agent).
    store.write("tool:ghost_tool:description", "leftover from a removed tool")
    assert "tool:ghost_tool:description" in store.list_confirmed_slots()

    result = _run("init", "--agent", "agent_pkg.agents:agent", "--force")
    assert result.exit_code == 0, result.output
    assert "tool:ghost_tool:description" not in store.list_confirmed_slots()
    assert "Removed 1 orphan" in result.output


def test_init_without_force_leaves_orphans_alone(empty_repo: Path) -> None:
    """Without --force, init refuses to overwrite gepa.toml so orphans are untouched."""
    _run("init", "--agent", "agent_pkg.agents:agent")
    store = ComponentStore(empty_repo)
    store.write("tool:ghost_tool:description", "leftover")
    # Second init without --force returns 1 (gepa.toml exists); orphans should still be there.
    result = _run("init", "--agent", "agent_pkg.agents:agent")
    assert result.exit_code == 1
    assert "tool:ghost_tool:description" in store.list_confirmed_slots()


def test_init_rejects_invalid_agent_ref(empty_repo: Path) -> None:
    result = _run("init", "--agent", "no_such_module:agent")
    assert result.exit_code == 1
    assert "Could not import" in result.output


# ---------- --install-skill ----------


SKILL_DEST = Path(".agents") / "skills" / "gepa-optimize" / "SKILL.md"


def test_init_without_install_skill_does_not_touch_agents_dir(empty_repo: Path) -> None:
    result = _run("init", "--agent", "agent_pkg.agents:agent")
    assert result.exit_code == 0, result.output
    assert not (empty_repo / ".agents").exists()


def test_init_install_skill_copies_packaged_md(empty_repo: Path) -> None:
    result = _run("init", "--agent", "agent_pkg.agents:agent", "--install-skill")
    assert result.exit_code == 0, result.output

    dest = empty_repo / SKILL_DEST
    assert dest.is_file()
    text = dest.read_text(encoding="utf-8")
    assert "name: gepa-optimize" in text
    assert "content-file" in text.lower()
    assert "Installed gepa-optimize skill" in result.output


def test_init_install_skill_refuses_existing_without_force(empty_repo: Path) -> None:
    first = _run("init", "--agent", "agent_pkg.agents:agent", "--install-skill")
    assert first.exit_code == 0, first.output

    dest = empty_repo / SKILL_DEST
    dest.write_text("custom skill body", encoding="utf-8")

    second = _run(
        "init",
        "--agent",
        "agent_pkg.agents:agent",
        "--install-skill",
        "--force",  # forces gepa.toml overwrite, NOT the skill
    )
    # We want to be sure the skill is NOT clobbered. The init verb's --force
    # path applies to BOTH gepa.toml and the installed skill; to assert the
    # "refuse without --force" behaviour we drive the install separately by
    # rolling back to an existing file and re-running without --force.
    dest.write_text("custom skill body", encoding="utf-8")
    third = _run(
        "init",
        "--agent",
        "agent_pkg.agents:agent",
        "--install-skill",
    )
    # init refuses because .gepa/gepa.toml already exists — exit 1.
    # The skill should not have been replaced.
    assert third.exit_code == 1
    assert dest.read_text(encoding="utf-8") == "custom skill body"
    # Sanity: the second run with --force was allowed (returned 0 in that path).
    assert second.exit_code == 0, second.output


def test_init_install_skill_with_force_overwrites_existing_skill(
    empty_repo: Path,
) -> None:
    first = _run("init", "--agent", "agent_pkg.agents:agent", "--install-skill")
    assert first.exit_code == 0, first.output

    dest = empty_repo / SKILL_DEST
    dest.write_text("custom skill body", encoding="utf-8")

    overwritten = _run(
        "init",
        "--agent",
        "agent_pkg.agents:agent",
        "--install-skill",
        "--force",
    )
    assert overwritten.exit_code == 0, overwritten.output
    # The packaged content replaced the custom body.
    assert "name: gepa-optimize" in dest.read_text(encoding="utf-8")


def test_init_install_skill_honors_skill_dest_directory(empty_repo: Path) -> None:
    """A directory passed via --skill-dest receives the bundled SKILL.md inside."""
    custom_dir = empty_repo / "my-agents-config"
    custom_dir.mkdir()
    result = _run(
        "init",
        "--agent",
        "agent_pkg.agents:agent",
        "--install-skill",
        "--skill-dest",
        str(custom_dir),
    )
    assert result.exit_code == 0, result.output
    assert (custom_dir / "SKILL.md").is_file()
    # Default destination should not have been written.
    assert not (empty_repo / SKILL_DEST).exists()


def test_init_install_skill_honors_skill_dest_file(empty_repo: Path) -> None:
    """A file path passed via --skill-dest is used literally."""
    custom_file = empty_repo / "docs" / "GEPA_SKILL.md"
    result = _run(
        "init",
        "--agent",
        "agent_pkg.agents:agent",
        "--install-skill",
        "--skill-dest",
        str(custom_file),
    )
    assert result.exit_code == 0, result.output
    assert custom_file.is_file()
    assert "name: gepa-optimize" in custom_file.read_text(encoding="utf-8")


def test_init_install_skill_existing_without_force_when_already_initialized(
    empty_repo: Path,
) -> None:
    """Standalone install-skill scenario: .gepa exists, skill exists, no --force.

    init refuses because of the existing .gepa/gepa.toml. The point of this
    test is just to confirm the skill file is not touched in that error path.
    """
    _run("init", "--agent", "agent_pkg.agents:agent", "--install-skill")
    dest = empty_repo / SKILL_DEST
    dest.write_text("user-customized skill", encoding="utf-8")

    result = _run("init", "--agent", "agent_pkg.agents:agent", "--install-skill")
    assert result.exit_code == 1
    assert dest.read_text(encoding="utf-8") == "user-customized skill"

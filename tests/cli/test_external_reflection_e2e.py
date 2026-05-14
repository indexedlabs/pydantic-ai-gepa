"""End-to-end test of the external-reflection workflow via the gepa CLI."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.layout import config_path, pareto_log_path
from pydantic_ai_gepa.cli.runs import ParetoLog
from pydantic_ai_gepa.cli.store import ComponentStore


AGENT_MODULE_SOURCE_V1 = textwrap.dedent("""
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(
        TestModel(custom_output_text="Paris"),
        instructions="You are a geography assistant.",
        name="geo",
    )
""").lstrip()


# A "v2" agent module that adds a new tool. Used to exercise the
# stage-and-confirm flow when source mutates between propose calls.
AGENT_MODULE_SOURCE_V2 = textwrap.dedent('''
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(
        TestModel(custom_output_text="Paris"),
        instructions="You are a geography assistant.",
        name="geo",
    )

    @agent.tool_plain
    def lookup_country(name: str) -> str:
        """Return the country a city is in."""
        return "France"
''').lstrip()


DATASET = [
    {"name": "case-paris", "inputs": "?", "expected_output": "Paris"},
    {"name": "case-paris2", "inputs": "?", "expected_output": "paris"},
    {"name": "case-berlin", "inputs": "?", "expected_output": "Berlin"},
]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    module_dir = tmp_path / "agent_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").touch()
    (module_dir / "agents.py").write_text(AGENT_MODULE_SOURCE_V1, encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    yield tmp_path

    for name in list(sys.modules):
        if name.startswith("agent_pkg"):
            sys.modules.pop(name, None)


def _run(*argv: str, input_: str | None = None) -> object:
    return CliRunner().invoke(gepa_app, list(argv), input=input_)


def _reload_agent_module() -> None:
    """Drop the agent_pkg.agents module so the next CLI invocation reimports v2 source."""
    for name in list(sys.modules):
        if name.startswith("agent_pkg"):
            sys.modules.pop(name, None)


def test_full_workflow_init_propose_eval_apply(repo: Path) -> None:
    # 1. Init the repo.
    result = _run("init", "--agent", "agent_pkg.agents:agent")
    assert result.exit_code == 0, result.output
    assert config_path(repo).exists()
    store = ComponentStore(repo)
    # All introspected slots should be confirmed at init time.
    assert "instructions" in store.list_confirmed_slots()

    # Write the dataset.
    (repo / ".gepa" / "dataset.jsonl").write_text(
        "\n".join(json.dumps(row) for row in DATASET) + "\n", encoding="utf-8"
    )

    # 2. components list works.
    listing = _run("components", "list", "--format", "json")
    assert listing.exit_code == 0, listing.output
    rows = json.loads(listing.output)
    assert any(r["name"] == "instructions" and r["status"] == "confirmed" for r in rows)

    # 3. propose runs to completion.
    propose = _run(
        "propose", "--minibatch-size", "3", "--seed", "0", "--max-iterations", "5"
    )
    assert propose.exit_code == 0, propose.output
    summary = json.loads(propose.output)
    proposal_path = Path(summary["proposal_path"])
    run_id = summary["run_id"]
    assert proposal_path.exists()
    rows = ParetoLog(run_id, repo).iter_rows()
    statuses = [r.status for r in rows]
    assert "baseline" in statuses and "proposal" in statuses

    # 4. eval the proposal candidate.
    eval_ = _run(
        "eval",
        "--candidate-file",
        str(proposal_path),
        "--size",
        "2",
        "--run-id",
        run_id,
    )
    assert eval_.exit_code == 0, eval_.output

    # 5. apply the proposal as the new baseline.
    apply_ = _run("apply", "--candidate-file", str(proposal_path))
    assert apply_.exit_code == 0, apply_.output

    # 6. pareto tsv listing the run shows non-empty rows.
    tsv = _run("pareto", "--run-id", run_id, "--all", "--format", "tsv")
    assert tsv.exit_code == 0, tsv.output
    body_lines = [line for line in tsv.output.strip().splitlines()[1:] if line.strip()]
    assert len(body_lines) >= 2

    # 7. journal append + show.
    note = repo / "note.md"
    note.write_text(
        "Tried minibatch size 3, found that exact-match scores dominate.",
        encoding="utf-8",
    )
    j_append = _run(
        "journal",
        "append",
        "--content-file",
        str(note),
        "--strategy",
        "minibatch-tuning",
    )
    assert j_append.exit_code == 0, j_append.output
    j_show = _run("journal", "show", "--limit", "1")
    assert j_show.exit_code == 0, j_show.output
    assert "minibatch-tuning" in j_show.output


def test_mid_run_tool_addition_triggers_stage_and_confirm(repo: Path) -> None:
    # Init and write dataset.
    _run("init", "--agent", "agent_pkg.agents:agent")
    (repo / ".gepa" / "dataset.jsonl").write_text(
        "\n".join(json.dumps(row) for row in DATASET) + "\n", encoding="utf-8"
    )

    # Run propose with v1 source — should succeed.
    first = _run("propose", "--minibatch-size", "2")
    assert first.exit_code == 0, first.output
    run_id = json.loads(first.output)["run_id"]

    # Edit source: add a new tool. Reimport to pick it up.
    (repo / "agent_pkg" / "agents.py").write_text(
        AGENT_MODULE_SOURCE_V2, encoding="utf-8"
    )
    _reload_agent_module()

    # Next propose must refuse with stage-and-confirm because the new tool
    # introduces unconfirmed component slots.
    second = _run("propose", "--minibatch-size", "2", "--run-id", run_id)
    assert second.exit_code == 2, second.output
    assert "unconfirmed component slots" in second.output

    store = ComponentStore(repo)
    new_slots = [s for s in store.list_staged_slots() if "lookup_country" in s]
    assert new_slots, store.list_staged_slots()

    # Confirm each newly staged slot; propose should now succeed.
    for slot in new_slots:
        confirm = _run("components", "confirm", slot)
        assert confirm.exit_code == 0, confirm.output

    third = _run("propose", "--minibatch-size", "2", "--run-id", run_id)
    assert third.exit_code == 0, third.output
    # The same run accumulates a second proposal.
    assert pareto_log_path(run_id, repo).exists()


def test_skill_md_shipped_with_package() -> None:
    """Smoke test that the gepa-optimize skill is importable as package data."""
    import importlib.resources

    files = importlib.resources.files("pydantic_ai_gepa")
    skill = files / "skills" / "gepa_optimize" / "SKILL.md"
    assert skill.is_file()
    text = skill.read_text(encoding="utf-8")
    assert "name: gepa-optimize" in text
    assert "content-file" in text.lower()
    # No NEVER STOP language (per dec-xd6).
    assert "NEVER STOP" not in text

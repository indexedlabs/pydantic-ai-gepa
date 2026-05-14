"""End-to-end tests for `gepa journal`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.layout import ensure_layout, journal_path


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.chdir(tmp_path)
    ensure_layout(tmp_path)
    yield tmp_path


def _run(*argv: str, input_: str | None = None) -> object:
    return CliRunner().invoke(gepa_app, list(argv), input=input_)


def test_append_from_file_and_show(repo: Path, tmp_path: Path) -> None:
    entry = tmp_path / "entry.md"
    entry.write_text(
        "Discovered that minibatch seed 42 surfaces math errors.", encoding="utf-8"
    )

    result = _run(
        "journal",
        "append",
        "--content-file",
        str(entry),
        "--strategy",
        "minibatch-tuning",
    )
    assert result.exit_code == 0, result.output
    assert journal_path(repo).read_text(encoding="utf-8").strip()

    show = _run("journal", "show", "--limit", "5")
    assert show.exit_code == 0, show.output
    rows = json.loads(show.output)
    assert len(rows) == 1
    assert rows[0]["content"].startswith("Discovered that")
    assert rows[0]["strategy"] == "minibatch-tuning"


def test_append_from_stdin(repo: Path) -> None:
    result = _run(
        "journal",
        "append",
        "--content-file",
        "-",
        input_="From stdin entry.\n",
    )
    assert result.exit_code == 0, result.output
    rows = json.loads(_run("journal", "show").output)
    assert rows[-1]["content"].startswith("From stdin entry.")


def test_show_tsv(repo: Path, tmp_path: Path) -> None:
    entry = tmp_path / "e.md"
    entry.write_text("Entry one.", encoding="utf-8")
    _run("journal", "append", "--content-file", str(entry))
    result = _run("journal", "show", "--format", "tsv")
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0].split("\t") == ["timestamp", "strategy", "content"]


def test_show_unknown_format(repo: Path) -> None:
    result = _run("journal", "show", "--format", "yaml")
    assert result.exit_code == 2


def test_append_rejects_inline_content_flag(repo: Path) -> None:
    result = _run("journal", "append", "--content", "inline-text")
    assert result.exit_code == 2

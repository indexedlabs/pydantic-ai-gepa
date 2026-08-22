"""Read-only reflector-note behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai_gepa.cli.notes import load_note, notes_index
from pydantic_ai_gepa.gepa_graph.proposal.note_tools import create_note_toolset


def test_load_note_unknown_name_lists_available_notes(tmp_path: Path) -> None:
    notes_dir = tmp_path / ".gepa" / "notes"
    notes_dir.mkdir(parents=True)
    (notes_dir / "focus.md").write_text(
        "---\nname: focus\ndescription: Keep changes narrow\n---\nreference body\n",
        encoding="utf-8",
    )

    assert [note.to_dict() for note in notes_index(notes_dir)] == [
        {"name": "focus", "description": "Keep changes narrow"}
    ]
    with pytest.raises(ValueError, match="Available notes: focus"):
        load_note(notes_dir, "missing")

    toolset = create_note_toolset(notes_dir)
    assert toolset.tools["load_note"].function(name="focus") == "reference body\n"  # type: ignore
    with pytest.raises(Exception, match="Available notes: focus"):
        toolset.tools["load_note"].function(name="missing")  # type: ignore


def test_notes_index_skips_missing_frontmatter_keys(tmp_path: Path) -> None:
    notes_dir = tmp_path / ".gepa" / "notes"
    notes_dir.mkdir(parents=True)
    (notes_dir / "incomplete.md").write_text(
        "---\nname: incomplete\n---\nignored\n", encoding="utf-8"
    )
    (notes_dir / "valid.md").write_text(
        "---\nname: valid\ndescription: Usable note\n---\nbody\n",
        encoding="utf-8",
    )

    assert [note.to_dict() for note in notes_index(notes_dir)] == [
        {"name": "valid", "description": "Usable note"}
    ]

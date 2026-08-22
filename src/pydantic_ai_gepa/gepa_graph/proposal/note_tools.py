"""Lazy, read-only reflector-note tools."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import FunctionToolset
from pydantic_ai.exceptions import ModelRetry

from ...cli.notes import load_note as read_note
from ...cli.notes import notes_index


def create_note_toolset(notes_dir: Path) -> FunctionToolset[None]:
    """Build a note loader whose tool description includes the metadata index."""

    toolset = FunctionToolset()
    index = notes_index(notes_dir)
    index_text = (
        "; ".join(f"{note.name}: {note.description}" for note in index)
        or "(no notes available)"
    )

    def load_note(name: str) -> str:
        try:
            return read_note(notes_dir, name)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    load_note.__doc__ = (
        "Load a reflector note body by name. Available notes: " + index_text
    )
    toolset.tool_plain(load_note)

    return toolset

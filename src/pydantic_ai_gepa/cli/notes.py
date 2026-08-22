"""Read-only reflector notes stored beside the managed-run journal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class NoteSummary:
    name: str
    description: str
    path: Path

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}


def notes_index(notes_dir: Path) -> list[NoteSummary]:
    """Return note metadata only; note bodies are intentionally lazy."""

    if not notes_dir.is_dir():
        return []
    notes: list[NoteSummary] = []
    for path in sorted(notes_dir.glob("*.md")):
        try:
            frontmatter, _ = _parse_note(path)
            name = frontmatter["name"]
            description = frontmatter["description"]
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            continue
        if isinstance(name, str) and isinstance(description, str):
            notes.append(NoteSummary(name=name, description=description, path=path))
    return notes


def load_note(notes_dir: Path, name: str) -> str:
    """Load one note body by its frontmatter name."""

    available = notes_index(notes_dir)
    for note in available:
        if note.name == name:
            _, body = _parse_note(note.path)
            return body
    choices = ", ".join(note.name for note in available) or "(none)"
    raise ValueError(f"Unknown note {name!r}. Available notes: {choices}.")


def _parse_note(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("missing YAML frontmatter")
    _, frontmatter_text, body = text.split("---\n", 2)
    frontmatter = yaml.safe_load(frontmatter_text)
    if not isinstance(frontmatter, dict):
        raise ValueError("frontmatter must be a mapping")
    return frontmatter, body

"""Packaging regression tests.

The CLI's `gepa init --install-skill` reads the bundled gepa-optimize SKILL.md
via ``importlib.resources``. If the build config ever drops .md files from the
wheel (default hatchling behavior is to include them, but a future version
change could regress this), `--install-skill` would silently fail. These tests
fail fast in that case.
"""

from __future__ import annotations

import importlib.resources


def test_bundled_skill_is_importable_resource() -> None:
    source = (
        importlib.resources.files("pydantic_ai_gepa")
        / "skills"
        / "gepa_optimize"
        / "SKILL.md"
    )
    assert source.is_file(), "Bundled SKILL.md should ship with the package."
    text = source.read_text(encoding="utf-8")
    assert "name: gepa-optimize" in text
    assert "content-file" in text.lower()

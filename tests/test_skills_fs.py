from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai_gepa.skills import (
    OverlayFS,
    SkillsFS,
    normalize_rel_path,
    parse_skill_md,
    render_skill_md,
)


def test_normalize_rel_path_rejects_invalid_paths() -> None:
    with pytest.raises(ValueError):
        normalize_rel_path("")
    with pytest.raises(ValueError):
        normalize_rel_path("/abs/path")
    with pytest.raises(ValueError):
        normalize_rel_path("../escape")

    assert normalize_rel_path("a/b") == "a/b"
    assert normalize_rel_path("a/./b") == "a/b"
    assert normalize_rel_path("a//b") == "a/b"


def test_skills_fs_root_listdir_and_read_write() -> None:
    fs = SkillsFS()
    fs.mkdir("skills")
    fs.write_text("skills/hello.txt", "hi")

    assert fs.listdir("") == ["skills"]
    assert fs.listdir("skills") == ["hello.txt"]
    assert fs.read_text("skills/hello.txt") == "hi"


def test_overlay_fs_overrides_base() -> None:
    base = SkillsFS()
    base.write_text("s1/SKILL.md", "---\nname: s1\ndescription: d\n---\n# base\n")

    overlay = SkillsFS()
    overlay.write_text("s1/SKILL.md", "---\nname: s1\ndescription: d\n---\n# overlay\n")

    fs = OverlayFS(base, overlay)
    assert fs.read_text("s1/SKILL.md").endswith("# overlay\n")


def test_iter_skill_dirs_detects_skill_dirs() -> None:
    fs = SkillsFS()
    fs.write_text("a/SKILL.md", "---\nname: a\ndescription: d\n---\n# A\n")
    fs.write_text("a/references/REF.md", "ref")
    fs.write_text("b/c/SKILL.md", "---\nname: c\ndescription: d\n---\n# C\n")

    assert sorted(fs.iter_skill_dirs()) == ["a", "b/c"]


def test_parse_and_render_skill_md_roundtrip_preserves_extras() -> None:
    raw = """---
name: pdf-processing
description: Extract text and tables.
allowed-tools: Bash(git:*) Read
metadata:
  author: example-org
boundary: trusted
toolsets: [index.search_records]
---
# PDF Processing

Use this skill when working with PDFs.
"""
    parsed = parse_skill_md(raw)
    assert parsed.frontmatter.name == "pdf-processing"
    assert parsed.frontmatter.description.startswith("Extract text")
    assert parsed.frontmatter.allowed_tools.startswith("Bash(")
    assert parsed.frontmatter.metadata == {"author": "example-org"}
    assert parsed.frontmatter.extras["boundary"] == "trusted"
    assert parsed.frontmatter.extras["toolsets"] == ["index.search_records"]

    rendered = render_skill_md(parsed)
    reparsed = parse_skill_md(rendered)
    assert reparsed.frontmatter.extras["boundary"] == "trusted"
    assert reparsed.frontmatter.extras["toolsets"] == ["index.search_records"]
    assert "PDF Processing" in reparsed.body


def test_skills_fs_from_disk(tmp_path: Path) -> None:
    (tmp_path / "pack" / "a").mkdir(parents=True)
    (tmp_path / "pack" / "a" / "SKILL.md").write_text(
        "---\nname: a\ndescription: d\n---\n# A\n", encoding="utf-8"
    )
    (tmp_path / "pack" / "a" / "references").mkdir(parents=True)
    (tmp_path / "pack" / "a" / "references" / "REF.md").write_text(
        "ref", encoding="utf-8"
    )

    fs = SkillsFS.from_disk(tmp_path / "pack")
    assert fs.is_file("a/SKILL.md")
    assert fs.is_file("a/references/REF.md")


def test_parse_skill_md_validation_errors() -> None:
    raw = """---
name: invalid name!
description: Extract text and tables.
---
# PDF Processing
"""
    with pytest.raises(ValueError, match="Invalid SKILL.md frontmatter metadata:"):
        parse_skill_md(raw)


def test_skills_fs_symlink_escape_protection(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "a").mkdir()
    (pack / "a" / "SKILL.md").write_text("---\nname: a\ndescription: d\n---\n# A")

    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    import os

    os.symlink(outside, pack / "a" / "escape.txt")

    with pytest.raises(ValueError, match="Symlink escape detected"):
        SkillsFS.from_disk(pack)


def test_skills_fs_max_files_limit(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    for i in range(5):
        (pack / f"file_{i}.txt").write_text("hi")

    with pytest.raises(
        ValueError, match="Filesystem traversal exceeded max_files limit"
    ):
        SkillsFS.from_disk(pack, max_files=3)

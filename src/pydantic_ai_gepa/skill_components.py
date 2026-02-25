"""GEPA component helpers for Agent Skills content."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from typing import Iterator

from .gepa_graph.models import CandidateMap, ComponentValue
from .skills import (
    OverlayFS,
    SkillsFS,
    SkillMd,
    normalize_rel_path,
    parse_skill_md,
    render_skill_md,
)


def skill_description_key(skill_path: str) -> str:
    return f"skill:{skill_path}:frontmatter:description"


def skill_body_key(skill_path: str) -> str:
    return f"skill:{skill_path}:body"


def skill_file_key(skill_path: str, relative_path: str) -> str:
    normalized = normalize_rel_path(relative_path)
    return f"skill:{skill_path}:file:{normalized}"


def iter_skill_component_keys(skill_paths: Iterable[str]) -> Iterator[str]:
    for path in skill_paths:
        yield skill_description_key(path)
        yield skill_body_key(path)


def extract_skill_components(skills_fs: SkillsFS) -> CandidateMap:
    """Extract skill components (description/body/examples) from a SkillsFS."""
    candidate: CandidateMap = {}
    for skill_dir in skills_fs.iter_skill_dirs():
        skill_md_path = _skill_md_path(skill_dir)
        try:
            raw = skills_fs.read_text(skill_md_path)
        except Exception:
            continue
        try:
            skill_md = parse_skill_md(raw)
        except Exception:
            continue

        candidate[skill_description_key(skill_dir)] = ComponentValue(
            name=skill_description_key(skill_dir),
            text=skill_md.frontmatter.description,
        )
        candidate[skill_body_key(skill_dir)] = ComponentValue(
            name=skill_body_key(skill_dir),
            text=skill_md.body,
        )

        for file_path in _iter_skill_example_files(skills_fs, skill_dir):
            try:
                content = skills_fs.read_text(file_path)
            except Exception:
                continue
            rel = file_path[len(skill_dir) + 1 :] if skill_dir else file_path
            candidate[skill_file_key(skill_dir, rel)] = ComponentValue(
                name=skill_file_key(skill_dir, rel),
                text=content,
            )
    return candidate


def materialize_skill_components_for_path(
    skills_fs: SkillsFS | OverlayFS,
    *,
    skill_path: str,
    include_examples: bool = False,
) -> CandidateMap:
    """Return baseline component values for a single skill path.

    This is intentionally filesystem-driven so callers can use an OverlayFS that
    reflects the current candidate.
    """
    normalized_skill = normalize_rel_path(skill_path)
    skill_md_path = _skill_md_path(normalized_skill)
    raw = skills_fs.read_text(skill_md_path)
    skill_md = parse_skill_md(raw)

    candidate: CandidateMap = {
        skill_description_key(normalized_skill): ComponentValue(
            name=skill_description_key(normalized_skill),
            text=skill_md.frontmatter.description,
        ),
        skill_body_key(normalized_skill): ComponentValue(
            name=skill_body_key(normalized_skill),
            text=skill_md.body,
        ),
    }

    if include_examples:
        prefix = f"{normalized_skill}/examples/"
        for path, file in skills_fs.iter_files():
            if not path.startswith(prefix):
                continue
            rel = path[len(normalized_skill) + 1 :]
            try:
                content = file.read_text()
            except Exception:
                continue
            candidate[skill_file_key(normalized_skill, rel)] = ComponentValue(
                name=skill_file_key(normalized_skill, rel),
                text=content,
            )

    return candidate


@contextmanager
def apply_candidate_to_skills(
    skills_fs: SkillsFS,
    candidate: CandidateMap | None,
) -> Iterator[OverlayFS]:
    """Apply a candidate to skills via an in-memory overlay."""
    if not candidate:
        yield OverlayFS(skills_fs)
        return

    overlay = SkillsFS()

    skill_dirs = set(skills_fs.iter_skill_dirs())
    # Add any skill dirs that only exist in the candidate
    for key in candidate.keys():
        if key.startswith("skill:"):
            parts = key.split(":")
            if len(parts) >= 3 and parts[2] in ("description", "body", "file"):
                skill_dirs.add(parts[1])

    for skill_dir in skill_dirs:
        desc_key = skill_description_key(skill_dir)
        body_key = skill_body_key(skill_dir)
        desc = candidate.get(desc_key)
        body = candidate.get(body_key)
        if desc is None and body is None:
            continue

        skill_md_path = _skill_md_path(skill_dir)
        if skills_fs.exists(skill_md_path):
            raw = skills_fs.read_text(skill_md_path)
            skill_md = parse_skill_md(raw)
        else:
            from .skills.skill_md import SkillFrontmatter, SkillMd

            skill_name = skill_dir.replace("/", "-").replace("_", "-")
            if not skill_name.replace("-", "").isalnum():
                skill_name = "new-skill"
            skill_md = SkillMd(
                frontmatter=SkillFrontmatter(name=skill_name, description=""), body=""
            )

        updated = _apply_component_updates(skill_md, description=desc, body=body)
        overlay.write_text(skill_md_path, render_skill_md(updated))

    for key, value in candidate.items():
        if not key.startswith("skill:") or ":file:" not in key:
            continue
        skill_path, rel_path = _parse_skill_file_key(key)
        if not skill_path or skill_path not in skill_dirs:
            continue
        if not rel_path.startswith("examples/"):
            continue
        overlay.write_text(f"{skill_path}/{rel_path}", value.text)

    yield OverlayFS(skills_fs, overlay)


def _apply_component_updates(
    skill_md: SkillMd,
    *,
    description: ComponentValue | None,
    body: ComponentValue | None,
) -> SkillMd:
    frontmatter = skill_md.frontmatter.model_copy()
    if description is not None:
        frontmatter.description = description.text
    updated_body = body.text if body is not None else skill_md.body
    return SkillMd(frontmatter=frontmatter, body=updated_body)


def _skill_md_path(skill_dir: str) -> str:
    if not skill_dir:
        return "SKILL.md"
    return f"{skill_dir}/SKILL.md"


def _parse_skill_file_key(key: str) -> tuple[str, str]:
    # skill:{skill_path}:file:{relative_path}
    rest = key.removeprefix("skill:")
    parts = rest.split(":file:", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid skill file key: {key}")
    skill_path, relative_path = parts
    return skill_path, normalize_rel_path(relative_path)


def _iter_skill_example_files(skills_fs: SkillsFS, skill_dir: str) -> Iterator[str]:
    base = f"{skill_dir}/examples" if skill_dir else "examples"
    if not skills_fs.is_dir(base):
        return

    def walk(path: str) -> Iterator[str]:
        for name in skills_fs.listdir(path):
            child = f"{path}/{name}"
            if skills_fs.is_file(child):
                yield child
            elif skills_fs.is_dir(child):
                yield from walk(child)

    yield from walk(base)

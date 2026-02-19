"""Tests for the student-facing skills toolset."""

from __future__ import annotations

from typing import Any, Callable

import pytest

from pydantic_ai_gepa.gepa_graph.proposal.student_tools import create_skills_toolset
from pydantic_ai_gepa.skill_components import (
    apply_candidate_to_skills,
    skill_description_key,
)
from pydantic_ai_gepa.skills.models import SkillCapability
from pydantic_ai_gepa.skills import SkillsFS
from pydantic_ai_gepa.gepa_graph.models import ComponentValue
from pydantic_ai_gepa.skills.models import SkillSearchResult
from pydantic_ai_gepa.skills.search import SkillsSearchProvider
from pydantic_ai.exceptions import ModelRetry


def _get_tool_fn(toolset: Any, name: str) -> Callable[..., Any]:
    return toolset.tools[name].function  # type: ignore[no-any-return]


def test_create_skills_toolset_list_search_load() -> None:
    fs = SkillsFS()
    fs.write_text(
        "index/tasks/SKILL.md",
        "---\nname: index-tasks\ndescription: Create and update tasks\n---\n# Tasks\n\nGuidance\n",
    )
    fs.write_text("index/tasks/references/REF.md", "reference text")

    toolset = create_skills_toolset(fs)
    tool_names = set(toolset.tools.keys())
    assert tool_names == {
        "list_skills",
        "search_skills",
        "load_skill",
        "load_skill_file",
    }

    list_fn = _get_tool_fn(toolset, "list_skills")
    skills = list_fn()
    assert skills[0].skill_path == "index/tasks"
    assert "Create and update tasks" in skills[0].description

    search_fn = _get_tool_fn(toolset, "search_skills")
    results = search_fn(query="update tasks")
    assert results
    assert results[0].skill_path == "index/tasks"

    ref_results = search_fn(query="reference")
    assert any(r.file_path == "references/REF.md" for r in ref_results)

    load_fn = _get_tool_fn(toolset, "load_skill")
    loaded = load_fn(skill_path="index/tasks")
    assert loaded.content.startswith("---")

    load_file_fn = _get_tool_fn(toolset, "load_skill_file")
    loaded_file = load_file_fn(skill_path="index/tasks", path="references/REF.md")
    assert loaded_file.content == "reference text"


def test_create_skills_toolset_reflects_candidate_overlay() -> None:
    fs = SkillsFS()
    fs.write_text(
        "spaces/SKILL.md",
        "---\nname: spaces\ndescription: old\n---\n# Spaces\n\nBody\n",
    )
    candidate = {
        skill_description_key("spaces"): ComponentValue(
            name=skill_description_key("spaces"), text="new"
        )
    }
    with apply_candidate_to_skills(fs, candidate) as view:
        toolset = create_skills_toolset(view)
        load_fn = _get_tool_fn(toolset, "load_skill")
        loaded = load_fn(skill_path="spaces")
        assert "description: new" in loaded.content


def test_load_skill_accepts_underscore_alias() -> None:
    fs = SkillsFS()
    fs.write_text(
        "roman-numerals/SKILL.md",
        "---\nname: roman-numerals\ndescription: Convert roman numerals\n---\n# Roman\n",
    )

    toolset = create_skills_toolset(fs)
    load_fn = _get_tool_fn(toolset, "load_skill")
    loaded = load_fn(skill_path="roman_numerals")
    assert loaded.skill_path == "roman-numerals"


def test_load_skill_unknown_path_raises_model_retry() -> None:
    fs = SkillsFS()
    fs.write_text(
        "roman-numerals/SKILL.md",
        "---\nname: roman-numerals\ndescription: Convert roman numerals\n---\n# Roman\n",
    )

    toolset = create_skills_toolset(fs)
    load_fn = _get_tool_fn(toolset, "load_skill")
    with pytest.raises(ModelRetry):
        load_fn(skill_path="does-not-exist")


class _BackendStub(SkillsSearchProvider):
    async def search(self, *, query: str, top_k: int, fs, candidate):  # type: ignore[override]
        return [
            SkillSearchResult(
                skill_path="spaces",
                file_path="SKILL.md",
                doc_type="skill_md",
                snippet="stub",
                relevance_score=1.0,
            )
        ][:top_k]


@pytest.mark.asyncio
async def test_create_skills_toolset_with_custom_backend_uses_async_search() -> None:
    fs = SkillsFS()
    fs.write_text(
        "spaces/SKILL.md",
        "---\nname: spaces\ndescription: d\n---\n# Spaces\n",
    )

    toolset = create_skills_toolset(fs, search_backend=_BackendStub(), candidate=None)
    fn = _get_tool_fn(toolset, "search_skills")
    results = await fn(query="x", top_k=3)
    assert results[0].snippet == "stub"




def test_create_skills_toolset_capabilities_gating() -> None:
    fs = SkillsFS()
    fs.write_text("a/SKILL.md", "---\nname: a\ndescription: d\n---\n")

    # Default is READ only
    toolset = create_skills_toolset(fs)
    assert "list_skills" in toolset.tools
    assert "execute_skill_script" not in toolset.tools

    # Custom capabilities (EXECUTE only)
    toolset_exec = create_skills_toolset(fs, capabilities={SkillCapability.EXECUTE})
    assert "list_skills" not in toolset_exec.tools
    assert "execute_skill_script" in toolset_exec.tools

    # Ensure execute throws NotImplementedError
    exec_fn = _get_tool_fn(toolset_exec, "execute_skill_script")
    with pytest.raises(NotImplementedError, match="explicitly disabled"):
        exec_fn(skill_path="a", script_name="test.py", args=[])

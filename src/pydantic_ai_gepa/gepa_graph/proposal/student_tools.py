"""Tools for the student agent to use during execution."""

from __future__ import annotations

import difflib
import hashlib
from typing import Any

from pydantic_ai import FunctionToolset
from pydantic_ai.exceptions import ModelRetry

from ..example_bank import InMemoryExampleBank
from ...skills import OverlayFS, SkillsFS, normalize_rel_path, parse_skill_md
from ...skills.models import (
    SkillCapability,
    SkillFileResult,
    SkillLoadResult,
    SkillSearchResult,
    SkillSummary,
)
from ...skills.search import (
    LocalSkillsSearchProvider,
    SkillsSearchProvider,
    local_search_skills_sync,
)


def create_example_search_tool(
    bank: InMemoryExampleBank,
    instruction: str,
    k: int = 3,
) -> FunctionToolset:
    """Create the example search tool for the student agent.

    This tool allows the student agent to search for relevant few-shot
    examples during execution.

    Args:
        bank: The example bank to search.
        instruction: Description of when to use this tool.
        k: Number of examples to retrieve.
    """
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool(description=instruction)
    def search_examples(query: str) -> str:
        """Search for relevant examples to guide your response.

        Args:
            query: What kind of example are you looking for?
        """
        if len(bank) == 0:
            return "No examples have been added to the example bank yet."
        results = bank.search(query, k=k)
        if not results:
            return "No relevant examples found."

        formatted = []
        for ex in results:
            formatted.append(f"### {ex.title}\n{ex.content}")
        return "\n\n---\n\n".join(formatted)

    return toolset


def create_skills_toolset(
    fs: SkillsFS | OverlayFS,
    *,
    search_backend: SkillsSearchProvider | None = None,
    candidate: dict[str, Any] | None = None,
    capabilities: set[SkillCapability] | None = None,
) -> FunctionToolset:
    """Create a skills toolset backed by a SkillsFS (or overlay)."""
    if capabilities is None:
        capabilities = {SkillCapability.READ}
    toolset: FunctionToolset[None] = FunctionToolset()
    backend = search_backend or LocalSkillsSearchProvider()

    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _resolve_skill_dir(skill_path: str) -> str:
        try:
            normalized = normalize_rel_path(skill_path)
        except Exception as e:
            raise ModelRetry(
                f"Invalid skill_path={skill_path!r}: {e}. Use list_skills() to see valid skill paths."
            ) from e

        candidates = [
            normalized,
            normalized.replace("_", "-"),
            normalized.casefold(),
        ]
        for candidate_path in candidates:
            if candidate_path and fs.exists(f"{candidate_path}/SKILL.md"):
                return candidate_path

        available = sorted(set(fs.iter_skill_dirs()))
        close = difflib.get_close_matches(normalized, available, n=8, cutoff=0.45)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ModelRetry(
            f"Unknown skill_path={skill_path!r}.{hint} Use list_skills() to see valid skill paths."
        )

    def _read_skill_md(skill_path: str) -> tuple[str, str]:
        normalized = _resolve_skill_dir(skill_path)
        path = f"{normalized}/SKILL.md"
        content = fs.read_text(path)
        return content, _hash_text(content)

    if SkillCapability.READ in capabilities:

        @toolset.tool
        def list_skills() -> list[SkillSummary]:
            """List available skills with their name and description."""
            items: list[SkillSummary] = []
            for skill_dir in fs.iter_skill_dirs():
                if not skill_dir:
                    continue
                try:
                    raw, _ = _read_skill_md(skill_dir)
                    skill_md = parse_skill_md(raw)
                except Exception:
                    continue
                items.append(
                    SkillSummary(
                        skill_path=skill_dir,
                        name=skill_md.frontmatter.name,
                        description=skill_md.frontmatter.description,
                    )
                )
            return sorted(items, key=lambda s: s.skill_path)

    if SkillCapability.READ in capabilities:
        if search_backend is None:

            @toolset.tool
            def search_skills(query: str, top_k: int = 8) -> list[SkillSearchResult]:
                """Search skills by simple keyword matching (local fallback)."""
                return local_search_skills_sync(query=query, top_k=top_k, fs=fs)

        else:
            from ...gepa_graph.models import CandidateMap, ComponentValue

            @toolset.tool
            async def search_skills(
                query: str, top_k: int = 8
            ) -> list[SkillSearchResult]:
                """Search skills using the configured backend."""
                candidate_map: CandidateMap | None = None
                if isinstance(candidate, dict):
                    candidate_map = {}
                    for k, v in candidate.items():
                        if isinstance(v, ComponentValue):
                            candidate_map[k] = v
                        elif isinstance(v, str):
                            candidate_map[k] = ComponentValue(name=k, text=v)

                return await backend.search(
                    query=query,
                    top_k=top_k,
                    fs=fs,
                    candidate=candidate_map,
                )

    if SkillCapability.READ in capabilities:

        @toolset.tool
        def load_skill(skill_path: str) -> SkillLoadResult:
            """Load the full SKILL.md for a skill."""
            content, content_hash = _read_skill_md(skill_path)
            normalized = _resolve_skill_dir(skill_path)
            return SkillLoadResult(
                skill_path=normalized,
                content=content,
                content_hash=content_hash,
            )

    if SkillCapability.READ in capabilities:

        @toolset.tool
        def load_skill_file(skill_path: str, path: str) -> SkillFileResult:
            """Load a file within a skill directory."""
            normalized_skill = _resolve_skill_dir(skill_path)
            try:
                normalized_file = normalize_rel_path(path)
            except Exception as e:
                raise ModelRetry(
                    f"Invalid path={path!r}: {e}. Use load_skill(...) to find valid file paths within a skill."
                ) from e
            full_path = f"{normalized_skill}/{normalized_file}"
            if not fs.exists(full_path):
                raise ModelRetry(
                    f"Unknown file path={normalized_file!r} for skill_path={normalized_skill!r}. "
                    "Use load_skill(...) to find valid file paths within a skill."
                )
            content = fs.read_text(full_path)
            return SkillFileResult(
                skill_path=normalized_skill,
                file_path=normalized_file,
                content=content,
                content_hash=_hash_text(content),
            )

    if SkillCapability.EXECUTE in capabilities:

        @toolset.tool
        def run_skill_script(skill_path: str, script_name: str, args: list[str]) -> str:
            """Execute a script associated with a skill."""
            raise NotImplementedError("Script execution is explicitly disabled.")

    # Capability gating
    _allowed_tool_names = set()
    if SkillCapability.READ in capabilities:
        _allowed_tool_names.update(
            ["list_skills", "search_skills", "load_skill", "load_skill_file"]
        )

    if SkillCapability.EXECUTE in capabilities:

        @toolset.tool
        def execute_skill_script(
            skill_path: str, script_name: str, args: list[str]
        ) -> str:
            """Execute a script associated with a skill.

            Args:
                skill_path: The path of the skill containing the script.
                script_name: The name of the script to execute.
                args: Command-line arguments to pass to the script.
            """
            raise NotImplementedError(
                "Script execution is explicitly disabled in this GEPA optimization environment. "
                "This capability is a placeholder for external runtime integration."
            )

        _allowed_tool_names.add("execute_skill_script")

    # Filter the registered tools based on capabilities
    toolset.tools = {k: v for k, v in toolset.tools.items() if k in _allowed_tool_names}

    return toolset


__all__ = ["create_example_search_tool", "create_skills_toolset"]

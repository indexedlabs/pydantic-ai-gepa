"""Pydantic models for skills tool APIs."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SkillCapability(str, Enum):
    """Capabilities that can be enabled for GEPA skills tools."""

    READ = "read"
    EXECUTE = "execute"


class SkillSummary(BaseModel):
    skill_path: str
    name: str
    description: str


class SkillSearchResult(BaseModel):
    skill_path: str
    file_path: str
    doc_type: str = Field(description="e.g. 'skill_md' or 'file'")
    snippet: str | None = None
    relevance_score: float | None = None


class SkillLoadResult(BaseModel):
    skill_path: str
    content: str
    content_hash: str
    files: list[str] = Field(
        default_factory=list,
        description=(
            "Relative file paths available within this skill. "
            "Pass one of these paths to load_skill_file(skill_path, path)."
        ),
    )


class SkillFileResult(BaseModel):
    skill_path: str
    file_path: str
    content: str
    content_hash: str


__all__ = [
    "SkillCapability",
    "SkillSummary",
    "SkillSearchResult",
    "SkillLoadResult",
    "SkillFileResult",
]

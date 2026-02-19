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

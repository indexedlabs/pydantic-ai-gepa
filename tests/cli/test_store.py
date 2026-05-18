"""Tests for `pydantic_ai_gepa.cli.store`."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_gepa.cli.layout import ensure_layout
from pydantic_ai_gepa.cli.store import (
    ComponentStore,
    SlotStatus,
    filename_to_slot,
    introspect_agent,
    slot_to_filename,
)
from pydantic_ai_gepa.skills import SkillsFS


def _make_agent() -> Agent[None, str]:
    agent = Agent(
        TestModel(),
        instructions="You are a helpful test assistant.",
        name="test-agent",
    )

    @agent.tool_plain
    def format_text(text: str, style: str = "plain") -> str:
        """Format the given text.

        Args:
            text: The text to format.
            style: Formatting style.
        """
        return f"{style}: {text}"

    return agent


def _make_skills_fs() -> SkillsFS:
    fs = SkillsFS()
    fs.write_text(
        "month-grid/SKILL.md",
        "---\n"
        "name: month-grid\n"
        "description: Use for visual calendar grids.\n"
        "---\n"
        "# Month Grid\n",
    )
    return fs


def test_slot_filename_round_trip() -> None:
    for slot in [
        "instructions",
        "tool:format_text:description",
        "tool:format_text:param:text",
        "output:my_output:description",
    ]:
        encoded = slot_to_filename(slot)
        assert encoded.endswith(".md")
        assert "/" not in encoded
        assert ":" not in encoded
        assert filename_to_slot(encoded) == slot


def test_slot_filename_rejects_empty() -> None:
    with pytest.raises(ValueError):
        slot_to_filename("")


def test_introspect_agent_finds_instructions_and_tools() -> None:
    agent = _make_agent()
    slots = introspect_agent(agent)
    assert "instructions" in slots
    assert slots["instructions"].startswith("You are a helpful")
    # The function tool's description should appear.
    description_slots = [
        s for s in slots if s.startswith("tool:format_text:description")
    ]
    assert description_slots, slots
    # Parameter descriptions should appear too.
    param_slots = [s for s in slots if s.startswith("tool:format_text:param:")]
    assert any("text" in s for s in param_slots), slots


def test_introspect_agent_with_skills_includes_skill_tool_slots() -> None:
    agent = _make_agent()
    slots = introspect_agent(agent, skills_fs=_make_skills_fs())

    assert "tool:list_skills:description" in slots
    assert "tool:search_skills:description" in slots
    assert "tool:load_skill:description" in slots
    assert "tool:load_skill_file:description" in slots


def test_introspect_agent_with_signature_includes_input_fields() -> None:
    from pydantic_ai_gepa import SignatureAgent

    class Query(BaseModel):
        question: str = Field(description="The question being asked.")

    inner = Agent(TestModel(), instructions="Q&A assistant", name="qa")
    sig_agent = SignatureAgent(inner, input_type=Query)
    slots = introspect_agent(sig_agent)
    # Some signature-related slot must be present (exact key shape depends on
    # input_type.py — the important thing is the signature input is part of the
    # surface).
    assert any("question" in slot for slot in slots), slots


def test_store_write_and_read(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    store = ComponentStore(tmp_path)

    assert store.read("instructions") is None
    store.write("instructions", "New instructions text.")
    assert store.read("instructions") == "New instructions text."
    assert "instructions" in store.list_confirmed_slots()
    assert store.list_staged_slots() == []


def test_store_write_clears_staged_by_default(tmp_path: Path) -> None:
    """Default write() supersedes any staged stub for the slot (confirmation semantics)."""
    ensure_layout(tmp_path)
    store = ComponentStore(tmp_path)
    store.stage("tool:foo:description", "staged")
    store.write("tool:foo:description", "confirmed")
    assert store.read("tool:foo:description") == "confirmed"
    assert store.read_staged("tool:foo:description") is None


def test_store_write_with_clear_staged_false_preserves_stub(tmp_path: Path) -> None:
    """clear_staged=False (used by `init --force`) leaves the staged file alone."""
    ensure_layout(tmp_path)
    store = ComponentStore(tmp_path)
    store.stage("tool:foo:description", "staged")
    store.write("tool:foo:description", "confirmed", clear_staged=False)
    assert store.read("tool:foo:description") == "confirmed"
    assert store.read_staged("tool:foo:description") == "staged"


def test_store_stage_and_confirm(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    store = ComponentStore(tmp_path)

    store.stage("tool:foo:description", "Initial seed text.")
    assert store.read_staged("tool:foo:description") == "Initial seed text."
    assert "tool:foo:description" in store.list_staged_slots()
    assert "tool:foo:description" not in store.list_confirmed_slots()

    # Confirming without an override keeps the seed text.
    store.confirm_staged("tool:foo:description")
    assert store.read("tool:foo:description") == "Initial seed text."
    assert store.read_staged("tool:foo:description") is None

    # Confirming with override replaces the text.
    store.stage("tool:bar:description", "Bar seed.")
    store.confirm_staged(
        "tool:bar:description", override_text="Better description for bar."
    )
    assert store.read("tool:bar:description") == "Better description for bar."


def test_confirm_staged_missing_stub_raises(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    store = ComponentStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="No staged stub"):
        store.confirm_staged("tool:nope:description")


def test_delete_removes_both_copies(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    store = ComponentStore(tmp_path)
    store.stage("instructions", "staged seed")
    store.write("instructions", "confirmed text")
    assert store.delete("instructions") is True
    assert store.read("instructions") is None
    assert store.read_staged("instructions") is None
    # Idempotent
    assert store.delete("instructions") is False


def test_slot_records_status_resolution(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    store = ComponentStore(tmp_path)
    agent = _make_agent()

    records = store.slot_records(agent)
    by_name = {r.name: r for r in records}
    # All slots start as introspected-only.
    assert all(r.status == SlotStatus.INTROSPECTED_ONLY for r in records), records

    # Confirm instructions.
    store.write("instructions", "Override text")
    records = store.slot_records(agent)
    by_name = {r.name: r for r in records}
    assert by_name["instructions"].status == SlotStatus.CONFIRMED
    assert by_name["instructions"].confirmed_text == "Override text"

    # Stage a tool description.
    store.stage("tool:format_text:description", "Stub seed")
    records = store.slot_records(agent)
    by_name = {r.name: r for r in records}
    assert by_name["tool:format_text:description"].status == SlotStatus.STAGED

    # Write a slot that no longer exists on the agent — should appear as orphan.
    store.write("tool:ghost:description", "removed")
    records = store.slot_records(agent)
    by_name = {r.name: r for r in records}
    assert by_name["tool:ghost:description"].status == SlotStatus.ORPHAN


def test_slot_records_keep_confirmed_skill_tool_slots_valid(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    store = ComponentStore(tmp_path)
    agent = _make_agent()
    skills_fs = _make_skills_fs()

    store.write("tool:list_skills:description", "Custom skill list guidance.")

    records = store.slot_records(agent, skills_fs=skills_fs)
    by_name = {r.name: r for r in records}

    assert by_name["tool:list_skills:description"].status == SlotStatus.CONFIRMED
    assert by_name["tool:list_skills:description"].introspected_seed is not None
    assert (
        store.effective_candidate(agent, skills_fs=skills_fs)[
            "tool:list_skills:description"
        ]
        == "Custom skill list guidance."
    )


def test_detect_new_slots_stages_unconfirmed(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    store = ComponentStore(tmp_path)
    agent = _make_agent()

    # First call: every introspected slot becomes staged.
    staged = store.detect_new_slots(agent)
    assert "instructions" in staged
    assert any(s.startswith("tool:format_text:description") for s in staged)

    # Confirm one, leave others staged.
    store.confirm_staged("instructions")

    # Second call: confirmed slot is no longer surfaced; already-staged stay put (no new stagings).
    staged_again = store.detect_new_slots(agent)
    assert "instructions" not in staged_again


def test_effective_candidate_prefers_confirmed_over_seed(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    store = ComponentStore(tmp_path)
    agent = _make_agent()

    # No confirmed files: everything falls back to the introspected seed.
    candidate = store.effective_candidate(agent)
    assert candidate["instructions"].startswith("You are a helpful")

    # Override instructions; other slots stay on seed.
    store.write("instructions", "Custom override.")
    candidate = store.effective_candidate(agent)
    assert candidate["instructions"] == "Custom override."
    # A tool description slot is still present.
    assert any(slot.startswith("tool:format_text:description") for slot in candidate)

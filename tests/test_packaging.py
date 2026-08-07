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
    assert "## Git-native candidates" in text
    assert "GEPA_TRACE_FILE" in text
    assert "git reset --hard <reflection_baseline_commit_sha>" in text


def test_bundled_skill_documents_parallel_lanes() -> None:
    """The installed skill must document the lanes workflow (task-80f) —
    this is the file `gepa init --install-skill` drops into user repos, so
    drift here means agents never learn the protocol."""
    source = (
        importlib.resources.files("pydantic_ai_gepa")
        / "skills"
        / "gepa_optimize"
        / "SKILL.md"
    )
    text = source.read_text(encoding="utf-8")
    assert "## Parallel reflection lanes" in text
    assert "gepa run start --lanes" in text
    assert "gepa next" in text and "gepa ack" in text
    assert "gepa lane lease" in text
    assert "gepa lane continue" in text
    assert "gepa lane reset" in text
    assert "gepa run select" in text
    # The orchestrator loop must name all seven event types.
    for event_type in (
        "lane_ready",
        "verdict",
        "selection_due",
        "merge_opportunity",
        "lane_stalled",
        "budget_low",
        "run_done",
    ):
        assert event_type in text, f"orchestrator loop must handle {event_type}"
    # The reflector contract: packet-only inputs, terminal act is continue.
    assert "packet" in text
    assert "continue_invocation" in text

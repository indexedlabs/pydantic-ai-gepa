"""Tests for `pydantic_ai_gepa.cli.candidates`."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from pydantic_ai_gepa.cli.candidates import Candidate, git_candidate_state
from pydantic_ai_gepa.cli.lanes import _auto_commit_worktree
from pydantic_ai_gepa.cli.layout import candidate_identity_exempt_paths


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "tests@example.com")
    _git(tmp_path, "config", "user.name", "GEPA Tests")
    tracked = tmp_path / "pipeline.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "pipeline.py")
    _git(tmp_path, "commit", "-m", "Seed pipeline")
    return tmp_path


def test_candidate_round_trip(tmp_path: Path) -> None:
    cand = Candidate(id="abc", components={"instructions": "hello"})
    out = tmp_path / "c.json"
    cand.write(out)
    loaded = Candidate.load(out)
    assert loaded.id == "abc"
    assert loaded.components == {"instructions": "hello"}


def test_candidate_load_assigns_stable_id_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps({"components": {"instructions": "txt"}}),
        encoding="utf-8",
    )
    cand = Candidate.load(path)
    assert cand.id.startswith("candidate-")


def test_candidate_load_rejects_bad_json_with_path(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{this is not json", encoding="utf-8")
    with pytest.raises(ValueError, match="broken.json") as excinfo:
        Candidate.load(path)
    # The user-facing message must mention the path AND that it's a JSON parse issue.
    assert "valid JSON" in str(excinfo.value)


def test_candidate_load_rejects_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object at the top level"):
        Candidate.load(path)


def test_candidate_load_missing_components_field(tmp_path: Path) -> None:
    path = tmp_path / "no_components.json"
    path.write_text(json.dumps({"id": "abc"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required 'components'"):
        Candidate.load(path)


def test_candidate_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Candidate.load(tmp_path / "does_not_exist.json")


def test_git_candidate_id_is_short_head_sha_when_clean(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)

    state = git_candidate_state(repo)

    assert state.candidate_id == _git(repo, "rev-parse", "--short=12", "HEAD")
    assert state.commit_sha == _git(repo, "rev-parse", "HEAD")
    assert state.dirty is False
    assert state.dirty_hash is None


def test_git_candidate_dirty_hash_is_stable_and_content_sensitive(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    tracked = repo / "pipeline.py"
    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "prompt.md").write_text("first\n", encoding="utf-8")

    first = git_candidate_state(repo)
    repeated = git_candidate_state(repo)
    (repo / "prompt.md").write_text("second\n", encoding="utf-8")
    changed = git_candidate_state(repo)

    assert first.dirty is True
    assert first.dirty_hash is not None
    assert first.candidate_id == repeated.candidate_id
    assert first.candidate_id.startswith(f"{first.short_commit_sha}-dirty-")
    assert changed.candidate_id != first.candidate_id


def test_git_candidate_can_exclude_cli_run_artifacts(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    run_dir = repo / ".gepa" / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}\n", encoding="utf-8")

    included = git_candidate_state(repo)
    excluded = git_candidate_state(repo, exclude_paths=[run_dir])

    assert included.dirty is True
    assert excluded.dirty is False


def test_notes_are_excluded_from_primary_and_lane_candidate_identity(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    primary_note = repo / ".gepa" / "notes" / "strategy.md"
    primary_note.parent.mkdir(parents=True)
    first = git_candidate_state(
        repo, exclude_paths=candidate_identity_exempt_paths(repo)
    )
    primary_note.write_text(
        "---\nname: strategy\ndescription: First note\n---\nbody one\n",
        encoding="utf-8",
    )
    second = git_candidate_state(
        repo, exclude_paths=candidate_identity_exempt_paths(repo)
    )
    assert second.candidate_id == first.candidate_id

    worktree = repo / "worktrees" / "lane-1"
    worktree.parent.mkdir()
    branch = "gepa/lane/test"
    _git(repo, "worktree", "add", "-b", branch, str(worktree))
    lane_note = worktree / ".gepa" / "notes" / "strategy.md"
    lane_note.parent.mkdir(parents=True)
    lane_before = git_candidate_state(
        worktree, exclude_paths=candidate_identity_exempt_paths(worktree)
    )
    lane_note.write_text(
        "---\nname: strategy\ndescription: Changed note\n---\nbody two\n",
        encoding="utf-8",
    )
    lane_after = git_candidate_state(
        worktree, exclude_paths=candidate_identity_exempt_paths(worktree)
    )
    assert lane_after.candidate_id == lane_before.candidate_id

    (worktree / "pipeline.py").write_text("VALUE = 2\n", encoding="utf-8")
    _auto_commit_worktree(worktree, branch, "lane-1")
    assert ".gepa/notes/strategy.md" not in _git(
        worktree, "show", "--format=", "--name-only", "HEAD"
    )


def test_lane_auto_commit_skips_excluded_only_changes(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    worktree = repo / "worktrees" / "lane-1"
    worktree.parent.mkdir()
    branch = "gepa/lane/test"
    _git(repo, "worktree", "add", "-b", branch, str(worktree))
    journal = worktree / ".gepa" / "journal.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text('{"event": "redirect"}\n', encoding="utf-8")
    before = _git(worktree, "rev-parse", "HEAD")

    assert _auto_commit_worktree(worktree, branch, "lane-1") == before
    assert _git(worktree, "rev-parse", "HEAD") == before

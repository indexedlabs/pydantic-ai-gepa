"""Tests for `pydantic_ai_gepa.cli.candidates`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pydantic_ai_gepa.cli.candidates import Candidate


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

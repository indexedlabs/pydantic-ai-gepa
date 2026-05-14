"""Tests for `pydantic_ai_gepa.cli.dataset`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pydantic_ai_gepa.cli.dataset import case_ids, cases_by_id, load_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_load_dataset_basic(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {"name": "a", "inputs": "in-a", "expected_output": "out-a"},
            {"name": "b", "inputs": "in-b", "expected_output": "out-b"},
        ],
    )
    cases = load_dataset(path)
    assert case_ids(cases) == ["a", "b"]
    assert cases_by_id(cases)["a"].inputs == "in-a"


def test_load_dataset_assigns_default_names_when_missing(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [{"inputs": "x"}, {"inputs": "y"}],
    )
    cases = load_dataset(path)
    assert [c.name for c in cases] == ["case-1", "case-2"]


def test_load_dataset_rejects_duplicate_case_names(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [
            {"name": "dup", "inputs": "first"},
            {"name": "other", "inputs": "middle"},
            {"name": "dup", "inputs": "second"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate case name 'dup'"):
        load_dataset(path)


def test_load_dataset_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"name": "ok"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_dataset(path)


def test_load_dataset_rejects_non_object_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('"a string"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_dataset(path)


def test_load_dataset_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "does_not_exist.jsonl")

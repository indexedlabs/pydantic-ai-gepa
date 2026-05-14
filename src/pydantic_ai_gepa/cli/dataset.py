"""JSONL dataset loader for the gepa CLI.

Each line of `.gepa/dataset.jsonl` is a JSON object with the shape::

    {
        "name": "case-1",            # optional; auto-assigned if absent
        "inputs": "<prompt or any JSON>",
        "expected_output": "...",    # optional
        "metadata": {...}            # optional
    }

The loader produces a list of ``pydantic_evals.Case`` instances suitable for
``evaluate_candidate_dataset`` and the rest of GEPA's machinery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_evals import Case


def load_dataset(path: Path) -> list[Case[Any, Any, Any]]:
    """Load a JSONL dataset into a list of ``Case`` objects."""
    if not path.exists():
        raise FileNotFoundError(f"No dataset at {path}")

    cases: list[Case[Any, Any, Any]] = []
    for idx, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{idx + 1}: not valid JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{idx + 1}: each row must be a JSON object")
        name = row.get("name") or f"case-{idx + 1}"
        inputs = row.get("inputs")
        expected_output = row.get("expected_output")
        metadata = row.get("metadata")
        cases.append(
            Case(
                name=name,
                inputs=inputs,
                expected_output=expected_output,
                metadata=metadata,
            )
        )
    return cases


def case_ids(cases: list[Case[Any, Any, Any]]) -> list[str]:
    """Return the (name-based) case ids in dataset order."""
    return [case.name or f"case-{idx + 1}" for idx, case in enumerate(cases)]


def cases_by_id(
    cases: list[Case[Any, Any, Any]],
) -> dict[str, Case[Any, Any, Any]]:
    return {case.name or f"case-{idx + 1}": case for idx, case in enumerate(cases)}

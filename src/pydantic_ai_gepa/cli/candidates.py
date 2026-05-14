"""Read/write candidate JSON files for the gepa CLI.

A candidate file captures the component overrides applied to the agent during
one evaluation. Schema::

    {
        "id": "candidate-abc123",
        "components": {
            "instructions": "...",
            "tool:foo:description": "..."
        },
        "metadata": {...}     # optional, free-form (origin run/proposal info)
    }
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..gepa_graph.models import CandidateMap, ComponentValue


@dataclass
class Candidate:
    """In-memory representation of a candidate JSON file."""

    id: str
    components: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "components": dict(self.components),
            "metadata": dict(self.metadata),
        }

    def to_candidate_map(self) -> CandidateMap:
        return {
            name: ComponentValue(name=name, text=text)
            for name, text in self.components.items()
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Candidate:
        if "components" not in data:
            raise ValueError("Candidate JSON missing required 'components' field.")
        components = data["components"]
        if not isinstance(components, dict):
            raise ValueError("'components' must be an object mapping slot -> text.")
        return Candidate(
            id=str(data.get("id") or _hash_components(components)),
            components={str(k): str(v) for k, v in components.items()},
            metadata=dict(data.get("metadata", {})),
        )

    @staticmethod
    def load(path: Path) -> Candidate:
        if not path.exists():
            raise FileNotFoundError(f"No candidate file at {path}")
        return Candidate.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def _hash_components(components: dict[str, str]) -> str:
    payload = json.dumps(components, sort_keys=True).encode("utf-8")
    return "candidate-" + hashlib.sha256(payload).hexdigest()[:10]


def candidate_id_from_components(components: dict[str, str]) -> str:
    """Return a stable id derived from the component text content."""
    return _hash_components(components)

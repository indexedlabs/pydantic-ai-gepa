"""Candidate identities and candidate-JSON files for the gepa CLI.

Component candidates use a JSON file containing agent overrides. Git-native
candidates use the current commit plus an optional dirty-tree hash.

Candidate JSON schema::

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
import os
import stat
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..gepa_graph.models import CandidateMap, ComponentValue
from .safe_git import GitExecutableError, safe_repository


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
        raw = path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Candidate file {path} is not valid JSON "
                f"(line {exc.lineno}, column {exc.colno}): {exc.msg}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"Candidate file {path} must contain a JSON object at the top level."
            )
        return Candidate.from_dict(data)

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


class GitCandidateError(RuntimeError):
    """Raised when a git-backed candidate cannot be identified."""


@dataclass(frozen=True, slots=True)
class GitCandidateState:
    """Identity of the repository state used as a git-backed candidate."""

    candidate_id: str
    commit_sha: str
    short_commit_sha: str
    dirty: bool
    dirty_hash: str | None = None


def git_candidate_state(
    root: Path | None = None,
    *,
    exclude_paths: Sequence[Path] = (),
) -> GitCandidateState:
    """Return the HEAD-based identity of the current repository tree.

    Tracked changes are represented by ``git diff HEAD``. Untracked,
    non-ignored files are added by path, mode, and content so identical dirty
    trees have stable identities while any candidate-relevant change produces
    a distinct id.
    """

    from .lane_repositories import candidate_root

    repository = candidate_root((root or Path.cwd()).resolve())
    excluded = _relative_exclusions(repository, exclude_paths)
    pathspecs = [".", *(f":(exclude){path}" for path in excluded)]
    try:
        with safe_repository(repository) as git:
            commit_sha = git.head_oid
            if commit_sha is None:
                raise GitCandidateError("A git candidate requires a committed HEAD.")
            short_commit_sha = (
                git.run(
                    "rev-parse", "--short=12", "HEAD", check=True, capture_output=True
                )
                .stdout.decode()
                .strip()
            )
            tracked_diff = git.run(
                "diff",
                "--binary",
                "--full-index",
                "HEAD",
                "--",
                *pathspecs,
                check=True,
                capture_output=True,
            ).stdout
            untracked_output = git.run(
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *pathspecs,
                check=True,
                capture_output=True,
            ).stdout
    except GitExecutableError as exc:
        raise GitCandidateError(str(exc)) from None
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GitCandidateError(
            f"Could not identify git candidate at {repository}."
        ) from exc
    untracked_paths = [path for path in untracked_output.split(b"\0") if path]
    dirty = bool(tracked_diff or untracked_paths)
    if not dirty:
        return GitCandidateState(
            candidate_id=short_commit_sha,
            commit_sha=commit_sha,
            short_commit_sha=short_commit_sha,
            dirty=False,
        )

    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(tracked_diff)
    for raw_path in sorted(untracked_paths):
        path = repository / os.fsdecode(raw_path)
        file_stat = path.lstat()
        digest.update(b"untracked\0")
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(_git_mode(file_stat.st_mode).encode())
        digest.update(b"\0")
        if stat.S_ISLNK(file_stat.st_mode):
            digest.update(os.fsencode(os.readlink(path)))
        elif stat.S_ISDIR(file_stat.st_mode):
            # Git emits untracked nested repositories as a directory. Include
            # their ordinary files without opening their Git metadata.
            for directory, dirs, files in os.walk(path, followlinks=False):
                dirs[:] = sorted(name for name in dirs if name.casefold() != ".git")
                for name in sorted(files):
                    if name.casefold() == ".git":
                        continue
                    child = Path(directory) / name
                    digest.update(os.fsencode(child.relative_to(path)))
                    digest.update(b"\0")
                    digest.update(_git_mode(child.lstat().st_mode).encode())
                    digest.update(b"\0")
                    digest.update(
                        os.fsencode(os.readlink(child))
                        if child.is_symlink()
                        else child.read_bytes()
                    )
                    digest.update(b"\0")
        else:
            digest.update(path.read_bytes())
        digest.update(b"\0")

    dirty_hash = digest.hexdigest()[:12]
    return GitCandidateState(
        candidate_id=f"{short_commit_sha}-dirty-{dirty_hash}",
        commit_sha=commit_sha,
        short_commit_sha=short_commit_sha,
        dirty=True,
        dirty_hash=dirty_hash,
    )


def _git_mode(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "120000"
    return "100755" if mode & stat.S_IXUSR else "100644"


def _relative_exclusions(repository: Path, paths: Sequence[Path]) -> list[str]:
    excluded: list[str] = []
    for path in paths:
        candidate = path if path.is_absolute() else repository / path
        try:
            relative = candidate.resolve().relative_to(repository)
        except ValueError:
            continue
        excluded.append(relative.as_posix())
    return sorted(excluded)

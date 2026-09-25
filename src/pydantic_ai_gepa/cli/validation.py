"""Keep held-out datasets outside every reflector checkout and its Git objects."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import typer


def validation_dataset_path(
    configured_path: str,
    *,
    project_root: Path,
    candidate_root: Path | None = None,
    allow_missing: bool = False,
) -> Path:
    """Resolve a harness-owned dataset, refusing checkout and historical copies."""
    lexical_path = Path(os.path.abspath(project_root / configured_path))
    path = lexical_path.resolve()
    fix = (
        f"Validation dataset: {path}. "
        "Keep the validation dataset outside the GEPA workspace/repository and point validation_dataset "
        "(or init --validation-dataset) at its absolute path. "
        "Start the workspace from Git history that never contained the validation dataset."
    )
    roots = {project_root.resolve(), (candidate_root or project_root).resolve()}
    repositories: set[Path] = set()
    for root in tuple(roots):
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            repository = Path(result.stdout.strip()).resolve()
            roots.add(repository)
            repositories.add(repository)
        elif "not a git repository" not in result.stderr:
            raise typer.BadParameter(
                "Cannot verify validation dataset isolation. " + fix
            )
    if any(
        path.is_relative_to(root) or lexical_path.is_relative_to(root) for root in roots
    ):
        raise typer.BadParameter(
            "Held-out validation must be outside the reflector checkout. " + fix
        )
    if allow_missing and not path.exists():
        return path
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise typer.BadParameter(
            "Cannot read held-out validation dataset. " + fix
        ) from exc
    for repository in repositories:
        # Checking the object database also catches staged files, deleted files,
        # renamed files, other branches and reflogs. Moving a file is not enough.
        hashed = subprocess.run(
            ["git", "-C", str(repository), "hash-object", "--stdin"],
            input=contents,
            capture_output=True,
        )
        if hashed.returncode:
            raise typer.BadParameter(
                "Cannot verify validation dataset Git history. " + fix
            )
        found = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "cat-file",
                "--batch-check",
            ],
            input=hashed.stdout,
            capture_output=True,
        )
        if found.returncode:
            raise typer.BadParameter(
                "Cannot verify validation dataset Git history. " + fix
            )
        if found.stdout.strip() != hashed.stdout.strip() + b" missing":
            raise typer.BadParameter(
                "Held-out validation is recoverable from Git objects. " + fix
            )
    return path

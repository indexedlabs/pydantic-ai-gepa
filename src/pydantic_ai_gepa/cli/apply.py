"""`gepa apply` — apply a candidate's component overrides to .gepa/components/."""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer

from .candidates import Candidate
from .layout import (
    GepaConfig,
    components_dir,
    config_path,
    insert_repo_root_on_path,
    repo_root,
    resolve_agent,
)
from .store import ComponentStore, introspect_agent


def apply(
    candidate_file: Path = typer.Option(
        ..., "--candidate-file", help="Path to a candidate JSON file."
    ),
    commit: bool = typer.Option(
        False, "--commit", help="Run git add/commit after writing component files."
    ),
    message: str | None = typer.Option(
        None,
        "--message",
        help="Commit message (defaults to 'gepa apply: <candidate_id>'). Inline ok — commit messages are flag content, not optimization content.",
    ),
) -> None:
    """Write a candidate's component overrides into .gepa/components/."""
    cfg = GepaConfig.load(config_path())
    insert_repo_root_on_path()
    agent = resolve_agent(cfg)

    candidate = Candidate.load(candidate_file)
    valid_slots = set(introspect_agent(agent))
    orphans = sorted(slot for slot in candidate.components if slot not in valid_slots)
    if orphans:
        typer.echo(
            "Candidate references slots not present on the current agent:\n  "
            + "\n  ".join(orphans),
            err=True,
        )
        raise typer.Exit(code=1)

    store = ComponentStore()
    for slot, text in candidate.components.items():
        store.write(slot, text)
    typer.echo(f"Wrote {len(candidate.components)} component(s) to {components_dir()}")

    if commit:
        commit_message = message or f"gepa apply: {candidate.id}"
        root = repo_root()
        try:
            components_path = components_dir(root)
            try:
                git_path = str(components_path.relative_to(root))
            except ValueError:
                # Components dir lives outside the repo root (absolute
                # --gepa-dir pointing elsewhere); fall back to the absolute
                # path so git still receives a valid target.
                git_path = str(components_path)
            subprocess.run(
                ["git", "add", git_path],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            typer.echo(f"git not available: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            typer.echo(
                f"git commit failed: {stderr or stdout or exc}",
                err=True,
            )
            raise typer.Exit(code=1) from exc
        typer.echo(f"Committed: {commit_message}")

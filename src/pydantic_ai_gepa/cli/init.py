"""`gepa init` — scaffold .gepa/ and seed components from the agent."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import typer

from .layout import (
    DEFAULT_DATASET_PATH,
    GepaConfigError,
    ensure_layout,
    insert_repo_root_on_path,
    repo_root,
    resolve_module_attr,
    write_default_config,
)
from .store import ComponentStore, introspect_agent


SKILL_DEST_RELATIVE = Path(".agents") / "skills" / "gepa-optimize" / "SKILL.md"


def _install_packaged_skill(*, force: bool) -> Path | None:
    """Copy the bundled gepa-optimize SKILL.md into ``<repo>/.agents/skills/``.

    Returns the destination path if a copy happened. Returns ``None`` when the
    destination already exists and ``force`` is False (caller decides what to
    print). Raises if the packaged source is missing (should never happen for
    an installed wheel).
    """
    source = (
        importlib.resources.files("pydantic_ai_gepa")
        / "skills"
        / "gepa_optimize"
        / "SKILL.md"
    )
    if not source.is_file():
        raise GepaConfigError(
            "Bundled gepa-optimize SKILL.md is missing from the installed package."
        )

    dest = repo_root() / SKILL_DEST_RELATIVE
    if dest.exists() and not force:
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def init(
    agent: str = typer.Option(
        ..., "--agent", help='Agent module ref, e.g. "mypkg.agents:my_agent".'
    ),
    dataset: str = typer.Option(
        DEFAULT_DATASET_PATH,
        "--dataset",
        help="Relative path to the dataset JSONL.",
    ),
    install_skill: bool = typer.Option(
        False,
        "--install-skill",
        help="Also copy the bundled gepa-optimize skill to <repo>/.agents/skills/gepa-optimize/SKILL.md so coding agents discover it automatically.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing .gepa/gepa.toml (and an existing installed skill, when used with --install-skill).",
    ),
) -> None:
    """Bootstrap .gepa/ in the current repo: write gepa.toml + seed components."""
    insert_repo_root_on_path()

    # Sanity-check the agent ref before persisting config.
    try:
        agent_obj = resolve_module_attr(agent, kind="agent")
    except GepaConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    ensure_layout()
    try:
        cfg_path = write_default_config(agent, dataset, force=force)
    except GepaConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    # Seed components/ with the introspected slot values. These are *confirmed*
    # at init time — init is the explicit bootstrap moment.
    store = ComponentStore()
    seeds = introspect_agent(agent_obj)
    for slot, text in seeds.items():
        if store.read(slot) is None:
            store.write(slot, text)

    typer.echo(f"Wrote {cfg_path}")
    typer.echo(f"Seeded {len(seeds)} component slot(s) under {store.components_dir}")

    skill_dest: Path | None = None
    if install_skill:
        try:
            skill_dest = _install_packaged_skill(force=force)
        except GepaConfigError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        if skill_dest is None:
            existing = repo_root() / SKILL_DEST_RELATIVE
            typer.echo(
                f"Skill already installed at {existing}; pass --force to overwrite."
            )
        else:
            typer.echo(f"Installed gepa-optimize skill at {skill_dest}")

    typer.echo(
        "Next steps:\n"
        f"  1. Write dataset cases as JSONL at {dataset}\n"
        "  2. Run `gepa eval --size N` to score the baseline + write the per-case report"
    )

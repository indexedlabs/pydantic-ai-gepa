"""`gepa init` — scaffold .gepa/ and seed components from the agent."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import typer

from .layout import (
    GepaConfigError,
    current_gepa_dirname,
    default_dataset_path,
    ensure_layout,
    insert_repo_root_on_path,
    repo_root,
    resolve_module_attr,
    write_default_config,
)
from .store import ComponentStore, introspect_agent


SKILL_DEST_RELATIVE = Path(".agents") / "skills" / "gepa-optimize" / "SKILL.md"


def _install_packaged_skill(dest: Path, *, force: bool) -> Path | None:
    """Copy the bundled gepa-optimize SKILL.md to ``dest``.

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

    if dest.exists() and not force:
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def init(
    agent: str = typer.Option(
        ..., "--agent", help='Agent module ref, e.g. "mypkg.agents:my_agent".'
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help=(
            "Relative path to the dataset JSONL. "
            "Defaults to `<workspace>/dataset.jsonl` for the active --gepa-dir."
        ),
    ),
    metric: str | None = typer.Option(
        None,
        "--metric",
        help='Optional metric module ref written to gepa.toml as `metric = "..."`, e.g. "mypkg.metrics:my_metric". When omitted, gepa falls back to a substring/equality scorer.',
    ),
    case_factory: str | None = typer.Option(
        None,
        "--case-factory",
        help='Optional case-factory module ref written to gepa.toml as `case_factory = "..."`, e.g. "mypkg.eval:my_case_factory". Use when dataset rows carry deferred references (file paths, Mighty file ids, base64 blobs) that need to be materialized into the agent\'s input model before each rollout.',
    ),
    install_skill: bool = typer.Option(
        False,
        "--install-skill",
        help="Also copy the bundled gepa-optimize skill into the repo so coding agents discover it automatically. Default destination is `.agents/skills/gepa-optimize/SKILL.md`; override with --skill-dest.",
    ),
    skill_dest: Path | None = typer.Option(
        None,
        "--skill-dest",
        help="Override the install path for --install-skill. Can be a directory (the SKILL.md is dropped inside) or a full file path. Defaults to `<repo>/.agents/skills/gepa-optimize/SKILL.md`.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Overwrite an existing .gepa/gepa.toml, re-seed every confirmed slot from agent introspection, "
            "delete orphan confirmed slot files (slots no longer present on the agent), and overwrite the "
            "installed skill (when used with --install-skill). Staged stubs in .gepa/staged/ are preserved so "
            "in-progress confirmations are not silently destroyed."
        ),
    ),
) -> None:
    """Bootstrap the workspace in the current repo: write gepa.toml + seed components."""
    insert_repo_root_on_path()

    # Sanity-check the agent ref (and metric / case_factory refs, when
    # provided) before persisting config.
    try:
        agent_obj = resolve_module_attr(agent, kind="agent")
        if metric:
            resolve_module_attr(metric, kind="metric")
        if case_factory:
            resolve_module_attr(case_factory, kind="case_factory")
    except GepaConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    resolved_dataset = dataset if dataset is not None else default_dataset_path()
    ensure_layout()
    try:
        cfg_path = write_default_config(
            agent,
            resolved_dataset,
            metric=metric,
            case_factory=case_factory,
            force=force,
        )
    except GepaConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    # Seed components/ with the introspected slot values. With --force the
    # existing slot files are re-seeded; without it, existing files win so
    # user edits between init runs survive. Either way, we leave staged stubs
    # alone — those represent in-progress confirmations the user hasn't
    # finalized yet, and silently destroying them on --force was confusing.
    store = ComponentStore()
    seeds = introspect_agent(agent_obj)
    reseeded = 0
    for slot, text in seeds.items():
        if force or store.read(slot) is None:
            store.write(slot, text, clear_staged=False)
            reseeded += 1

    orphans_removed = 0
    if force:
        introspected_names = set(seeds)
        for slot in store.list_confirmed_slots():
            if slot not in introspected_names:
                if store.delete(slot):
                    orphans_removed += 1

    typer.echo(f"Wrote {cfg_path}")
    typer.echo(f"Seeded {len(seeds)} component slot(s) under {store.components_dir}")
    if force and reseeded > 0:
        typer.echo(f"  Re-seeded {reseeded} slot(s) from introspection.")
    if orphans_removed:
        typer.echo(f"  Removed {orphans_removed} orphan slot file(s).")

    skill_installed: Path | None = None
    if install_skill:
        target = _resolve_skill_dest(skill_dest)
        try:
            skill_installed = _install_packaged_skill(target, force=force)
        except GepaConfigError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        if skill_installed is None:
            typer.echo(
                f"Skill already installed at {target}; pass --force to overwrite."
            )
        else:
            typer.echo(f"Installed gepa-optimize skill at {skill_installed}")

    dirname = current_gepa_dirname()
    workspace_hint = "" if dirname == ".gepa" else f" --gepa-dir {dirname}"
    typer.echo(
        "Next steps:\n"
        f"  1. Write dataset cases as JSONL at {resolved_dataset}\n"
        f"  2. Run `gepa{workspace_hint} eval --size N` to score the baseline + write the per-case report"
    )


def _resolve_skill_dest(dest: Path | None) -> Path:
    """Resolve ``--skill-dest`` (or the default) to a concrete SKILL.md path.

    If the caller passes a directory (or a path that already exists as a
    directory), the bundled ``SKILL.md`` is dropped inside it. Otherwise the
    path is taken literally as the destination file.
    """
    if dest is None:
        return repo_root() / SKILL_DEST_RELATIVE
    if dest.is_dir() or dest.suffix == "":
        return dest / "SKILL.md"
    return dest

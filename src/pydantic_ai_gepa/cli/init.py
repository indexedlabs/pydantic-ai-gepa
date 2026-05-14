"""`gepa init` — scaffold .gepa/ and seed components from the agent."""

from __future__ import annotations

import typer

from .layout import (
    DEFAULT_DATASET_PATH,
    GepaConfigError,
    ensure_layout,
    insert_repo_root_on_path,
    resolve_module_attr,
    write_default_config,
)
from .store import ComponentStore, introspect_agent


def init(
    agent: str = typer.Option(
        ..., "--agent", help='Agent module ref, e.g. "mypkg.agents:my_agent".'
    ),
    dataset: str = typer.Option(
        DEFAULT_DATASET_PATH,
        "--dataset",
        help="Relative path to the dataset JSONL.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite an existing .gepa/gepa.toml."
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
    typer.echo(
        "Next steps:\n"
        f"  1. Write dataset cases as JSONL at {dataset}\n"
        "  2. Run `gepa eval --size N` to score the baseline + write the per-case report"
    )

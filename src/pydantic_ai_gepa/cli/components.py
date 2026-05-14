"""`gepa components` — inspect and mutate optimizable components.

The four verbs in this group implement the surface described by
pydanticaigepa-spec-973 and enforce the content-file rule from
pydanticaigepa-dec-dmk: ``set`` and ``confirm`` accept text only via
``--content-file`` (or ``-`` for stdin), never inline.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ._io import read_content_file, write_content_file
from .layout import GepaConfig, config_path, insert_repo_root_on_path, resolve_agent
from .store import ComponentStore, SlotRecord


app = typer.Typer(
    no_args_is_help=True, help="Inspect and mutate optimizable components."
)


def _load_agent():
    cfg_path = config_path()
    cfg = GepaConfig.load(cfg_path)
    insert_repo_root_on_path()
    return resolve_agent(cfg)


def _records_for_output(records: list[SlotRecord]) -> list[dict[str, object]]:
    return [
        {
            "name": r.name,
            "status": r.status.value,
            "has_confirmed": r.confirmed_text is not None,
            "has_staged": r.staged_text is not None,
            "has_seed": r.introspected_seed is not None,
        }
        for r in records
    ]


def _format_table(records: list[SlotRecord]) -> str:
    if not records:
        return "(no components)"
    name_width = max(len(r.name) for r in records)
    status_width = max(len(r.status.value) for r in records)
    lines = [
        f"{'SLOT'.ljust(name_width)}  {'STATUS'.ljust(status_width)}  CONFIRMED  STAGED  SEED",
    ]
    for r in records:
        lines.append(
            f"{r.name.ljust(name_width)}  "
            f"{r.status.value.ljust(status_width)}  "
            f"{('yes' if r.confirmed_text is not None else 'no').ljust(9)}  "
            f"{('yes' if r.staged_text is not None else 'no').ljust(6)}  "
            f"{'yes' if r.introspected_seed is not None else 'no'}"
        )
    return "\n".join(lines)


def _format_tsv(records: list[SlotRecord]) -> str:
    lines = ["slot\tstatus\thas_confirmed\thas_staged\thas_seed"]
    for r in records:
        lines.append(
            "\t".join(
                [
                    r.name,
                    r.status.value,
                    "1" if r.confirmed_text is not None else "0",
                    "1" if r.staged_text is not None else "0",
                    "1" if r.introspected_seed is not None else "0",
                ]
            )
        )
    return "\n".join(lines)


@app.command("list")
def list_(
    format_: str = typer.Option(
        "table", "--format", help="table | json | tsv", show_default=True
    ),
) -> None:
    """List the canonical slot set with per-slot status."""
    agent = _load_agent()
    store = ComponentStore()
    records = store.slot_records(agent)

    if format_ == "json":
        typer.echo(json.dumps(_records_for_output(records), indent=2))
    elif format_ == "tsv":
        typer.echo(_format_tsv(records))
    elif format_ == "table":
        typer.echo(_format_table(records))
    else:
        typer.echo(
            f"Unknown --format {format_!r}; expected one of: table, json, tsv",
            err=True,
        )
        raise typer.Exit(code=2)


@app.command("show")
def show(
    slot: str = typer.Argument(..., help="Component slot name."),
    output_file: Path | None = typer.Option(
        None,
        "--output-file",
        help="Write to a file. Use - or omit for stdout.",
    ),
    source: str = typer.Option(
        "auto",
        "--source",
        help="auto (default: confirmed > staged > seed), confirmed, staged, or seed",
        show_default=True,
    ),
) -> None:
    """Print or write the current text for a slot.

    Resolution order with ``--source auto``: confirmed > staged > introspected seed.
    """
    store = ComponentStore()

    if source == "confirmed":
        text = store.read(slot)
    elif source == "staged":
        text = store.read_staged(slot)
    elif source == "seed":
        agent = _load_agent()
        from .store import introspect_agent

        text = introspect_agent(agent).get(slot)
    elif source == "auto":
        text = store.read(slot)
        if text is None:
            text = store.read_staged(slot)
        if text is None:
            agent = _load_agent()
            from .store import introspect_agent

            text = introspect_agent(agent).get(slot)
    else:
        typer.echo(
            f"Unknown --source {source!r}; expected one of: auto, confirmed, staged, seed",
            err=True,
        )
        raise typer.Exit(code=2)

    if text is None:
        typer.echo(f"No text available for slot {slot!r} (source={source}).", err=True)
        raise typer.Exit(code=1)

    write_content_file(output_file, text)


@app.command("set")
def set_(
    slot: str = typer.Argument(..., help="Component slot name."),
    content_file: Path = typer.Option(
        ...,
        "--content-file",
        help="Path to a file with the new content. Use - for stdin.",
    ),
) -> None:
    """Write a new confirmed value for a slot.

    Per pydanticaigepa-dec-dmk, content comes from a file or stdin — never
    inline — to avoid shell-escape bugs on multi-line text.
    """
    text = read_content_file(content_file)
    store = ComponentStore()
    path = store.write(slot, text)
    typer.echo(f"Wrote {len(text)} chars to {path}")


@app.command("confirm")
def confirm(
    slot: str = typer.Argument(
        ..., help="Component slot name to promote from staged to confirmed."
    ),
    content_file: Path | None = typer.Option(
        None,
        "--content-file",
        help="Optional override (file or - for stdin); defaults to the staged seed.",
    ),
) -> None:
    """Promote a staged stub to a confirmed value (with optional override)."""
    store = ComponentStore()
    override = read_content_file(content_file) if content_file is not None else None
    try:
        path = store.confirm_staged(slot, override_text=override)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Confirmed {slot} -> {path}")


__all__ = ["app"]

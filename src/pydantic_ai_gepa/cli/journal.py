"""`gepa journal` — read and append the Reflection Ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from ._io import read_content_file
from .layout import journal_path
from .runs import utc_now_iso


app = typer.Typer(no_args_is_help=True, help="Read and append the Reflection Ledger.")


def _append_journal_entry(entry: dict[str, Any]) -> Path:
    path = journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


def _iter_journal(limit: int) -> list[dict[str, Any]]:
    path = journal_path()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    if limit > 0:
        return rows[-limit:]
    return rows


@app.command("append")
def append(
    content_file: Path = typer.Option(
        ...,
        "--content-file",
        help="Path to the entry body (text or markdown). Use - for stdin.",
    ),
    strategy: str | None = typer.Option(
        None, "--strategy", help="Short inline tag describing the entry."
    ),
) -> None:
    """Append a new entry to .gepa/journal.jsonl."""
    body = read_content_file(content_file)
    entry: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "content": body,
    }
    if strategy:
        entry["strategy"] = strategy
    path = _append_journal_entry(entry)
    typer.echo(f"Appended entry to {path}")


@app.command("show")
def show(
    limit: int = typer.Option(20, "--limit", min=0),
    format_: str = typer.Option(
        "json", "--format", help="json | tsv", show_default=True
    ),
) -> None:
    """Tail the most recent journal entries."""
    rows = _iter_journal(limit)
    if format_ == "json":
        typer.echo(json.dumps(rows, indent=2))
    elif format_ == "tsv":
        lines = ["timestamp\tstrategy\tcontent"]
        for row in rows:
            content = str(row.get("content", "")).replace("\t", " ").replace("\n", " ")
            lines.append(
                "\t".join(
                    [
                        str(row.get("timestamp", "")),
                        str(row.get("strategy", "")),
                        content,
                    ]
                )
            )
        typer.echo("\n".join(lines))
    else:
        typer.echo(
            f"Unknown --format {format_!r}; expected one of: json, tsv",
            err=True,
        )
        raise typer.Exit(code=2)


__all__ = ["app"]

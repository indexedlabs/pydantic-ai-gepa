"""`gepa pareto` — show the Pareto front or full history."""

from __future__ import annotations

import json

import typer

from .layout import latest_run_id
from .runs import ParetoLog, ParetoRow


_TSV_HEADER = (
    "candidate_id",
    "commit_sha",
    "minibatch_id",
    "mean_score",
    "status",
    "summary",
)


def _resolve_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    latest = latest_run_id()
    if latest is None:
        raise typer.Exit(code=1)
    return latest


def _rows_for_output(rows: list[ParetoRow]) -> list[dict[str, object]]:
    return [r.to_dict() for r in rows]


def _format_tsv(rows: list[ParetoRow]) -> str:
    lines = ["\t".join(_TSV_HEADER)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.candidate_id,
                    row.commit_sha or "",
                    row.minibatch_id,
                    f"{row.mean_score:.4f}",
                    row.status,
                    row.summary.replace("\t", " ").replace("\n", " "),
                ]
            )
        )
    return "\n".join(lines)


def pareto(
    run_id: str | None = typer.Option(None, "--run-id"),
    format_: str = typer.Option(
        "json", "--format", help="json | tsv", show_default=True
    ),
    front: bool = typer.Option(
        True,
        "--front/--all",
        help="--front (default) shows current Pareto front; --all shows full history.",
    ),
) -> None:
    """Show the Pareto front or full history for a run."""
    try:
        active_run = _resolve_run_id(run_id)
    except typer.Exit:
        typer.echo(
            "No runs found under .gepa/runs/. Run `gepa propose` or `gepa eval` first.",
            err=True,
        )
        raise

    log = ParetoLog(active_run)
    rows = log.front() if front else log.iter_rows()

    if format_ == "json":
        typer.echo(json.dumps(_rows_for_output(rows), indent=2))
    elif format_ == "tsv":
        typer.echo(_format_tsv(rows))
    else:
        typer.echo(
            f"Unknown --format {format_!r}; expected one of: json, tsv",
            err=True,
        )
        raise typer.Exit(code=2)

"""Typer-based `gepa` CLI for external-reflection mode.

See pydanticaigepa-spec-973 for the surface contract and the related
decisions (content-file enforcement, skill packaging, stage-and-confirm,
git-SHA candidate identity).
"""

from __future__ import annotations

import typer

from . import apply as apply_cmd
from . import components as components_cmd
from . import eval as eval_cmd
from . import init as init_cmd
from . import journal as journal_cmd
from . import pareto as pareto_cmd
from .layout import load_dotenv

app = typer.Typer(
    name="gepa",
    help="External-reflection CLI for pydantic-ai-gepa. Coding agents drive the loop; the library handles minibatches, Pareto, evaluation, and reflection.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)


@app.callback()
def _gepa_root(
    no_dotenv: bool = typer.Option(
        False,
        "--no-dotenv",
        help="Skip auto-loading .env from the repo root.",
    ),
) -> None:
    """Load .env once before any verb runs.

    The user's agent module often resolves to a real provider (openai, anthropic,
    etc.) that eagerly constructs a client at import time. Auto-loading .env
    means `gepa components list`, `gepa eval`, etc. just work in a repo that
    already has API keys configured.
    """
    if not no_dotenv:
        load_dotenv()


app.command(name="init", help="Scaffold .gepa/ and seed components from the agent.")(
    init_cmd.init
)
app.command(
    name="eval",
    help="Evaluate the current baseline (default) or an explicit --candidate-file.",
)(eval_cmd.eval_)
app.command(
    name="apply", help="Apply a candidate's component overrides into .gepa/components/."
)(apply_cmd.apply)
app.command(name="pareto", help="Show Pareto front or full history (json or tsv).")(
    pareto_cmd.pareto
)

app.add_typer(
    components_cmd.app,
    name="components",
    help="Inspect and mutate optimizable components.",
)
app.add_typer(
    journal_cmd.app, name="journal", help="Read and append the Reflection Ledger."
)


__all__ = ["app"]

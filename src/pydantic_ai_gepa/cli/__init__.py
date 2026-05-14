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
from . import propose as propose_cmd

app = typer.Typer(
    name="gepa",
    help="External-reflection CLI for pydantic-ai-gepa. Coding agents drive the loop; the library handles minibatches, Pareto, evaluation, and reflection.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

app.command(name="init", help="Scaffold .gepa/ and seed components from the agent.")(
    init_cmd.init
)
app.command(
    name="propose",
    help="Run one optimization step: eval baseline, reflect, emit proposal.",
)(propose_cmd.propose)
app.command(
    name="eval", help="Evaluate a candidate JSON against the configured dataset."
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

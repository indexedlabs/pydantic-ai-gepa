"""Drive the two-process protocol synchronously with the fake-model CLI runner."""

from typer.testing import CliRunner
from pydantic_ai_gepa.cli import app


def scored_continue(*argv: str):
    nomination = CliRunner().invoke(app, [*argv, "--wait-secs", "0"])
    if nomination.exit_code or '"status": "pending"' not in nomination.output:
        return nomination
    run_id = argv[argv.index("--run-id") + 1]
    served = CliRunner().invoke(app, ["harness", "serve", "--run-id", run_id, "--once"])
    if served.exit_code:
        return served
    return CliRunner().invoke(app, [*argv, "--wait-secs", "0"])

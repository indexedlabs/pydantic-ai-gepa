"""End-to-end test for `gepa pareto`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.layout import ensure_layout, new_run_id, run_dir
from pydantic_ai_gepa.cli.runs import ParetoLog, ParetoRow, utc_now_iso


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.chdir(tmp_path)
    ensure_layout(tmp_path)
    yield tmp_path


def _make_row(
    name: str, scores: dict[str, float], status: str = "evaluated"
) -> ParetoRow:
    return ParetoRow(
        candidate_id=name,
        commit_sha="aaaa1111",
        component_overrides_id=f"comp-{name}",
        minibatch_id="mb-1",
        per_case_scores=scores,
        mean_score=sum(scores.values()) / len(scores) if scores else 0.0,
        status=status,
        summary=f"row {name}",
        timestamp=utc_now_iso(),
    )


def _seed_pareto(repo: Path, run_id: str, rows: list[ParetoRow]) -> None:
    run_dir(run_id, repo).mkdir(parents=True, exist_ok=True)
    log = ParetoLog(run_id, repo)
    for row in rows:
        log.append(row)


def _run(*argv: str) -> object:
    return CliRunner().invoke(gepa_app, list(argv))


def test_pareto_default_is_full_history(repo: Path) -> None:
    """Default is `--all` — full chronological history, not just the front."""
    run = new_run_id()
    _seed_pareto(
        repo,
        run,
        [
            _make_row("dominated", {"a": 0.1, "b": 0.2}),
            _make_row("dominator", {"a": 0.5, "b": 0.7}),
        ],
    )
    result = _run("pareto", "--run-id", run)
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert {row["candidate_id"] for row in rows} == {"dominated", "dominator"}


def test_pareto_front_keeps_only_dominators(repo: Path) -> None:
    run = new_run_id()
    _seed_pareto(
        repo,
        run,
        [
            _make_row("dominated", {"a": 0.1, "b": 0.2}),
            _make_row("dominator", {"a": 0.5, "b": 0.7}),
        ],
    )
    result = _run("pareto", "--run-id", run, "--front")
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert {row["candidate_id"] for row in rows} == {"dominator"}


def test_pareto_tsv_format(repo: Path) -> None:
    run = new_run_id()
    _seed_pareto(
        repo,
        run,
        [
            _make_row("c1", {"a": 0.4, "b": 0.6}),
            _make_row("c2", {"a": 0.7, "b": 0.3}),
        ],
    )
    result = _run("pareto", "--run-id", run, "--format", "tsv", "--all")
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0].split("\t") == [
        "candidate_id",
        "commit_sha",
        "minibatch_id",
        "mean_score",
        "status",
        "summary",
    ]
    body_ids = [line.split("\t")[0] for line in lines[1:]]
    assert sorted(body_ids) == ["c1", "c2"]


def test_pareto_picks_latest_run_when_no_run_id(repo: Path) -> None:
    # Hand-pick run ids so the lexicographic sort is unambiguous (new_run_id is
    # second-precision and rapid successive calls can collide).
    older = "20260101T000000Z-aaaaaaaa"
    newer = "20260102T000000Z-bbbbbbbb"
    _seed_pareto(repo, older, [_make_row("old", {"a": 0.5})])
    _seed_pareto(repo, newer, [_make_row("new", {"a": 0.6})])

    result = _run("pareto")  # default is now --all
    assert result.exit_code == 0, result.output
    ids = {row["candidate_id"] for row in json.loads(result.output)}
    assert "new" in ids


def test_pareto_errors_when_no_runs(repo: Path) -> None:
    result = _run("pareto")
    assert result.exit_code == 1
    assert "No runs found" in result.output


def test_pareto_rejects_unknown_format(repo: Path) -> None:
    run = new_run_id()
    _seed_pareto(repo, run, [_make_row("c1", {"a": 1.0})])
    result = _run("pareto", "--run-id", run, "--format", "yaml")
    assert result.exit_code == 2
    assert "Unknown --format" in result.output

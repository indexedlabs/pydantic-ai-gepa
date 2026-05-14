"""End-to-end test for `gepa eval`."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.candidates import Candidate
from pydantic_ai_gepa.cli.layout import (
    ensure_layout,
    latest_run_id,
    pareto_log_path,
    write_default_config,
)
from pydantic_ai_gepa.cli.runs import ParetoLog


# A tiny agent module written to disk. The TestModel always echoes "Paris" so
# we can drive metric outcomes by toggling expected_output between matches and
# non-matches.
AGENT_MODULE_SOURCE = textwrap.dedent("""
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(
        TestModel(custom_output_text="Paris"),
        instructions="You are a geography expert.",
        name="geo-agent",
    )
""").lstrip()


DATASET_LINES = [
    {
        "name": "case-paris",
        "inputs": "What is the capital of France?",
        "expected_output": "Paris",
    },
    {
        "name": "case-paris-substr",
        "inputs": "Capital of France?",
        "expected_output": "paris",
    },
    {"name": "case-fail", "inputs": "Capital of Germany?", "expected_output": "Berlin"},
]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    module_dir = tmp_path / "agent_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").touch()
    (module_dir / "agents.py").write_text(AGENT_MODULE_SOURCE, encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    ensure_layout(tmp_path)
    write_default_config("agent_pkg.agents:agent", root=tmp_path)

    dataset_path = tmp_path / ".gepa" / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in DATASET_LINES) + "\n",
        encoding="utf-8",
    )

    yield tmp_path

    for name in list(sys.modules):
        if name.startswith("agent_pkg"):
            sys.modules.pop(name, None)


def _candidate_file(tmp_path: Path, components: dict[str, str]) -> Path:
    cand = Candidate(id="cand-test-1", components=components)
    path = tmp_path / "candidate.json"
    cand.write(path)
    return path


def _run(*argv: str) -> object:
    return CliRunner().invoke(gepa_app, list(argv))


def test_eval_writes_pareto_row(repo: Path, tmp_path: Path) -> None:
    cand_path = _candidate_file(tmp_path, {"instructions": "Override text."})

    result = _run(
        "eval",
        "--candidate-file",
        str(cand_path),
        "--size",
        "3",
        "--seed",
        "0",
        "--epoch",
        "0",
    )
    assert result.exit_code == 0, result.output

    # Output should contain per-case scores + summary as JSONL.
    lines = [line for line in result.output.splitlines() if line.startswith("{")]
    summary_line = next(line for line in lines if '"summary"' in line)
    summary = json.loads(summary_line)["summary"]
    assert summary["candidate_id"] == "cand-test-1"
    assert summary["n_cases"] == 3

    # Pareto log must have been appended to.
    run_id = latest_run_id(repo)
    assert run_id is not None
    pareto_path = pareto_log_path(run_id, repo)
    assert pareto_path.exists()
    rows = ParetoLog(run_id, repo).iter_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row.candidate_id == "cand-test-1"
    assert row.status == "evaluated"
    assert row.minibatch_id == summary["minibatch_id"]
    assert set(row.per_case_scores) == {"case-paris", "case-paris-substr", "case-fail"}
    # case-paris and case-paris-substr both contain "Paris" in the output; case-fail does not.
    assert row.per_case_scores["case-paris"] > 0
    assert row.per_case_scores["case-paris-substr"] > 0
    assert row.per_case_scores["case-fail"] == pytest.approx(0.0)


def test_eval_writes_output_file(repo: Path, tmp_path: Path) -> None:
    cand_path = _candidate_file(tmp_path, {"instructions": "Custom."})
    out_path = tmp_path / "results.jsonl"

    result = _run(
        "eval",
        "--candidate-file",
        str(cand_path),
        "--size",
        "2",
        "--seed",
        "0",
        "--epoch",
        "0",
        "--output-file",
        str(out_path),
    )
    assert result.exit_code == 0, result.output

    text = out_path.read_text(encoding="utf-8")
    assert '"summary"' in text


def test_eval_loads_existing_minibatch(repo: Path, tmp_path: Path) -> None:
    cand_path = _candidate_file(tmp_path, {"instructions": "Custom."})

    # First run creates a minibatch
    first = _run(
        "eval",
        "--candidate-file",
        str(cand_path),
        "--size",
        "2",
        "--seed",
        "0",
        "--epoch",
        "0",
    )
    assert first.exit_code == 0, first.output
    first_summary = next(
        json.loads(line)
        for line in first.output.splitlines()
        if line.startswith("{") and '"summary"' in line
    )
    mb_id = first_summary["summary"]["minibatch_id"]

    # Second run reuses that minibatch — same case selection.
    second = _run(
        "eval",
        "--candidate-file",
        str(cand_path),
        "--run-id",
        first_summary["summary"]["run_id"],
        "--minibatch-id",
        mb_id,
    )
    assert second.exit_code == 0, second.output
    second_summary = next(
        json.loads(line)
        for line in second.output.splitlines()
        if line.startswith("{") and '"summary"' in line
    )
    assert second_summary["summary"]["minibatch_id"] == mb_id


def test_eval_rejects_missing_candidate_file(repo: Path, tmp_path: Path) -> None:
    result = _run("eval", "--candidate-file", str(tmp_path / "does_not_exist.json"))
    assert result.exit_code != 0


def test_eval_threshold_flag_filters_report(repo: Path, tmp_path: Path) -> None:
    """--threshold controls which cases land in the per-case report."""
    cand_path = _candidate_file(tmp_path, {"instructions": "Custom."})

    # Threshold 0.0: no case is a failure → report is the "all passed" body.
    result_lenient = _run(
        "eval",
        "--candidate-file",
        str(cand_path),
        "--size",
        "3",
        "--seed",
        "0",
        "--epoch",
        "0",
        "--threshold",
        "0.0",
    )
    assert result_lenient.exit_code == 0, result_lenient.output
    lenient_summary = next(
        json.loads(line)
        for line in result_lenient.output.splitlines()
        if line.startswith("{") and '"summary"' in line
    )
    report_path_lenient = Path(lenient_summary["summary"]["report_path"])
    lenient_text = report_path_lenient.read_text(encoding="utf-8")
    assert "Every case in this minibatch passed" in lenient_text

    # Threshold 1.0 (strict): every case scores < 1.0 here because TestModel's
    # output text contains, but does not equal, the expected strings — so the
    # report lists them as failures.
    result_strict = _run(
        "eval",
        "--candidate-file",
        str(cand_path),
        "--size",
        "3",
        "--seed",
        "0",
        "--epoch",
        "0",
        "--threshold",
        "1.0",
    )
    assert result_strict.exit_code == 0, result_strict.output
    strict_summary = next(
        json.loads(line)
        for line in result_strict.output.splitlines()
        if line.startswith("{") and '"summary"' in line
    )
    strict_text = Path(strict_summary["summary"]["report_path"]).read_text(
        encoding="utf-8"
    )
    assert "case(s) underperformed" in strict_text
    assert "case-fail" in strict_text

"""End-to-end tests for the managed `gepa run` controller."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest
from click.testing import Result
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.candidates import candidate_id_from_components
from pydantic_ai_gepa.cli.layout import final_report_path, run_state_path
from pydantic_ai_gepa.evaluation import EvaluationRecord
from pydantic_ai_gepa.types import RolloutOutput


AGENT_MODULE_SOURCE = textwrap.dedent("""
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(
        TestModel(custom_output_text="Paris"),
        instructions="You are a geography assistant.",
        name="geo",
    )
""").lstrip()


DATASET = [
    {"name": "case-paris", "inputs": "?", "expected_output": "Paris"},
    {"name": "case-berlin", "inputs": "?", "expected_output": "Berlin"},
]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    module_dir = tmp_path / "agent_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").touch()
    (module_dir / "agents.py").write_text(AGENT_MODULE_SOURCE, encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    init_result = _run("init", "--agent", "agent_pkg.agents:agent")
    assert init_result.exit_code == 0, init_result.output
    (tmp_path / ".gepa" / "dataset.jsonl").write_text(
        "\n".join(json.dumps(row) for row in DATASET) + "\n", encoding="utf-8"
    )

    yield tmp_path

    for name in list(sys.modules):
        if name.startswith("agent_pkg"):
            sys.modules.pop(name, None)


def _run(*argv: str) -> Result:
    return CliRunner().invoke(gepa_app, list(argv))


def _run_payload(output: str) -> dict[str, object]:
    line = next(
        line
        for line in reversed(output.splitlines())
        if line.startswith("{") and '"run"' in line
    )
    payload = json.loads(line)
    return payload["run"]


def _fail_rollout_calls(
    monkeypatch: pytest.MonkeyPatch, call_numbers: set[int]
) -> None:
    from pydantic_ai_gepa.cli import run as run_module

    original = run_module.run_eval_once
    call_count = 0

    def wrapped(**kwargs):
        nonlocal call_count
        call_count += 1
        outcome = original(**kwargs)
        if call_count not in call_numbers:
            return outcome
        record = outcome.records[0]
        outcome.records[0] = EvaluationRecord(
            case_id=record.case_id,
            score=record.score,
            feedback=record.feedback,
            payload={
                **record.payload,
                "output": RolloutOutput.from_error(
                    RuntimeError(f"provider unavailable on call {call_count}"),
                    kind="system",
                ),
            },
        )
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", wrapped)


def test_managed_run_pauses_for_reflection_and_writes_trace_paths(repo: Path) -> None:
    result = _run("run", "start", "--size", "2", "--max-iterations", "3")

    assert result.exit_code == 0, result.output
    payload = _run_payload(result.output)
    assert payload["status"] == "paused_for_reflection"
    assert payload["iterations"] == 1
    assert payload["next_command"] == f"gepa run continue --run-id {payload['run_id']}"
    assert Path(str(payload["reflection_baseline_report_path"])).exists()
    assert Path(str(payload["reflection_baseline_trace_path"])).exists()
    assert run_state_path(str(payload["run_id"]), repo).exists()


def test_continue_reports_equivalent_when_candidate_does_not_change(
    repo: Path,
) -> None:
    start = _run("run", "start", "--size", "2", "--max-iterations", "3")
    run_id = str(_run_payload(start.output)["run_id"])

    result = _run("run", "continue", "--run-id", run_id)

    assert result.exit_code == 0, result.output
    assert "equivalent" in result.output
    payload = _run_payload(result.output)
    assert payload["status"] == "paused_after_candidate_eval"
    assert payload["iterations"] == 2
    comparison = payload["last_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["verdict"] == "equivalent"
    assert comparison["recommendation"] == "discard_no_material_change"
    assert comparison["delta"] == pytest.approx(0.0)
    assert Path(str(comparison["candidate_trace_path"])).exists()


def test_managed_run_repeats_baseline_and_candidate_on_saved_minibatch(
    repo: Path,
) -> None:
    start = _run(
        "run",
        "start",
        "--size",
        "2",
        "--max-iterations",
        "10",
        "--acceptance-repetitions",
        "3",
        "--acceptance-max-repetitions",
        "5",
    )

    assert start.exit_code == 0, start.output
    start_payload = _run_payload(start.output)
    assert start_payload["status"] == "paused_for_reflection"
    assert start_payload["iterations"] == 5
    baseline_samples = start_payload["reflection_baseline_samples"]
    baseline_report_paths = start_payload["reflection_baseline_report_paths"]
    assert isinstance(baseline_samples, list)
    assert isinstance(baseline_report_paths, list)
    assert len(baseline_samples) == 5
    assert len(set(baseline_report_paths)) == 5

    result = _run("run", "continue", "--run-id", str(start_payload["run_id"]))

    assert result.exit_code == 0, result.output
    payload = _run_payload(result.output)
    comparison = payload["last_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["verdict"] == "equivalent"
    assert comparison["baseline_sample_count"] == 3
    assert comparison["candidate_sample_count"] == 3
    candidate_report_paths = comparison["candidate_report_paths"]
    assert isinstance(candidate_report_paths, list)
    assert len(set(candidate_report_paths)) == 3
    assert payload["iterations"] == 8


def test_baseline_rollout_failure_pauses_without_installing_baseline(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_rollout_calls(monkeypatch, {1})

    result = _run("run", "start", "--size", "2", "--max-iterations", "4")

    assert result.exit_code == 0, result.output
    payload = _run_payload(result.output)
    failed_minibatch_id = payload["last_minibatch_id"]
    assert payload["status"] == "paused_after_infrastructure_error"
    assert payload["best_candidate_id"] is None
    assert payload["reflection_baseline_samples"] == []
    comparison = payload["last_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["outcome"] == "infrastructure_failure"
    assert comparison["selectable"] is False
    assert comparison["verdict"] is None
    assert comparison["phase"] == "baseline"
    assert comparison["evaluation_error_count"] == 1
    retried = _run("run", "continue", "--run-id", str(payload["run_id"]))
    assert retried.exit_code == 0, retried.output
    retry_payload = _run_payload(retried.output)
    assert retry_payload["status"] == "paused_for_reflection"
    assert retry_payload["best_candidate_id"] is not None
    retry_samples = retry_payload["reflection_baseline_samples"]
    assert isinstance(retry_samples, list)
    assert len(retry_samples) == 1
    assert retry_payload["reflection_minibatch_id"] == failed_minibatch_id
    assert retry_payload["last_comparison"] is None


def test_budget_edge_baseline_failure_is_terminal_without_a_quality_best(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_rollout_calls(monkeypatch, {1})

    result = _run("run", "start", "--size", "2", "--max-iterations", "1")

    assert result.exit_code == 0, result.output
    payload = _run_payload(result.output)
    assert payload["status"] == "done"
    assert payload["best_candidate_id"] is None
    comparison = payload["last_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["outcome"] == "infrastructure_failure"
    assert comparison["retryable"] is False
    assert comparison["recommendation"] == "stop_budget_exhausted"
    report = Path(str(payload["final_report_path"])).read_text(encoding="utf-8")
    assert "accepted_best_candidate_id" not in report


def test_budget_edge_candidate_failure_finishes_without_promoting_candidate(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_rollout_calls(monkeypatch, {2})
    start = _run("run", "start", "--size", "2", "--max-iterations", "2")
    start_payload = _run_payload(start.output)
    incumbent = start_payload["best_candidate_id"]

    result = _run("run", "continue", "--run-id", str(start_payload["run_id"]))

    assert result.exit_code == 0, result.output
    payload = _run_payload(result.output)
    assert payload["status"] == "done"
    assert payload["best_candidate_id"] == incumbent
    comparison = payload["last_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["outcome"] == "infrastructure_failure"
    assert comparison["retryable"] is False


def test_mixed_baseline_repetitions_discard_partial_samples(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_rollout_calls(monkeypatch, {2})

    result = _run(
        "run",
        "start",
        "--size",
        "2",
        "--max-iterations",
        "10",
        "--acceptance-repetitions",
        "3",
        "--acceptance-max-repetitions",
        "3",
    )

    assert result.exit_code == 0, result.output
    payload = _run_payload(result.output)
    assert payload["status"] == "paused_after_infrastructure_error"
    assert payload["iterations"] == 2
    assert payload["best_candidate_id"] is None
    assert payload["reflection_baseline_samples"] == []
    comparison = payload["last_comparison"]
    assert isinstance(comparison, dict)
    assert len(comparison["valid_samples_before_failure"]) == 1
    assert comparison["verdict"] is None


def test_healthy_baseline_before_later_failure_remains_the_incumbent(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic_ai_gepa.cli import run as run_module

    original = run_module.run_eval_once
    call_count = 0

    def wrapped(**kwargs):
        nonlocal call_count
        call_count += 1
        outcome = original(**kwargs)
        if call_count == 1:
            outcome.summary["n_failures"] = 0
        elif call_count == 2:
            record = outcome.records[0]
            outcome.records[0] = EvaluationRecord(
                case_id=record.case_id,
                score=record.score,
                feedback=record.feedback,
                payload={
                    **record.payload,
                    "output": RolloutOutput.from_error(
                        RuntimeError("provider unavailable"), kind="system"
                    ),
                },
            )
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", wrapped)

    result = _run("run", "start", "--size", "2", "--max-iterations", "4")

    assert result.exit_code == 0, result.output
    payload = _run_payload(result.output)
    assert payload["status"] == "paused_after_infrastructure_error"
    assert payload["best_candidate_id"] is not None
    assert payload["best_mean_score"] == pytest.approx(0.5)


def test_candidate_rollout_failure_preserves_incumbent_and_can_retry(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_rollout_calls(monkeypatch, {2})
    start = _run("run", "start", "--size", "2", "--max-iterations", "4")
    start_payload = _run_payload(start.output)
    incumbent = start_payload["best_candidate_id"]

    failed = _run("run", "continue", "--run-id", str(start_payload["run_id"]))

    assert failed.exit_code == 0, failed.output
    failed_payload = _run_payload(failed.output)
    assert failed_payload["status"] == "paused_after_infrastructure_error"
    assert failed_payload["best_candidate_id"] == incumbent
    comparison = failed_payload["last_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["outcome"] == "infrastructure_failure"
    assert comparison["phase"] == "candidate"
    assert comparison["verdict"] is None

    retried = _run("run", "continue", "--run-id", str(start_payload["run_id"]))
    assert retried.exit_code == 0, retried.output
    retry_comparison = _run_payload(retried.output)["last_comparison"]
    assert isinstance(retry_comparison, dict)
    assert retry_comparison["outcome"] == "valid"
    assert retry_comparison["verdict"] == "equivalent"


def test_mixed_candidate_repetitions_never_compare_partial_samples(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_rollout_calls(monkeypatch, {5})
    start = _run(
        "run",
        "start",
        "--size",
        "2",
        "--max-iterations",
        "10",
        "--acceptance-repetitions",
        "3",
        "--acceptance-max-repetitions",
        "3",
    )
    start_payload = _run_payload(start.output)
    baseline_samples = start_payload["reflection_baseline_samples"]
    assert isinstance(baseline_samples, list)
    assert len(baseline_samples) == 3

    failed = _run("run", "continue", "--run-id", str(start_payload["run_id"]))

    assert failed.exit_code == 0, failed.output
    payload = _run_payload(failed.output)
    assert payload["status"] == "paused_after_infrastructure_error"
    comparison = payload["last_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["outcome"] == "infrastructure_failure"
    assert comparison["valid_samples_before_failure"] == [pytest.approx(0.5)]
    assert comparison["verdict"] is None


def test_continue_after_revert_discards_candidate_and_advances(repo: Path) -> None:
    start = _run("run", "start", "--size", "2", "--max-iterations", "3")
    run_id = str(_run_payload(start.output)["run_id"])

    first_continue = _run("run", "continue", "--run-id", run_id)
    assert first_continue.exit_code == 0, first_continue.output
    assert (
        _run_payload(first_continue.output)["status"] == "paused_after_candidate_eval"
    )

    second_continue = _run("run", "continue", "--run-id", run_id)

    assert second_continue.exit_code == 0, second_continue.output
    assert "discarding the losing candidate and advancing" in second_continue.output
    payload = _run_payload(second_continue.output)
    assert payload["status"] == "done"
    assert payload["iterations"] == 3


def test_current_baseline_candidate_id_includes_configured_skills(repo: Path) -> None:
    skills_dir = repo / "skills" / "month-grid"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        "---\n"
        "name: month-grid\n"
        "description: Use for visual calendar grids.\n"
        "---\n"
        "# Month Grid\n",
        encoding="utf-8",
    )
    config = repo / ".gepa" / "gepa.toml"
    config.write_text(
        config.read_text(encoding="utf-8") + 'skills = "skills"\n',
        encoding="utf-8",
    )

    from pydantic_ai_gepa.cli.layout import (
        GepaConfig,
        config_path,
        resolve_agent,
        resolve_skills,
    )
    from pydantic_ai_gepa.cli.run import _current_baseline_candidate_id
    from pydantic_ai_gepa.cli.store import ComponentStore

    cfg = GepaConfig.load(config_path())
    agent = resolve_agent(cfg)
    store = ComponentStore()
    without_skills = candidate_id_from_components(store.effective_candidate(agent))
    expected = candidate_id_from_components(
        store.effective_candidate(agent, skills_fs=resolve_skills(cfg))
    )

    assert expected != without_skills
    assert _current_baseline_candidate_id() == expected


def test_managed_run_prints_final_report_at_max_iterations(repo: Path) -> None:
    start = _run("run", "start", "--size", "2", "--max-iterations", "2")
    run_id = str(_run_payload(start.output)["run_id"])

    done = _run("run", "continue", "--run-id", run_id)

    assert done.exit_code == 0, done.output
    payload = _run_payload(done.output)
    assert payload["status"] == "done"
    assert payload["final_report_path"] == str(final_report_path(run_id, repo))
    assert Path(str(payload["final_report_path"])).exists()
    assert "GEPA Run Final Report" in done.output

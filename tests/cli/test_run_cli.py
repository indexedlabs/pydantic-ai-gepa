"""End-to-end tests for the managed `gepa run` controller."""

from __future__ import annotations

from dataclasses import replace
import json
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest
from tests.cli.harness_helpers import scored_continue
from click.testing import Result
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.candidates import candidate_id_from_components
from pydantic_ai_gepa.cli.layout import final_report_path, run_state_path
from pydantic_ai_gepa.cli.run import (
    _consume_candidate_verdict,
    _load_state,
    _public_state,
)
from pydantic_ai_gepa.cli.runs import ParetoLog
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
    result = _run("run", "start", "--size", "2", "--max-iterations", "8")

    assert result.exit_code == 0, result.output
    payload = _run_payload(result.output)
    assert payload["status"] == "paused_for_reflection"
    assert payload["iterations"] == 4
    assert payload["next_command"] == f"gepa run continue --run-id {payload['run_id']}"
    assert Path(str(payload["reflection_baseline_report_path"])).exists()
    assert Path(str(payload["reflection_baseline_trace_path"])).exists()
    assert run_state_path(str(payload["run_id"]), repo).exists()


def test_run_start_defaults_match_minibatch_evaluation(repo: Path) -> None:
    result = _run("run", "start", "--size", "2", "--max-iterations", "8")

    assert result.exit_code == 0, result.output
    payload = _run_payload(result.output)
    assert payload["concurrency"] == 2
    assert payload["acceptance_repetitions"] == 3
    assert payload["acceptance_max_repetitions"] == 3
    baseline_samples = payload["reflection_baseline_samples"]
    assert isinstance(baseline_samples, list)
    assert len(baseline_samples) == 3


def test_continue_rejects_external_validation_tampered_after_start(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_path = repo.parent / "validation.jsonl"
    validation_path.write_text(
        json.dumps(
            {"name": "secret-holdout", "inputs": "x", "expected_output": "Paris"}
        )
        + "\n"
    )
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(validation_path))
    started = _run(
        "run",
        "start",
        "--lanes",
        "0",
        "--size",
        "2",
        "--max-iterations",
        "20",
        "--acceptance-repetitions",
        "1",
    )
    assert started.exit_code == 0, started.output
    payload = _run_payload(started.output)
    assert payload["status"] == "paused_for_reflection"
    validation_path.write_text(
        json.dumps({"name": "tampered", "inputs": "x", "expected_output": "Berlin"})
        + "\n"
    )
    result = scored_continue("run", "continue", "--run-id", str(payload["run_id"]))
    assert result.exit_code == 2, result.output
    assert "changed after run start" in result.output


def test_held_out_validation_selects_without_exposing_reflection_evidence(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic_ai_gepa.cli import run as run_module

    validation_cases = [
        {"name": "secret-validation-alpha", "inputs": "?", "expected_output": "Paris"},
        {"name": "secret-validation-beta", "inputs": "?", "expected_output": "Berlin"},
    ]
    validation_path = repo.parent / "validation.jsonl"
    validation_path.write_text(
        "\n".join(json.dumps(row) for row in validation_cases) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(validation_path))

    started = _run(
        "run",
        "start",
        "--size",
        "2",
        "--max-iterations",
        "16",
        "--acceptance-repetitions",
        "1",
    )
    assert started.exit_code == 0, started.output
    start_payload = _run_payload(started.output)
    assert start_payload["status"] == "paused_for_reflection"
    assert start_payload["validation_seeded"] is True
    assert start_payload["validation_evaluations"] == 3

    original = run_module.run_eval_once
    calls: list[dict[str, object]] = []

    def training_wins_validation_loses(**kwargs):
        calls.append(kwargs)
        outcome = original(**kwargs)
        if kwargs.get("dataset_role", "training") == "training":
            outcome.summary["mean_score"] = 1.0
        elif kwargs.get("dataset_role") == "validation":
            outcome.summary["mean_score"] = 0.0
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", training_wins_validation_loses)
    result = scored_continue(
        "run", "continue", "--run-id", str(start_payload["run_id"])
    )

    assert result.exit_code == 0, result.output
    payload = _run_payload(result.output)
    comparison = payload["last_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["training_verdict"] == "accepted"
    assert comparison["validation_improved"] is False
    assert comparison["rejection_reason"] == "validation"
    assert payload["status"] == "paused_after_candidate_eval"
    assert payload["validation_evaluations"] == 6
    assert [call.get("dataset_role", "training") for call in calls] == [
        "training",
        "training",
        "training",
        "validation",
        "validation",
        "validation",
    ]
    assert calls[-1]["capture_traces"] is False
    assert calls[-1]["persist_report"] is False
    assert calls[-1]["redact_selection_evidence"] is True
    status = _run("run", "status", "--run-id", str(start_payload["run_id"]))
    assert status.exit_code == 0, status.output
    for secret in ("secret-validation-alpha", "secret-validation-beta"):
        assert secret not in started.output + result.output + status.output

    run_root = repo / ".gepa" / "runs" / str(start_payload["run_id"])
    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in run_root.rglob("*")
        if path.is_file()
    )
    assert "secret-validation-alpha" not in persisted
    assert "secret-validation-beta" not in persisted
    validation_rows = ParetoLog(str(start_payload["run_id"])).validation_rows()
    assert len(validation_rows) == 6
    assert all(not row.per_case_scores for row in validation_rows)
    assert ParetoLog(str(start_payload["run_id"])).count_budget_rows() == 13


def test_held_out_final_report_is_aggregate_only(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_path = repo.parent / "validation.jsonl"
    validation_path.write_text(
        json.dumps(
            {
                "name": "WITHHELD_CASE",
                "inputs": "WITHHELD_INPUT",
                "expected_output": "Paris",
            }
        )
        + "\n"
    )
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(validation_path))
    started = _run("run", "start", "--size", "2", "--max-iterations", "3")
    assert started.exit_code == 0, started.output
    payload = _run_payload(started.output)
    run_id = str(payload["run_id"])
    if payload["status"] != "done":
        done = scored_continue("run", "continue", "--run-id", run_id)
        assert done.exit_code == 0, done.output
        payload = _run_payload(done.output)
    assert payload["status"] == "done"
    report = Path(str(payload["final_report_path"])).read_text()
    assert "validation_evaluations: 3" in report
    assert "WITHHELD" not in report


def test_stall_block_appears_after_five_non_promoting_verdicts_and_clears(
    repo: Path,
) -> None:
    start = _run(
        "run",
        "start",
        "--size",
        "2",
        "--max-iterations",
        "8",
        "--acceptance-repetitions",
        "1",
    )
    run_id = str(_run_payload(start.output)["run_id"])
    state = _load_state(run_id)

    for _ in range(5):
        state = _consume_candidate_verdict(state, accepted=False)

    stalled = _public_state(state, outcomes=[])
    assert stalled["stall"] == {
        "stalled": True,
        "iterations_since_acceptance": 5,
    }

    accepted = _public_state(
        _consume_candidate_verdict(state, accepted=True), outcomes=[]
    )
    assert "stall" not in accepted


def test_gate_rejection_skips_full_minibatch_and_pareto(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic_ai_gepa.cli import run as run_module

    start = _run(
        "run",
        "start",
        "--size",
        "2",
        "--max-iterations",
        "8",
        "--acceptance-repetitions",
        "1",
    )
    run_id = str(_run_payload(start.output)["run_id"])
    before = ParetoLog(run_id).count_rows()
    original = run_module.run_eval_once
    calls: list[dict[str, object]] = []

    def gate_loses(**kwargs):
        calls.append(kwargs)
        outcome = original(**kwargs)
        if kwargs.get("selected_case_ids"):
            outcome.summary["mean_score"] = 0.0
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", gate_loses)
    result = _run("run", "continue", "--run-id", run_id, "--gate-case", "case-paris")

    assert result.exit_code == 0, result.output
    assert len(calls) == 3
    assert calls[0]["selected_case_ids"] == ("case-paris",)
    payload = _run_payload(result.output)
    comparison = payload["last_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["verdict"] == "rejected"
    assert comparison["rejection_reason"] == "gate"
    assert ParetoLog(run_id).count_rows() == before


def test_gate_rejection_consumes_managed_run_budget(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic_ai_gepa.cli import run as run_module

    start = _run(
        "run",
        "start",
        "--size",
        "2",
        "--max-iterations",
        "7",
        "--acceptance-repetitions",
        "1",
    )
    run_id = str(_run_payload(start.output)["run_id"])
    original = run_module.run_eval_once

    def gate_loses(**kwargs):
        outcome = original(**kwargs)
        if kwargs.get("selected_case_ids"):
            outcome.summary["mean_score"] = 0.0
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", gate_loses)
    result = _run("run", "continue", "--run-id", run_id, "--gate-case", "case-paris")

    assert result.exit_code == 0, result.output
    payload = _run_payload(result.output)
    assert payload["status"] == "paused_after_candidate_eval"
    assert payload["iterations"] == 5
    assert _load_state(run_id).gate_consumed_iterations == 1


def test_gate_passes_then_full_minibatch_verdict_governs(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic_ai_gepa.cli import run as run_module

    start = _run(
        "run",
        "start",
        "--size",
        "2",
        "--max-iterations",
        "8",
        "--acceptance-repetitions",
        "1",
    )
    run_id = str(_run_payload(start.output)["run_id"])
    original = run_module.run_eval_once
    calls: list[dict[str, object]] = []

    def gate_passes(**kwargs):
        calls.append(kwargs)
        outcome = original(**kwargs)
        if kwargs.get("selected_case_ids") == ("case-paris",):
            outcome.summary["mean_score"] = 2.0
        return outcome

    monkeypatch.setattr(run_module, "run_eval_once", gate_passes)
    result = _run("run", "continue", "--run-id", run_id, "--gate-case", "case-paris")

    assert result.exit_code == 0, result.output
    assert [call.get("selected_case_ids") for call in calls] == [
        ("case-paris",),
        ("case-paris",),
        ("case-paris",),
        ["case-berlin"],
        None,
        None,
    ]
    comparison = _run_payload(result.output)["last_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["verdict"] == "equivalent"
    assert "rejection_reason" not in comparison


def test_unknown_gate_case_errors_before_evaluation(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic_ai_gepa.cli import run as run_module

    start = _run(
        "run",
        "start",
        "--size",
        "2",
        "--max-iterations",
        "8",
        "--acceptance-repetitions",
        "1",
    )
    run_id = str(_run_payload(start.output)["run_id"])
    monkeypatch.setattr(
        run_module,
        "run_eval_once",
        lambda **_: pytest.fail("gate validation must happen before evaluation"),
    )

    result = _run("run", "continue", "--run-id", run_id, "--gate-case", "missing")

    assert result.exit_code == 2
    assert "Available:" in result.output
    assert "case-paris" in result.output
    assert "case-berlin" in result.output


def test_no_gate_repeats_the_full_minibatch(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic_ai_gepa.cli import run as run_module

    start = _run(
        "run",
        "start",
        "--size",
        "2",
        "--max-iterations",
        "8",
        "--acceptance-repetitions",
        "1",
    )
    run_id = str(_run_payload(start.output)["run_id"])
    before = ParetoLog(run_id).count_rows()
    original = run_module.run_eval_once
    calls: list[dict[str, object]] = []

    def count_calls(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(run_module, "run_eval_once", count_calls)
    result = _run("run", "continue", "--run-id", run_id)

    assert result.exit_code == 0, result.output
    assert len(calls) == 3
    assert calls[0].get("selected_case_ids") is None
    assert ParetoLog(run_id).count_rows() == before + 3


def test_continue_reports_equivalent_when_candidate_does_not_change(
    repo: Path,
) -> None:
    start = _run("run", "start", "--size", "2", "--max-iterations", "8")
    run_id = str(_run_payload(start.output)["run_id"])

    result = _run("run", "continue", "--run-id", run_id)

    assert result.exit_code == 0, result.output
    assert "equivalent" in result.output
    payload = _run_payload(result.output)
    assert payload["status"] == "paused_after_candidate_eval"
    assert payload["iterations"] == 7
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
        "11",
        "--acceptance-repetitions",
        "3",
        "--acceptance-max-repetitions",
        "5",
    )

    assert start.exit_code == 0, start.output
    start_payload = _run_payload(start.output)
    assert start_payload["status"] == "paused_for_reflection"
    assert start_payload["iterations"] == 6
    baseline_samples = start_payload["reflection_baseline_samples"]
    baseline_report_paths = start_payload["reflection_baseline_report_paths"]
    assert isinstance(baseline_samples, list)
    assert isinstance(baseline_report_paths, list)
    assert len(baseline_samples) == 5
    assert len(set(baseline_report_paths)) == 6

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
    assert payload["iterations"] == 9


def test_baseline_rollout_failure_pauses_without_installing_baseline(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fail_rollout_calls(monkeypatch, {1})

    result = _run("run", "start", "--size", "2", "--max-iterations", "8")

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
    assert len(retry_samples) == 3
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
    _fail_rollout_calls(monkeypatch, {7})
    start = _run("run", "start", "--size", "2", "--max-iterations", "7")
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
    _fail_rollout_calls(monkeypatch, {3})

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
    assert payload["iterations"] == 3
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
    _fail_rollout_calls(monkeypatch, {5})
    start = _run("run", "start", "--size", "2", "--max-iterations", "10")
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
    _fail_rollout_calls(monkeypatch, {6})
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
    start = _run("run", "start", "--size", "2", "--max-iterations", "8")
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
    assert payload["iterations"] == 8


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


@pytest.mark.parametrize(
    "mismatch", [None, "training", "validation", "missing_training"]
)
def test_paired_config_drives_single_repetition_promotion(
    repo: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str | None
) -> None:
    from pydantic_ai_gepa.cli import eval as eval_module
    from pydantic_ai_gepa.cli.store import ComponentStore

    validation = [
        {"name": f"held-out-{i}", "inputs": "?", "expected_output": "a"}
        for i in range(10)
    ]
    (repo / ".gepa/dataset.jsonl").write_text(
        "".join(
            json.dumps({"name": f"training-{i}", "inputs": "?", "expected_output": "a"})
            + "\n"
            for i in range(10)
        )
    )
    validation_path = repo.parent / "validation.jsonl"
    validation_path.write_text("\n".join(json.dumps(row) for row in validation) + "\n")
    config = repo / ".gepa" / "gepa.toml"
    config.write_text(config.read_text() + "[acceptance]\npaired_min_cases = 2\n")

    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(validation_path))

    async def evaluate(**kwargs):
        improved = kwargs["candidate"]["instructions"].text == "Improved prompt"
        return [
            EvaluationRecord(
                case_id=("different-" + case.name)
                if improved
                and mismatch
                == ("validation" if case.name.startswith("held-out") else "training")
                else case.name,
                score=0.6 if improved else 0.4,
                feedback="Synthetic feedback",
                payload={},
            )
            for case in kwargs["dataset"]
        ]

    def assert_validation_ids_withheld() -> None:
        for artifact in (repo / ".gepa").rglob("*"):
            if artifact.is_file():
                contents = artifact.read_bytes()
                for case in validation:
                    assert case["name"].encode() not in contents, artifact

    monkeypatch.setattr(eval_module, "evaluate_candidate_dataset", evaluate)
    started = _run("run", "start", "--size", "10", "--max-iterations", "6")
    assert started.exit_code == 0, (started.output, started.exception)
    start_payload = _run_payload(started.output)
    assert start_payload["acceptance_paired_min_cases"] == 2
    assert start_payload["reflection_baseline_samples"] == pytest.approx([0.4])
    assert start_payload["best_validation_samples"] == pytest.approx([0.4])
    assert "best_validation_per_case_scores" not in start_payload
    for case in validation:
        assert case["name"] not in started.output
    run_id = str(start_payload["run_id"])
    seed_state = _load_state(run_id).restore_validation_evidence()
    assert seed_state.best_validation_per_case_scores == {
        case["name"]: 0.4 for case in validation
    }
    assert_validation_ids_withheld()
    ComponentStore().write("instructions", "Improved prompt")
    if mismatch == "missing_training":
        from pydantic_ai_gepa.cli import run as run_module

        monkeypatch.setattr(run_module, "_reflection_case_scores", lambda *a, **k: {})
    result = scored_continue(
        "run", "continue", "--run-id", str(start_payload["run_id"])
    )
    assert result.exit_code == 0, (result.output, result.exception)
    state = _load_state(str(start_payload["run_id"])).restore_validation_evidence()
    if mismatch:
        assert state.best_candidate_id == seed_state.best_candidate_id
        assert state.last_comparison["verdict"] == "inconclusive"
        assert state.last_comparison["reason_code"] == (
            "paired_evidence_missing"
            if mismatch == "missing_training"
            else "paired_cases_mismatched"
        )
        assert state.last_comparison["selectable"] is False
        assert state.last_comparison["improved"] is False
        status = _run("run", "status", "--run-id", state.run_id)
        assert status.exit_code == 0, status.output
        assert _run_payload(status.output)["last_comparison"] == state.last_comparison
        assert_validation_ids_withheld()
        return
    assert state.best_validation_samples == pytest.approx((0.6,))
    assert state.best_validation_per_case_scores == {
        case["name"]: 0.6 for case in validation
    }
    assert state.validation_evaluations == 2
    for case in validation:
        assert case["name"] not in result.output
    assert_validation_ids_withheld()


@pytest.mark.parametrize("remaining", [0, 2])
def test_reflected_candidate_budget_refusal_records_comparison_and_exits_70(
    repo: Path, monkeypatch: pytest.MonkeyPatch, remaining: int
) -> None:
    from pydantic_ai_gepa.cli import run as run_module

    started = _run("run", "start", "--size", "2", "--max-iterations", "7")
    assert started.exit_code == 0, (started.output, started.exception)
    run_id = str(_run_payload(started.output)["run_id"])
    before = _load_state(run_id)
    replace(before, max_iterations=before.iterations + remaining).save()
    rows_before = ParetoLog(run_id).count_rows()

    def no_evaluation(**kwargs):
        pytest.fail("Insufficient candidate budget must not start an evaluation")

    monkeypatch.setattr(run_module, "run_eval_once", no_evaluation)
    result = _run("run", "continue", "--run-id", run_id)
    assert result.exit_code == 70, (result.output, result.exception)
    after = _load_state(run_id)
    assert after.last_comparison["verdict"] == "inconclusive"
    assert after.last_comparison["reason_code"] == "candidate_budget_exhausted"
    assert after.last_comparison["improved"] is False
    assert after.best_candidate_id == before.best_candidate_id
    assert after.iterations == before.iterations
    assert ParetoLog(run_id).count_rows() == rows_before
    assert _run_payload(result.output)["last_comparison"] == after.last_comparison


@pytest.mark.parametrize(
    "damage",
    [
        "moved",
        "changed",
        "unreadable_dataset",
        "unreadable_evidence",
        "corrupt",
        "invalid_utf8",
    ],
)
@pytest.mark.parametrize("finished", [False, True])
def test_paired_status_survives_unavailable_private_evidence(
    repo, monkeypatch, damage, finished
):
    import os
    from pydantic_ai_gepa.cli import run as run_module
    from pydantic_ai_gepa.cli.validation import validation_evidence_path
    from pydantic_ai_gepa.cli.store import ComponentStore

    private = repo.parent / f"{repo.name}-private"
    private.mkdir()
    dataset = private / "validation.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({**row, "name": "private-" + row["name"]}) + "\n"
            for row in DATASET
        )
    )
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(dataset))
    started = _run(
        "run",
        "start",
        "--size",
        "2",
        "--max-iterations",
        "8",
        "--acceptance-paired-min-cases",
        "2",
    )
    assert started.exit_code == 0, (started.output, started.exception)
    run_id = str(_run_payload(started.output)["run_id"])
    before = _load_state(run_id).restore_validation_evidence()
    assert before.best_validation_per_case_scores
    if finished:
        replace(before, status="done").save()
    evidence = validation_evidence_path(str(dataset), project_root=repo, run_id=run_id)
    assert evidence.parent.stat().st_mode & 0o777 == 0o700
    if damage == "moved":
        dataset.rename(private / "moved.jsonl")
    elif damage == "changed":
        dataset.write_text(dataset.read_text() + "\n")
    elif damage in {"unreadable_dataset", "unreadable_evidence"}:
        if os.geteuid() == 0:
            pytest.skip("root bypasses file permissions")
        target = dataset if damage == "unreadable_dataset" else evidence
        target.chmod(0)
    elif damage == "invalid_utf8":
        evidence.write_bytes(b"\xff")
    else:
        evidence.write_text('{"scores": "PRIVATE_CORRUPT_DATA"')
    try:
        status = _run("run", "status", "--run-id", run_id)
        assert status.exit_code == 0, status.output
        restored = _load_state(run_id).restore_validation_evidence()
        assert restored.best_validation_per_case_scores == {}
        assert str(dataset) not in status.output
        assert "PRIVATE_CORRUPT_DATA" not in status.output
        assert "best_validation_per_case_scores" not in _run_payload(status.output)
        # The same restore operation clears even evidence already held in memory.
        assert (
            before.restore_validation_evidence().best_validation_per_case_scores == {}
        )
        if damage in {"moved", "changed", "unreadable_dataset"}:
            import typer
            from pydantic_ai_gepa.cli.spend import _rows

            if damage != "changed":
                assert _run_payload(status.output)["spend"][
                    "validation_checkpoint_missing"
                ]
                with pytest.raises(typer.BadParameter):
                    _rows(run_id, repo)  # Admission must still fail closed.
            for evaluate in (
                run_module._confirm_validation_candidate,
                run_module._evaluate_validation_candidate,
                run_module._ensure_validation_seed,
            ):
                with pytest.raises(typer.BadParameter):
                    evaluate(restored)
        else:
            updated, outcomes, comparison = run_module._confirm_validation_candidate(
                restored
            )
            assert not outcomes
            assert comparison["reason_code"] == "incumbent_evidence_missing"
            assert comparison["improved"] is False
            assert updated.best_candidate_id == before.best_candidate_id
        # A public continuation also cannot promote with a missing/changed pin.
        if not finished and damage in {"moved", "changed", "unreadable_dataset"}:
            ComponentStore().write("instructions", "Improved prompt")
            result = scored_continue("run", "continue", "--run-id", run_id)
            assert result.exit_code != 0, result.output
            assert str(dataset) not in result.output
            assert _load_state(run_id).best_candidate_id == before.best_candidate_id
    finally:
        if damage in {"unreadable_dataset", "unreadable_evidence"}:
            target.chmod(0o600)


@pytest.mark.parametrize("blocked", ["parent", "directory", "file"])
def test_paired_start_refuses_unwritable_evidence_before_run_creation(
    repo, monkeypatch, blocked
):
    import os
    from pydantic_ai_gepa.cli.layout import runs_dir

    if os.geteuid() == 0:
        pytest.skip("root bypasses directory permissions")
    private = repo.parent / f"{repo.name}-private"
    private.mkdir()
    dataset = private / "validation.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({**row, "name": "private-" + row["name"]}) + "\n"
            for row in DATASET
        )
    )
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(dataset))
    directory = private / ".gepa-validation-evidence"
    if blocked == "directory":
        directory.mkdir(mode=0o500)
    elif blocked == "file":
        directory.write_text("not a directory")
    else:
        private.chmod(0o500)
    before = set(private.iterdir())
    store_before = list(runs_dir(repo).glob("*"))
    try:
        result = _run("run", "start", "--acceptance-paired-min-cases", "2")
        assert result.exit_code == 2, (result.output, result.exception)
        assert "Cannot write private held-out validation evidence" in result.output
        assert str(private) not in result.output
        assert "private path must not escape" not in result.output
        assert set(private.iterdir()) == before
        assert list(runs_dir(repo).glob("*")) == store_before
    finally:
        if blocked == "directory":
            directory.chmod(0o700)
        private.chmod(0o700)

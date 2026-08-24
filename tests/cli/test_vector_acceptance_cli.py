"""Artifact-level regressions for the assertion-vector lane and probe flow."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import pytest
import typer
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.candidates import git_candidate_state
from pydantic_ai_gepa.cli.eval import EvalOutcome, _format_failures, _write_trace_file
from pydantic_ai_gepa.cli.lanes import (
    _candidate_gate,
    _run_lane_eval_loop,
    LaneState,
    load_lane_state,
    write_packet,
)
from pydantic_ai_gepa.cli.layout import (
    journal_path,
    probe_receipts_dir,
    vector_records_path,
)
from pydantic_ai_gepa.cli.probe import component_hash
from pydantic_ai_gepa.cli.run import RunState
from pydantic_ai_gepa.cli.runs import ParetoLog
from pydantic_ai_gepa.cli.select import _numeric_ranking_key
from pydantic_ai_gepa.evaluation import EvaluationRecord
from pydantic_ai_gepa.types import RolloutOutput
from pydantic_ai_gepa.vector_acceptance import (
    VectorComparison,
    VectorComparisonRequest,
    VectorRecord,
    VectorRecordStore,
)


def _run(*argv: str):
    return CliRunner().invoke(gepa_app, list(argv), catch_exceptions=False)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _run_payload(output: str) -> dict[str, object]:
    line = next(line for line in reversed(output.splitlines()) if '"run"' in line)
    return json.loads(line)["run"]


def _config(
    *,
    components: tuple[str, ...] = ("score.txt",),
    receipt: bool = False,
    rebaseline_interval: int | None = None,
    pinned_scorer: bool = False,
    reviewer: bool = False,
    probe_allowance_per_lease: int | None = None,
) -> str:
    files = ", ".join(json.dumps(item) for item in components)
    rebaseline = (
        ""
        if rebaseline_interval is None
        else f"rebaseline_interval = {rebaseline_interval}\n"
    )
    pinned = "pinned_scorer = true\n" if pinned_scorer else ""
    reviewer_line = (
        'reviewer = "vector_pkg.reviewer:make_reviewer"\n' if reviewer else ""
    )
    probe_allowance = (
        ""
        if probe_allowance_per_lease is None
        else f"probe_allowance_per_lease = {probe_allowance_per_lease}\n"
    )
    return textwrap.dedent(f"""
        evaluate = "vector_pkg.evaluation:evaluate"
        candidate_source = "git"
        dataset = ".gepa/dataset.jsonl"
        metric = "vector_pkg.metric:metric"

        [acceptance]
        mode = "vector"
        comparator = "vector_pkg.comparator:make_comparator"
        {pinned}
        component_files = [{files}]
        meta_files = ["prediction.json"]
        require_probe_receipt = {str(receipt).lower()}
        {rebaseline}
        {reviewer_line}
        {probe_allowance}
    """).lstrip()


@pytest.fixture
def vector_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from pydantic_ai_gepa.cli import layout

    saved_dirname = layout._explicit_gepa_dirname
    monkeypatch.delenv("GEPA_DIR", raising=False)
    package = tmp_path / "vector_pkg"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "evaluation.py").write_text(
        "from pathlib import Path\n"
        "async def evaluate(case):\n"
        "    return Path('score.txt').read_text(encoding='utf-8').strip()\n",
        encoding="utf-8",
    )
    (package / "metric.py").write_text(
        "from pydantic_ai_gepa import MetricResult\n"
        "def metric(case, output):\n"
        "    status = 'preferred' if output == 'preferred' else ('pass' if output == 'good' else 'fail')\n"
        "    return MetricResult(float(status == 'pass'), side_info={\n"
        "        'assertions': {'quality': {'status': status}},\n"
        "        'latency': {'engine': 1.0},\n"
        "    })\n",
        encoding="utf-8",
    )
    (package / "comparator.py").write_text(
        "from pydantic_ai_gepa.vector_acceptance import VectorComparison\n"
        "class Comparator:\n"
        "    def compare(self, request):\n"
        "        context = dict(request.journal_context)\n"
        "        kind = context.get('comparison_kind')\n"
        "        statuses = [case['quality']['status'] for record in request.candidate for case in record.assertions.values()]\n"
        "        case_ids = [case_id for record in request.candidate for case_id in record.assertions]\n"
        "        held_out_vector = case_ids == ['secret-vector-validation']\n"
        "        verdict = 'rejected' if kind == 'run_start_rebaseline' or (kind == 'validation_selection' and (context.get('lane') == 'lane-3' or not held_out_vector)) else 'accepted'\n"
        "        ranking = ({'lane-1': 1.0, 'lane-2': 2.0, 'lane-3': 3.0}.get(context.get('lane'), 1.0),)\n"
        "        return VectorComparison(verdict, ranking_key=ranking, display_score=1.0, detail={\n"
        "            'context': context,\n"
        "            **({'held_out_private_detail': {'statuses': statuses, 'case_ids': case_ids}} if kind == 'validation_selection' else {}),\n"
        "            'incumbent_records': len(request.incumbent),\n"
        "            'candidate_records': len(request.candidate),\n"
        "        })\n"
        "def make_comparator():\n"
        "    return Comparator()\n",
        encoding="utf-8",
    )
    (package / "reviewer.py").write_text(
        "from pydantic_ai_gepa.candidate_review import CandidateReviewVerdict, ReviewFinding\n"
        "class Reviewer:\n"
        "    def review(self, request):\n"
        "        if any('blocked' in value for value in request.components.values()):\n"
        "            return CandidateReviewVerdict('fail', (ReviewFinding(None, None, 'blocked candidate', 'error'),))\n"
        "        return CandidateReviewVerdict('pass')\n"
        "def make_reviewer():\n"
        "    return Reviewer()\n",
        encoding="utf-8",
    )
    (tmp_path / "score.txt").write_text("bad\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(
        "__pycache__/\n.gepa/runs/\n", encoding="utf-8"
    )
    (tmp_path / ".gepa").mkdir()
    (tmp_path / ".gepa" / "runs").mkdir()
    (tmp_path / ".gepa" / "gepa.toml").write_text(_config(), encoding="utf-8")
    cases = [
        {"name": "case-1", "inputs": "one", "expected_output": "good"},
        {"name": "case-2", "inputs": "two", "expected_output": "good"},
    ]
    (tmp_path / ".gepa" / "dataset.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in cases), encoding="utf-8"
    )
    (tmp_path / ".gepa" / "journal.jsonl").touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "tests@example.com")
    _git(tmp_path, "config", "user.name", "GEPA Tests")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "Seed vector candidate")
    yield tmp_path
    layout._explicit_gepa_dirname = saved_dirname
    for name in list(sys.modules):
        if name.startswith("vector_pkg"):
            sys.modules.pop(name, None)


def _start(
    repo: Path, *, repetitions: int = 1, lanes: int = 1
) -> tuple[dict[str, object], RunState]:
    result = _run(
        "--gepa-dir",
        str(repo / ".gepa"),
        "run",
        "start",
        "--lanes",
        str(lanes),
        "--size",
        "2",
        "--acceptance-repetitions",
        str(repetitions),
        "--acceptance-max-repetitions",
        str(repetitions + 1),
    )
    assert result.exit_code == 0, result.output
    payload = _run_payload(result.output)
    path = repo / ".gepa" / "runs" / str(payload["run_id"]) / "state.json"
    return payload, RunState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _configure_validation(repo: Path, *, pinned_scorer: bool) -> None:
    config = repo / ".gepa" / "gepa.toml"
    config.write_text(
        _config(pinned_scorer=pinned_scorer).replace(
            'dataset = ".gepa/dataset.jsonl"\n',
            'dataset = ".gepa/dataset.jsonl"\n'
            'validation_dataset = ".gepa/validation.jsonl"\n',
        ),
        encoding="utf-8",
    )
    (repo / ".gepa" / "validation.jsonl").write_text(
        json.dumps(
            {
                "name": "secret-vector-validation",
                "inputs": "private",
                "expected_output": "good",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".gepa/gepa.toml", ".gepa/validation.jsonl")
    _git(repo, "commit", "-m", "Configure vector validation")


def _continue_vector_lane(repo: Path, run_id: str, lane: str, value: str) -> LaneState:
    state = load_lane_state(repo, run_id, lane)
    worktree = Path(str(state.worktree_path))
    (worktree / "score.txt").write_text(value, encoding="utf-8")
    previous_cwd = Path.cwd()
    try:
        os.chdir(worktree)
        continued = _run(
            "--gepa-dir",
            str(repo / ".gepa"),
            "lane",
            "continue",
            lane,
            "--run-id",
            run_id,
            "--foreground",
        )
    finally:
        os.chdir(previous_cwd)
    assert continued.exit_code == 0, continued.output
    return load_lane_state(repo, run_id, lane)


def test_vector_mode_never_persists_validation_assertions(
    vector_repo: Path,
) -> None:
    config = vector_repo / ".gepa" / "gepa.toml"
    config.write_text(
        _config().replace(
            'dataset = ".gepa/dataset.jsonl"\n',
            'dataset = ".gepa/dataset.jsonl"\n'
            'validation_dataset = ".gepa/validation.jsonl"\n',
        ),
        encoding="utf-8",
    )
    (vector_repo / ".gepa" / "validation.jsonl").write_text(
        json.dumps(
            {
                "name": "secret-vector-validation",
                "inputs": "private",
                "expected_output": "good",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _git(vector_repo, "add", ".gepa/gepa.toml", ".gepa/validation.jsonl")
    _git(vector_repo, "commit", "-m", "Configure held-out validation")

    payload, state = _start(vector_repo)
    run_id = str(payload["run_id"])

    assert state.validation_evaluations == 1
    vectors = VectorRecordStore(vector_records_path(run_id, vector_repo)).records()
    assert len(vectors) == len(state.reflection_baseline_samples)
    run_root = vector_repo / ".gepa" / "runs" / run_id
    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in run_root.rglob("*")
        if path.is_file()
    )
    assert "secret-vector-validation" not in persisted


def test_vector_validation_uses_comparator_ranking_without_persisting_detail(
    vector_repo: Path,
) -> None:
    _configure_validation(vector_repo, pinned_scorer=True)

    payload, _ = _start(vector_repo, lanes=3)
    run_id = str(payload["run_id"])
    resolved: dict[str, LaneState] = {}
    for lane, value in (
        ("lane-1", "good\n"),
        ("lane-2", "preferred\n"),
        ("lane-3", "preferred\n"),
    ):
        resolved[lane] = _continue_vector_lane(vector_repo, run_id, lane, value)

    selected = _run(
        "--gepa-dir",
        str(vector_repo / ".gepa"),
        "run",
        "select",
        "--run-id",
        run_id,
    )

    assert selected.exit_code == 0, selected.output
    final = RunState.from_dict(
        json.loads((vector_repo / ".gepa" / "runs" / run_id / "state.json").read_text())
    )
    assert final.best_commit_sha == resolved["lane-2"].candidate_sha
    assert final.best_commit_sha != resolved["lane-3"].candidate_sha
    assert final.validation_evaluations == 5
    validation_rows = ParetoLog(run_id, vector_repo).validation_rows()
    assert len(validation_rows) == 5
    assert all(row.per_case_scores == {} for row in validation_rows)

    run_root = vector_repo / ".gepa" / "runs" / run_id
    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in run_root.rglob("*")
        if path.is_file()
    )
    assert "secret-vector-validation" not in persisted
    assert "held_out_private_detail" not in persisted


def test_vector_validation_resume_restarts_one_comparable_round(
    vector_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pydantic_ai_gepa.cli.select as select_module

    _configure_validation(vector_repo, pinned_scorer=True)
    payload, _ = _start(vector_repo, lanes=2)
    run_id = str(payload["run_id"])
    resolved = {
        "lane-1": _continue_vector_lane(vector_repo, run_id, "lane-1", "good\n"),
        "lane-2": _continue_vector_lane(vector_repo, run_id, "lane-2", "preferred\n"),
    }

    original_checkpoint = select_module._checkpoint
    crashed = False

    def crash_after_first_lane_result(
        state: Any,
        workspace_root: Path,
        phase: str | None,
        context: dict[str, Any] | None,
    ) -> Any:
        nonlocal crashed
        result = original_checkpoint(state, workspace_root, phase, context)
        if (
            not crashed
            and isinstance(context, dict)
            and len(context.get("validation_results", {})) == 1
        ):
            crashed = True
            raise RuntimeError("simulated crash during vector validation")
        return result

    monkeypatch.setattr(select_module, "_checkpoint", crash_after_first_lane_result)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(
            "--gepa-dir",
            str(vector_repo / ".gepa"),
            "run",
            "select",
            "--run-id",
            run_id,
        )

    monkeypatch.setattr(select_module, "_checkpoint", original_checkpoint)
    resumed = _run(
        "--gepa-dir",
        str(vector_repo / ".gepa"),
        "run",
        "select",
        "--run-id",
        run_id,
    )

    assert resumed.exit_code == 0, resumed.output
    final = RunState.from_dict(
        json.loads((vector_repo / ".gepa" / "runs" / run_id / "state.json").read_text())
    )
    assert final.best_commit_sha == resolved["lane-2"].candidate_sha
    assert final.validation_evaluations == 6
    assert len(ParetoLog(run_id, vector_repo).validation_rows()) == 6


def test_validation_ranking_key_requires_finite_non_boolean_numbers() -> None:
    assert _numeric_ranking_key([1, 2.5], label="test") == (1.0, 2.5)
    for invalid in ([True], [float("inf")], [10**1000], ["1"]):
        with pytest.raises(typer.BadParameter, match="ranking_key"):
            _numeric_ranking_key(invalid, label="test")


def test_candidate_gate_allows_meta_files_and_parses_rename_paths(
    vector_repo: Path,
) -> None:
    config = vector_repo / ".gepa" / "gepa.toml"
    config.write_text(
        _config(components=("score.txt", "score-new.txt")), encoding="utf-8"
    )
    _git(vector_repo, "add", str(config.relative_to(vector_repo)))
    _git(vector_repo, "commit", "-m", "Declare rename target")
    payload, run_state = _start(vector_repo)
    run_id = str(payload["run_id"])
    state = load_lane_state(vector_repo, run_id, "lane-1")
    worktree = Path(str(state.worktree_path))
    _git(worktree, "mv", "score.txt", "score-new.txt")
    (worktree / "prediction.json").write_text("{}\n", encoding="utf-8")
    assert (
        _candidate_gate(workspace_root=vector_repo, run_state=run_state, state=state)
        is None
    )


def test_foreground_continue_enforces_candidate_gate(vector_repo: Path) -> None:
    payload, _ = _start(vector_repo)
    run_id = str(payload["run_id"])
    state = load_lane_state(vector_repo, run_id, "lane-1")
    (Path(str(state.worktree_path)) / "unauthorized.py").write_text(
        "BAD = True\n", encoding="utf-8"
    )
    rows_before = ParetoLog(run_id, vector_repo).count_rows()
    result = _run(
        "--gepa-dir",
        str(vector_repo / ".gepa"),
        "lane",
        "continue",
        "lane-1",
        "--run-id",
        run_id,
        "--foreground",
    )
    assert result.exit_code == 1
    assert "candidate review failed" in result.output
    assert ParetoLog(run_id, vector_repo).count_rows() == rows_before
    journal = [
        json.loads(line)
        for line in journal_path(vector_repo).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rejection = next(
        row for row in journal if row.get("kind") == "candidate_review_rejection"
    )
    assert "unauthorized.py" in rejection["diff"]
    assert "BAD = True" in rejection["diff"]


def test_probe_diffs_against_full_inventory_incumbent(vector_repo: Path) -> None:
    payload, _ = _start(vector_repo)
    run_id = str(payload["run_id"])
    state = load_lane_state(vector_repo, run_id, "lane-1")
    (Path(str(state.worktree_path)) / "score.txt").write_text(
        "good\n", encoding="utf-8"
    )
    rows_before = ParetoLog(run_id, vector_repo).count_rows()
    result = _run(
        "--gepa-dir",
        str(vector_repo / ".gepa"),
        "probe",
        "--case",
        "case-1",
        "--lane",
        "lane-1",
        "--run-id",
        run_id,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["changes"]["quality"] == {"before": "fail", "after": "pass"}
    assert Path(payload["receipt_path"]).is_file()
    assert ParetoLog(run_id, vector_repo).count_rows() == rows_before
    journal = [
        json.loads(line)
        for line in journal_path(vector_repo).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    receipt = next(row for row in journal if row.get("kind") == "probe_receipt")
    assert receipt["probe_row_id"]
    assert receipt["changes"] == payload["changes"]


def test_probe_runs_candidate_review_before_rollout(vector_repo: Path) -> None:
    config = vector_repo / ".gepa" / "gepa.toml"
    config.write_text(_config(reviewer=True, pinned_scorer=True), encoding="utf-8")
    _git(vector_repo, "add", str(config.relative_to(vector_repo)))
    _git(vector_repo, "commit", "-m", "Enable vector reviewer")
    payload, _ = _start(vector_repo)
    run_id = str(payload["run_id"])
    state = load_lane_state(vector_repo, run_id, "lane-1")
    worktree = Path(str(state.worktree_path))
    (worktree / "score.txt").write_text("blocked\n", encoding="utf-8")
    rows_before = len(
        VectorRecordStore(vector_records_path(run_id, vector_repo)).records()
    )

    result = _run(
        "--gepa-dir",
        str(vector_repo / ".gepa"),
        "probe",
        "--case",
        "case-1",
        "--lane",
        "lane-1",
        "--run-id",
        run_id,
    )

    assert result.exit_code == 1
    assert "candidate review failed" in result.output
    assert (
        len(VectorRecordStore(vector_records_path(run_id, vector_repo)).records())
        == rows_before
    )
    journal = [
        json.loads(line)
        for line in journal_path(vector_repo).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row.get("kind") == "probe_review_rejection" for row in journal)


def test_probe_allowance_is_durable_per_lease(vector_repo: Path) -> None:
    config = vector_repo / ".gepa" / "gepa.toml"
    config.write_text(_config(probe_allowance_per_lease=1), encoding="utf-8")
    _git(vector_repo, "add", str(config.relative_to(vector_repo)))
    _git(vector_repo, "commit", "-m", "Limit probe allowance")
    payload, _ = _start(vector_repo)
    run_id = str(payload["run_id"])
    state = load_lane_state(vector_repo, run_id, "lane-1")
    (Path(str(state.worktree_path)) / "score.txt").write_text(
        "good\n", encoding="utf-8"
    )

    first = _run(
        "--gepa-dir",
        str(vector_repo / ".gepa"),
        "probe",
        "--case",
        "case-1",
        "--lane",
        "lane-1",
        "--run-id",
        run_id,
    )
    second = _run(
        "--gepa-dir",
        str(vector_repo / ".gepa"),
        "probe",
        "--case",
        "case-2",
        "--lane",
        "lane-1",
        "--run-id",
        run_id,
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code != 0
    assert "Probe allowance exhausted" in second.output
    journal = [
        json.loads(line)
        for line in journal_path(vector_repo).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    debits = [row for row in journal if row.get("kind") == "probe_budget_debit"]
    assert len(debits) == 1
    assert debits[0]["lease_epoch"] == state.lease_epoch


def test_receipt_gate_requires_exact_case_key_and_fixed_direction(
    vector_repo: Path,
) -> None:
    config = vector_repo / ".gepa" / "gepa.toml"
    config.write_text(_config(receipt=True), encoding="utf-8")
    _git(vector_repo, "add", str(config.relative_to(vector_repo)))
    _git(vector_repo, "commit", "-m", "Require probe receipt")
    payload, run_state = _start(vector_repo)
    run_id = str(payload["run_id"])
    state = load_lane_state(vector_repo, run_id, "lane-1")
    worktree = Path(str(state.worktree_path))
    (worktree / "score.txt").write_text("good\n", encoding="utf-8")
    (worktree / "prediction.json").write_text(
        json.dumps(
            {
                "predictions": [
                    {"key": "quality", "case": "case-1", "direction": "fail_to_pass"}
                ]
            }
        ),
        encoding="utf-8",
    )
    receipts = probe_receipts_dir(run_id, vector_repo)
    receipts.mkdir(parents=True)
    receipt_path = receipts / "wrong.json"
    base = {
        "run_id": run_id,
        "probe_row_id": "probe-row-1",
        "candidate_component_hash": component_hash(worktree, ("score.txt",)),
        "lane": "lane-1",
        "iteration": state.iteration,
        "proof": {"key": "quality", "case": "case-2", "direction": "fail_to_pass"},
        "changes": {"quality": {"before": "pass", "after": "fail"}},
    }
    receipt_path.write_text(json.dumps(base), encoding="utf-8")
    rejected = _candidate_gate(
        workspace_root=vector_repo, run_state=run_state, state=state
    )
    assert rejected is not None
    assert "No matching probe receipt" in rejected.review_findings[0]["explanation"]

    base["proof"] = {"key": "quality", "case": "case-1", "direction": "fail_to_pass"}
    base["changes"] = {"quality": {"before": "fail", "after": "pass"}}
    receipt_path.write_text(json.dumps(base), encoding="utf-8")
    rejected = _candidate_gate(
        workspace_root=vector_repo, run_state=run_state, state=state
    )
    assert rejected is not None
    assert "No matching probe receipt" in rejected.review_findings[0]["explanation"]
    with journal_path(vector_repo).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "probe_receipt", **base}) + "\n")
    assert (
        _candidate_gate(workspace_root=vector_repo, run_state=run_state, state=state)
        is None
    )


def test_handoff_refuses_component_mutation_after_parent_gate(
    vector_repo: Path,
) -> None:
    payload, _ = _start(vector_repo)
    run_id = str(payload["run_id"])
    state = load_lane_state(vector_repo, run_id, "lane-1")
    worktree = Path(str(state.worktree_path))
    LaneState(
        **{
            **state.to_dict(),
            "status": "leased",
            "lease_epoch": 7,
            "lease_purpose": "handoff",
            "lease_expires_at": "2999-01-01T00:00:00+00:00",
            "handoff_component_hash": component_hash(worktree, ("score.txt",)),
        }
    ).save(vector_repo, run_id)
    (worktree / "score.txt").write_text("good\n", encoding="utf-8")
    rows_before = ParetoLog(run_id, vector_repo).count_rows()

    result = _run(
        "--gepa-dir",
        str(vector_repo / ".gepa"),
        "lane",
        "continue",
        "lane-1",
        "--run-id",
        run_id,
        "--foreground",
        "--handoff-lease-epoch",
        "7",
    )

    assert result.exit_code == 1
    assert "component hash changed" in result.output
    assert ParetoLog(run_id, vector_repo).count_rows() == rows_before
    assert (
        load_lane_state(vector_repo, run_id, "lane-1").status == "paused_for_reflection"
    )
    journal = [
        json.loads(line)
        for line in journal_path(vector_repo).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row.get("kind") == "handoff_component_hash_mismatch" for row in journal)


def test_vector_pinned_packets_and_reflector_artifacts_hide_scores(
    vector_repo: Path,
) -> None:
    config = vector_repo / ".gepa" / "gepa.toml"
    config.write_text(_config(pinned_scorer=True), encoding="utf-8")
    _git(vector_repo, "add", str(config.relative_to(vector_repo)))
    _git(vector_repo, "commit", "-m", "Pin vector scorer")
    payload, run_state = _start(vector_repo)
    run_id = str(payload["run_id"])
    state = load_lane_state(vector_repo, run_id, "lane-1")
    packet = json.loads(
        write_packet(
            vector_repo,
            run_state,
            "lane-1",
            state.iteration,
            Path(str(state.worktree_path)),
            str(state.branch),
        ).read_text(encoding="utf-8")
    )
    assert "mean_score" not in packet["baseline"]
    assert "samples" not in packet["baseline"]

    record = EvaluationRecord("case-1", 0.25, "detail", {})
    assert "score" not in _format_failures([record], redact_scores=True)
    assert "score 0.250" in _format_failures([record])

    class Trace:
        def to_reflective_record(self) -> dict[str, str]:
            return {"safe": "context"}

    trace_path = _write_trace_file(
        path=vector_repo / ".gepa" / "runs" / run_id / "redacted.jsonl",
        records=[EvaluationRecord("case-1", 0.25, "detail", {"trajectory": Trace()})],
        redact_scores=True,
    )
    assert trace_path is not None
    assert "score" not in json.loads(trace_path.read_text(encoding="utf-8"))


def test_infra_retry_does_not_consume_scored_escalation_slot(
    vector_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic_ai_gepa.cli import lanes as lanes_module

    payload, run_state = _start(vector_repo, repetitions=2)
    run_id = str(payload["run_id"])
    state = load_lane_state(vector_repo, run_id, "lane-1")
    worktree = Path(str(state.worktree_path))
    (worktree / "score.txt").write_text("good\n", encoding="utf-8")
    store = VectorRecordStore(vector_records_path(run_id, vector_repo))
    incumbent = store.records()[0]
    attempts = 0
    comparator_attempts: list[int] = []

    def fake_eval(**kwargs: object) -> EvalOutcome:
        nonlocal attempts
        attempts += 1
        candidate_id = git_candidate_state(
            Path(str(kwargs["candidate_root"]))
        ).candidate_id
        summary = {
            "candidate_id": candidate_id,
            "mean_score": 1.0,
            "report_path": str(vector_repo / f"report-{attempts}.md"),
            "trace_path": None,
        }
        if attempts == 1:
            record = EvaluationRecord(
                "case-1",
                0.0,
                None,
                {
                    "output": RolloutOutput.from_error(
                        RuntimeError("outage"), kind="system"
                    )
                },
            )
            return EvalOutcome([record], summary, Path(summary["report_path"]), None)
        vector = VectorRecord(
            key=replace(
                incumbent.key,
                candidate_hash=candidate_id,
                repetition=attempts - 1,
            ),
            assertions=incumbent.assertions,
            latency=incumbent.latency,
            display_score=1.0,
        )
        store.append(vector)
        summary["vector_record"] = vector.to_dict()
        return EvalOutcome(
            [
                EvaluationRecord(
                    "case-1", 1.0, None, {"output": RolloutOutput.from_success("good")}
                )
            ],
            summary,
            Path(summary["report_path"]),
            None,
        )

    class Comparator:
        def compare(self, request: VectorComparisonRequest) -> VectorComparison:
            comparator_attempts.append(request.attempt)
            return VectorComparison(
                "accepted" if len(request.candidate) >= 3 else "needs_escalation",
                display_score=1.0,
            )

    monkeypatch.setattr(lanes_module, "run_eval_once", fake_eval)
    monkeypatch.setattr(
        lanes_module, "resolve_vector_comparator", lambda *a, **k: Comparator()
    )
    final = _run_lane_eval_loop(
        workspace_root=vector_repo, run_state=run_state, lane_state=state
    )
    assert attempts == 4
    assert tuple(final.eval_samples) == (1.0, 1.0, 1.0)
    assert comparator_attempts == [3, 4]
    assert final.verdict == "accepted"


def test_periodic_rebaseline_is_paired_journaled_and_never_reverts_incumbent(
    vector_repo: Path,
) -> None:
    config = vector_repo / ".gepa" / "gepa.toml"
    config.write_text(
        _config(rebaseline_interval=1, pinned_scorer=True), encoding="utf-8"
    )
    _git(vector_repo, "add", str(config.relative_to(vector_repo)))
    _git(vector_repo, "commit", "-m", "Enable periodic rebaseline")

    payload, initial = _start(vector_repo)
    run_id = str(payload["run_id"])
    run_start = initial.run_start_baseline
    assert run_start is not None
    assert run_start["candidate_id"] == initial.reflection_baseline_candidate_id
    assert set(run_start["component_hashes"]) == {"score.txt"}
    assert isinstance(run_start["component_hashes"]["score.txt"], str)
    assert len(run_start["vector_record_keys"]) == initial.acceptance_max_repetitions

    lane = load_lane_state(vector_repo, run_id, "lane-1")
    worktree = Path(str(lane.worktree_path))
    (worktree / "score.txt").write_text("good\n", encoding="utf-8")
    previous_cwd = Path.cwd()
    try:
        os.chdir(worktree)
        continued = _run(
            "--gepa-dir",
            str(vector_repo / ".gepa"),
            "lane",
            "continue",
            "lane-1",
            "--run-id",
            run_id,
            "--foreground",
        )
    finally:
        os.chdir(previous_cwd)
    assert continued.exit_code == 0, continued.output
    lane = load_lane_state(vector_repo, run_id, "lane-1")
    candidate_context = json.loads(Path(str(lane.comparison_path)).read_text())[
        "detail"
    ]["context"]
    assert candidate_context["accepted_promotion_count"] == 0
    assert (
        candidate_context["run_start_baseline"]["candidate_id"]
        == run_start["candidate_id"]
    )

    selected = _run(
        "--gepa-dir", str(vector_repo / ".gepa"), "run", "select", "--run-id", run_id
    )
    assert selected.exit_code == 0, selected.output
    final = RunState.from_dict(
        json.loads((vector_repo / ".gepa" / "runs" / run_id / "state.json").read_text())
    )
    assert final.accepted_promotion_count == 1
    assert final.best_commit_sha == lane.candidate_sha
    assert final.run_start_baseline == run_start

    journal = [
        json.loads(line)
        for line in (vector_repo / ".gepa" / "journal.jsonl").read_text().splitlines()
        if line.strip()
    ]
    promotions = [
        row
        for row in journal
        if row.get("kind") == "accepted_promotion" and row.get("run_id") == run_id
    ]
    assert [row["promotion_count"] for row in promotions] == [1]
    rebaselines = [
        row
        for row in journal
        if row.get("kind") == "run_start_rebaseline" and row.get("run_id") == run_id
    ]
    assert len(rebaselines) == 1
    assert rebaselines[0]["outcome"] == "failed"
    assert rebaselines[0]["incumbent_candidate_id"] == final.best_candidate_id
    assert rebaselines[0]["run_start_baseline"] == run_start
    rebaseline_context = rebaselines[0]["comparison"]["detail"]["context"]
    assert rebaseline_context["accepted_promotion_count"] == 1
    assert rebaseline_context["run_start_baseline"] == run_start
    assert rebaselines[0]["comparison"]["detail"]["incumbent_records"] == 2
    assert rebaselines[0]["comparison"]["detail"]["candidate_records"] == 2


def test_promotion_counter_is_not_doubled_when_select_resumes_after_a_crash(
    vector_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pydantic_ai_gepa.cli.select as select_module

    config = vector_repo / ".gepa" / "gepa.toml"
    config.write_text(_config(pinned_scorer=True), encoding="utf-8")
    _git(vector_repo, "add", str(config.relative_to(vector_repo)))
    _git(vector_repo, "commit", "-m", "Use pinned vector scorer")
    payload, _ = _start(vector_repo)
    run_id = str(payload["run_id"])
    lane = load_lane_state(vector_repo, run_id, "lane-1")
    worktree = Path(str(lane.worktree_path))
    (worktree / "score.txt").write_text("good\n", encoding="utf-8")
    previous_cwd = Path.cwd()
    try:
        os.chdir(worktree)
        continued = _run(
            "--gepa-dir",
            str(vector_repo / ".gepa"),
            "lane",
            "continue",
            "lane-1",
            "--run-id",
            run_id,
            "--foreground",
        )
    finally:
        os.chdir(previous_cwd)
    assert continued.exit_code == 0, continued.output

    original_reset = select_module._reset_primary_to

    def crash_after_reset(root: Path, commit_sha: str) -> None:
        original_reset(root, commit_sha)
        raise RuntimeError("simulated crash after durable promotion journal")

    monkeypatch.setattr(select_module, "_reset_primary_to", crash_after_reset)
    crashed = CliRunner().invoke(
        gepa_app,
        [
            "--gepa-dir",
            str(vector_repo / ".gepa"),
            "run",
            "select",
            "--run-id",
            run_id,
        ],
    )
    assert crashed.exit_code == 1
    journal = [
        json.loads(line)
        for line in (vector_repo / ".gepa" / "journal.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [
        row["promotion_count"]
        for row in journal
        if row.get("kind") == "accepted_promotion" and row.get("run_id") == run_id
    ] == [1]
    assert not [
        row
        for row in journal
        if row.get("kind") == "run_start_rebaseline" and row.get("run_id") == run_id
    ]

    monkeypatch.setattr(select_module, "_reset_primary_to", original_reset)
    resumed = _run(
        "--gepa-dir", str(vector_repo / ".gepa"), "run", "select", "--run-id", run_id
    )
    assert resumed.exit_code == 0, resumed.output
    final = RunState.from_dict(
        json.loads((vector_repo / ".gepa" / "runs" / run_id / "state.json").read_text())
    )
    assert final.accepted_promotion_count == 1
    journal = [
        json.loads(line)
        for line in (vector_repo / ".gepa" / "journal.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [
        row["promotion_count"]
        for row in journal
        if row.get("kind") == "accepted_promotion" and row.get("run_id") == run_id
    ] == [1]

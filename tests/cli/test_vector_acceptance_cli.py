"""Artifact-level regressions for the assertion-vector lane and probe flow."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import pytest
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.candidates import git_candidate_state
from pydantic_ai_gepa.cli.eval import EvalOutcome
from pydantic_ai_gepa.cli.lanes import (
    _candidate_gate,
    _run_lane_eval_loop,
    load_lane_state,
)
from pydantic_ai_gepa.cli.layout import probe_receipts_dir, vector_records_path
from pydantic_ai_gepa.cli.probe import component_hash
from pydantic_ai_gepa.cli.run import RunState
from pydantic_ai_gepa.cli.runs import ParetoLog
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
) -> str:
    files = ", ".join(json.dumps(item) for item in components)
    rebaseline = (
        "" if rebaseline_interval is None else f"rebaseline_interval = {rebaseline_interval}\n"
    )
    pinned = "pinned_scorer = true\n" if pinned_scorer else ""
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
        "    status = 'pass' if output == 'good' else 'fail'\n"
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
        "        verdict = 'rejected' if context.get('comparison_kind') == 'run_start_rebaseline' else 'accepted'\n"
        "        return VectorComparison(verdict, display_score=1.0, detail={\n"
        "            'context': context,\n"
        "            'incumbent_records': len(request.incumbent),\n"
        "            'candidate_records': len(request.candidate),\n"
        "        })\n"
        "def make_comparator():\n"
        "    return Comparator()\n",
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


def _start(repo: Path, *, repetitions: int = 1) -> tuple[dict[str, object], RunState]:
    result = _run(
        "--gepa-dir",
        str(repo / ".gepa"),
        "run",
        "start",
        "--lanes",
        "1",
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
    assert (
        _candidate_gate(workspace_root=vector_repo, run_state=run_state, state=state)
        is None
    )


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
    candidate_context = json.loads(Path(str(lane.comparison_path)).read_text())["detail"][
        "context"
    ]
    assert candidate_context["accepted_promotion_count"] == 0
    assert candidate_context["run_start_baseline"]["candidate_id"] == run_start["candidate_id"]

    selected = _run(
        "--gepa-dir", str(vector_repo / ".gepa"), "run", "select", "--run-id", run_id
    )
    assert selected.exit_code == 0, selected.output
    final = RunState.from_dict(
        json.loads(
            (vector_repo / ".gepa" / "runs" / run_id / "state.json").read_text()
        )
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
        json.loads(
            (vector_repo / ".gepa" / "runs" / run_id / "state.json").read_text()
        )
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

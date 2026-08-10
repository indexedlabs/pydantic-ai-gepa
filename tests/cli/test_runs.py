"""Tests for `pydantic_ai_gepa.cli.runs`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pydantic_ai_gepa.cli.layout import ensure_layout, new_run_id, run_dir
from pydantic_ai_gepa.cli.runs import (
    MinibatchStore,
    ParetoLog,
    ParetoRow,
    current_commit_sha,
    new_candidate_id,
    utc_now_iso,
)


# ---------- minibatch ----------


def test_minibatch_sample_is_deterministic(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    run = new_run_id()
    store = MinibatchStore(run, tmp_path)
    case_ids = [f"case-{i}" for i in range(20)]

    mb1 = store.sample(case_ids, size=5, seed=42, epoch=0)
    # Re-sample with the same params produces the same id and selection.
    mb2 = store.sample(case_ids, size=5, seed=42, epoch=0)
    assert mb1.id == mb2.id
    assert mb1.case_ids == mb2.case_ids


def test_minibatch_seed_and_epoch_differentiate(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    run = new_run_id()
    store = MinibatchStore(run, tmp_path)
    case_ids = [f"case-{i}" for i in range(20)]

    mb_seed0 = store.sample(case_ids, size=5, seed=0, epoch=0)
    mb_seed1 = store.sample(case_ids, size=5, seed=1, epoch=0)
    mb_epoch1 = store.sample(case_ids, size=5, seed=0, epoch=1)

    assert mb_seed0.id != mb_seed1.id
    assert mb_seed0.id != mb_epoch1.id


def test_minibatch_round_trip(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    run = new_run_id()
    store = MinibatchStore(run, tmp_path)
    case_ids = [f"case-{i}" for i in range(10)]

    mb = store.sample(case_ids, size=4, seed=7, epoch=2)
    loaded = store.load(mb.id)
    assert loaded.id == mb.id
    assert loaded.case_ids == mb.case_ids
    assert loaded.seed == 7
    assert loaded.epoch == 2


def test_minibatch_size_caps_to_pool(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    run = new_run_id()
    store = MinibatchStore(run, tmp_path)
    mb = store.sample(["a", "b", "c"], size=100, seed=0, epoch=0)
    assert sorted(mb.case_ids) == ["a", "b", "c"]


def test_minibatch_list_ids(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    run = new_run_id()
    store = MinibatchStore(run, tmp_path)
    store.sample(["a", "b", "c"], size=2, seed=0, epoch=0)
    store.sample(["a", "b", "c"], size=2, seed=1, epoch=0)
    assert len(store.list_ids()) == 2


# ---------- lane-aware ledger (task ae6 / dec-msy) ----------


def _ledger_row(candidate_id: str = "cand", lane: str | None = None) -> ParetoRow:
    return ParetoRow(
        candidate_id=candidate_id,
        commit_sha=None,
        component_overrides_id=None,
        minibatch_id="mb-1",
        per_case_scores={"case-a": 1.0},
        mean_score=1.0,
        status="evaluated",
        summary="summary",
        timestamp=utc_now_iso(),
        lane=lane,
    )


def test_pareto_row_lane_round_trip() -> None:
    row = _ledger_row(lane="lane-a")
    loaded = ParetoRow.from_dict(row.to_dict())
    assert loaded.lane == "lane-a"
    assert row.to_dict()["lane"] == "lane-a"


def test_pareto_row_single_path_lane_is_null() -> None:
    row = _ledger_row()
    assert row.to_dict()["lane"] is None
    assert ParetoRow.from_dict(row.to_dict()).lane is None


def test_pareto_row_pre_lane_rows_parse(tmp_path: Path) -> None:
    """Rows appended before lanes existed (no "lane" key) still load."""
    ensure_layout(tmp_path)
    run = new_run_id()
    log = ParetoLog(run, tmp_path)
    payload = _ledger_row().to_dict()
    del payload["lane"]
    log.path.parent.mkdir(parents=True, exist_ok=True)
    log.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    rows = log.iter_rows()
    assert len(rows) == 1
    assert rows[0].lane is None


def test_pareto_log_tolerates_trailing_torn_line(tmp_path: Path) -> None:
    """A writer killed mid-append leaves one trailing partial line: readers
    skip it (with a warning) and count_rows excludes it — the incomplete eval
    did not spend budget."""
    ensure_layout(tmp_path)
    run = new_run_id()
    log = ParetoLog(run, tmp_path)
    log.append(_ledger_row("cand-a"))
    log.append(_ledger_row("cand-b"))

    # Simulate a killed writer: half a row with no trailing newline.
    with log.path.open("a", encoding="utf-8") as fh:
        fh.write('{"candidate_id": "cand-torn", "mean_sc')

    assert log.count_rows() == 2  # the torn line is not a completed eval
    rows = log.iter_rows()
    assert [row.candidate_id for row in rows] == ["cand-a", "cand-b"]


def test_pareto_log_torn_line_never_poisons_next_append(tmp_path: Path) -> None:
    """The append after a torn line terminates the partial row first, so the
    healthy rows before and after stay readable and correctly counted."""
    ensure_layout(tmp_path)
    run = new_run_id()
    log = ParetoLog(run, tmp_path)
    log.append(_ledger_row("cand-a"))
    log.append(_ledger_row("cand-b"))
    with log.path.open("a", encoding="utf-8") as fh:
        fh.write('{"candidate_id": "cand-torn", "mean_sc')

    log.append(_ledger_row("cand-c"))

    # The torn fragment is quarantined on its own line and skipped; every
    # healthy row — before AND after the kill — survives.
    assert log.count_rows() == 3
    rows = log.iter_rows()
    assert [row.candidate_id for row in rows] == ["cand-a", "cand-b", "cand-c"]


def test_pareto_log_unparseable_lines_are_skipped_with_warning(
    tmp_path: Path, capsys: "pytest.CaptureFixture[str]"
) -> None:
    """Corrupt or non-row lines anywhere in the log are skipped with a
    stderr warning, never fatal — the ledger must stay readable."""
    ensure_layout(tmp_path)
    run = new_run_id()
    log = ParetoLog(run, tmp_path)
    log.append(_ledger_row("cand-a"))
    with log.path.open("a", encoding="utf-8") as fh:
        fh.write("{not json}\n")
        fh.write('{"unrelated": "valid json, not a row"}\n')
    log.append(_ledger_row("cand-b"))

    rows = log.iter_rows()
    assert [row.candidate_id for row in rows] == ["cand-a", "cand-b"]
    assert log.count_rows() == 2
    err = capsys.readouterr().err
    assert err.count("ignoring unparseable pareto row") == 2


def test_pareto_row_maps_to_artifact_paths(tmp_path: Path) -> None:
    """Slice-4 contract: a row's fields alone reconstruct its report/trace
    artifacts (glob on eval_id + candidate_id), no iteration field needed."""
    ensure_layout(tmp_path)
    run = new_run_id()
    eval_id = "abc123de"
    candidate_id = "cand-x"
    row = ParetoRow(
        candidate_id=candidate_id,
        commit_sha=None,
        component_overrides_id=None,
        minibatch_id="mb-1",
        per_case_scores={"case-a": 1.0},
        mean_score=1.0,
        status="evaluated",
        summary="s",
        timestamp=utc_now_iso(),
        lane="lane-1",
        extra={"eval_id": eval_id},
    )
    log = ParetoLog(run, tmp_path)
    log.append(row)

    reports = run_dir(run, tmp_path) / "reports"
    reports.mkdir(parents=True)
    report = reports / f"0007-{eval_id}-{candidate_id}.md"
    report.write_text("report", encoding="utf-8")
    traces = run_dir(run, tmp_path) / "traces" / "minibatches" / "mb-1"
    traces.mkdir(parents=True)
    trace = traces / f"0007-{eval_id}-{candidate_id}.jsonl"
    trace.write_text("{}\n", encoding="utf-8")

    loaded = log.iter_rows()[0]
    report_hits = list(
        (run_dir(run, tmp_path) / "reports").glob(
            f"*-{loaded.extra['eval_id']}-{loaded.candidate_id}.md"
        )
    )
    trace_hits = list(
        (run_dir(run, tmp_path) / "traces" / "minibatches" / loaded.minibatch_id).glob(
            f"*-{loaded.extra['eval_id']}-{loaded.candidate_id}.jsonl"
        )
    )
    assert report_hits == [report]
    assert trace_hits == [trace]


def test_pareto_log_concurrent_appends(tmp_path: Path) -> None:
    """N producers appending concurrently: row count equals completed evals."""
    import threading

    ensure_layout(tmp_path)
    run = new_run_id()
    log = ParetoLog(run, tmp_path)

    def append_n(lane: str, n: int) -> None:
        for i in range(n):
            log.append(_ledger_row(candidate_id=f"{lane}-{i}", lane=lane))

    threads = [
        threading.Thread(target=append_n, args=(f"lane-{t}", 10)) for t in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert log.count_rows() == 40
    rows = log.iter_rows()
    assert len(rows) == 40
    lanes = {row.lane for row in rows}
    assert lanes == {"lane-0", "lane-1", "lane-2", "lane-3"}


# ---------- Pareto log ----------


def _row(
    candidate_id: str, scores: dict[str, float], status: str = "evaluated"
) -> ParetoRow:
    return ParetoRow(
        candidate_id=candidate_id,
        commit_sha="abc1234567",
        component_overrides_id=f"comp-{candidate_id}",
        minibatch_id="mb-1",
        per_case_scores=scores,
        mean_score=sum(scores.values()) / len(scores) if scores else 0.0,
        status=status,
        summary=f"Row {candidate_id}",
        timestamp=utc_now_iso(),
    )


def test_pareto_append_and_iter(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    run = new_run_id()
    log = ParetoLog(run, tmp_path)
    log.append(_row("c1", {"a": 0.5, "b": 0.7}))
    log.append(_row("c2", {"a": 0.6, "b": 0.8}))

    rows = log.iter_rows()
    assert len(rows) == 2
    assert {r.candidate_id for r in rows} == {"c1", "c2"}


def test_pareto_front_simple_domination(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    run = new_run_id()
    log = ParetoLog(run, tmp_path)
    log.append(_row("dominated", {"a": 0.1, "b": 0.2}))
    log.append(_row("dominator", {"a": 0.5, "b": 0.7}))

    front = log.front()
    assert {r.candidate_id for r in front} == {"dominator"}


def test_pareto_front_incomparable_kept(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    run = new_run_id()
    log = ParetoLog(run, tmp_path)
    # All three are mutually incomparable (no row dominates another), so the
    # front keeps every row.
    log.append(_row("c1", {"a": 0.8, "b": 0.3}))
    log.append(_row("c2", {"a": 0.3, "b": 0.8}))
    log.append(_row("c3", {"a": 0.4, "b": 0.4}))

    front = log.front()
    assert {r.candidate_id for r in front} == {"c1", "c2", "c3"}


def test_pareto_front_drops_dominated(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    run = new_run_id()
    log = ParetoLog(run, tmp_path)
    log.append(_row("c1", {"a": 0.8, "b": 0.6}))
    log.append(_row("c2", {"a": 0.3, "b": 0.8}))
    # c3 is dominated by c1 (0.5<=0.8 and 0.5<=0.6 with strict <).
    log.append(_row("c3", {"a": 0.5, "b": 0.5}))

    front = log.front()
    assert {r.candidate_id for r in front} == {"c1", "c2"}


def test_pareto_front_excludes_non_selectable_infrastructure_rows(
    tmp_path: Path,
) -> None:
    ensure_layout(tmp_path)
    run = new_run_id()
    log = ParetoLog(run, tmp_path)
    healthy = _row("healthy", {"a": -1.0})
    infrastructure_failure = ParetoRow(
        **{
            **_row("failed", {"a": 0.0}, status="infrastructure_failure").to_dict(),
            "extra": {
                "outcome": "infrastructure_failure",
                "selectable": False,
            },
        }
    )
    log.append(healthy)
    log.append(infrastructure_failure)

    assert log.count_rows() == 2
    assert [row.candidate_id for row in log.selectable_rows()] == ["healthy"]
    assert [row.candidate_id for row in log.front()] == ["healthy"]


def test_pareto_persists_full_schema(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    run = new_run_id()
    log = ParetoLog(run, tmp_path)
    log.append(_row("c1", {"a": 0.5}))

    raw = log.path.read_text(encoding="utf-8").strip()
    data = json.loads(raw)
    assert set(data.keys()) >= {
        "candidate_id",
        "commit_sha",
        "component_overrides_id",
        "minibatch_id",
        "per_case_scores",
        "mean_score",
        "status",
        "summary",
        "timestamp",
    }


def test_pareto_path_under_run_dir(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    run = new_run_id()
    log = ParetoLog(run, tmp_path)
    assert run_dir(run, tmp_path) in log.path.parents


def test_pareto_count_rows(tmp_path: Path) -> None:
    """count_rows must match iter_rows without parsing each row."""
    ensure_layout(tmp_path)
    run = new_run_id()
    log = ParetoLog(run, tmp_path)
    assert log.count_rows() == 0
    log.append(_row("c1", {"a": 0.5}))
    log.append(_row("c2", {"a": 0.6}))
    log.append(_row("c3", {"a": 0.7}))
    assert log.count_rows() == 3
    assert log.count_rows() == len(log.iter_rows())


def test_current_commit_sha_outside_git(tmp_path: Path) -> None:
    # tmp_path is not a git repo so we expect None.
    assert current_commit_sha(tmp_path) is None


def test_new_candidate_id_unique() -> None:
    ids = {new_candidate_id() for _ in range(50)}
    assert len(ids) == 50

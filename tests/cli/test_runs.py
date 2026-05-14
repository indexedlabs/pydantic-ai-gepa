"""Tests for `pydantic_ai_gepa.cli.runs`."""

from __future__ import annotations

import json
from pathlib import Path

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

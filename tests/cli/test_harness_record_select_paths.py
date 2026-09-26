"""Held-out select cannot turn lane/view path redirection into privileged I/O."""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil

import pytest
import typer

from pydantic_ai_gepa.cli import harness_record, select, lanes, events
from tests.cli import test_select_cli as flows
from tests.cli import test_harness_record_races as races

# Controller tests inject fake scoring and bypass the separate 4923 lane-Git
# mutation refusal (tests/cli/conftest.py). No production switch bypasses it.
git_repo = flows.git_repo
record = races.record


@pytest.fixture
def lane_run(git_repo, monkeypatch):
    source = flows.EVALUATE_MODULE_SOURCE.replace(
        '    path = Path(f"out_{case.name}.txt")',
        '    if case.name == "secret":\n'
        '        return "v" if Path("out_case-2.txt").exists() or Path("out_case-3.txt").exists() else ""\n'
        '    path = Path(f"out_{case.name}.txt")',
    )
    (git_repo / "task_pkg/evaluation.py").write_text(source)
    flows._git(git_repo, "add", "task_pkg/evaluation.py")
    flows._git(git_repo, "commit", "-m", "Fake held-out evaluator")
    private = git_repo.parent / (git_repo.name + "-private")
    private.mkdir()
    dataset = private / "heldout.jsonl"
    dataset.write_text(
        json.dumps({"name": "secret", "inputs": "x", "expected_output": "v"}) + "\n"
    )
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(dataset))
    state = flows._start_lane_run(git_repo, 2)
    run_id = flows._run_id(state)
    monkeypatch.delenv("GEPA_HELDOUT_DATASET")
    first = flows._drive_lane(git_repo, run_id, "lane-1", {"out_case-2.txt": "b\n"})
    second = flows._drive_lane(git_repo, run_id, "lane-2", {"out_case-3.txt": "c\n"})
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(dataset))
    with harness_record.session():
        authority = harness_record.for_run(run_id, git_repo)
        assert authority is not None
        yield authority, first, second


def authority_bytes(record):
    return {p.name: p.read_bytes() for p in record.path.parent.glob("*.json")}


def test_lane_repository_metadata_comes_from_private_run(lane_run):
    record, _, _ = lane_run
    expected = json.loads(record.read("state.json"))
    view = record.directory / "state.json"
    forged = dict(
        expected,
        candidate_prefix="../../outside",
        lane_repository_version=0,
        project_root="/forged/scorer",
    )
    view.write_text(json.dumps(forged))
    root, state = lanes._resolve_lane_run(record.run_id)
    assert root == record.root
    assert state.candidate_prefix == expected["candidate_prefix"]
    assert state.lane_repository_version == 1
    assert state.project_root == expected["project_root"]
    assert json.loads(view.read_text()) == expected


@pytest.mark.parametrize(
    "attack",
    [
        "control",
        "winner-journal",
        "winner-runs",
        "winner-lanes",
        "winner-merge",
        "diffstat-leaf",
        "merge-dir",
        "lane-dir",
        "sentinel-dir",
    ],
)
def test_heldout_select_refuses_or_matches_control(lane_run, attack, tmp_path):
    record, first, second = lane_run
    victim = tmp_path.parent / (tmp_path.name + "-victim")
    victim.mkdir()
    keep = victim / "keep.json"
    keep.write_text("PRECIOUS")
    directory = record.directory
    if attack.startswith("winner-"):
        checkout = Path(first.worktree_path)
        relative = {
            "winner-journal": ".gepa/journal.jsonl",
            "winner-runs": ".gepa/runs",
            "winner-lanes": f".gepa/runs/{record.run_id}/lanes",
            "winner-merge": f".gepa/runs/{record.run_id}/merge_opportunities",
        }[attack]
        path = checkout / relative
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(keep if attack == "winner-journal" else victim)
        flows._git(checkout, "add", "-f", relative)
        flows._git(checkout, "commit", "-m", "Plant workspace link in winner")
        first = replace(first, candidate_sha=flows._git(checkout, "rev-parse", "HEAD"))
        first.save(record.root, record.run_id)
    elif attack != "control":
        relative = {
            "diffstat-leaf": "merge_opportunities/lane-1--lane-2.diffstat",
            "merge-dir": "merge_opportunities",
            "lane-dir": "lanes/lane-1",
            "sentinel-dir": "events/.reaped",
        }[attack]
        path = directory / relative
        if path.is_dir():
            shutil.rmtree(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(keep if attack == "diffstat-leaf" else victim)
    before = authority_bytes(record)
    old_best = json.loads(record.read("state.json"))["best_commit_sha"]
    result = flows._select(record.root, record.run_id)
    assert keep.read_text() == "PRECIOUS"
    assert list(victim.iterdir()) == [keep]
    if attack.startswith("winner-"):
        # An unsafe independent candidate is discarded without resetting the
        # scorer; the other valid lane may still win.
        assert result.exit_code == 0, (result.output, result.exception)
        state = json.loads(record.read("state.json"))
        assert state["best_commit_sha"] == second.candidate_sha
        assert flows._journal_outcomes(
            record.root, record.run_id, lane="lane-1", outcome="invalidated"
        )
    elif attack in {"control", "diffstat-leaf"}:
        assert result.exit_code == 0, (result.output, result.exception)
        state = json.loads(record.read("state.json"))
        # Both untampered and repaired-leaf cases choose the same tied winner.
        assert state["best_commit_sha"] == first.candidate_sha
        assert state["best_mean_score"] == 1.0
        assert state["best_commit_sha"] != second.candidate_sha
        diffstat = directory / "merge_opportunities/lane-1--lane-2.diffstat"
        assert not diffstat.is_symlink()
        assert "out_case-2.txt" in diffstat.read_text()
        assert "out_case-3.txt" in diffstat.read_text()
        assert all(
            p.read_bytes() == content
            for name, content in before.items()
            if not name.endswith(".record.json")
            for p in [record.path.parent / name]
        )
    else:
        assert result.exit_code == 2, (result.output, result.exception)
        assert (
            "unsafe GEPA_DIR path" in result.output
            or "private record missing" in result.output
        )
        assert authority_bytes(record) == before
        assert (
            json.loads(record.path.read_text())["files"]["state.json"]
            == record.load()["files"]["state.json"]
        )
        assert (
            json.loads(record.load()["files"]["state.json"])["best_commit_sha"]
            == old_best
        )


@pytest.mark.parametrize("target", ["record", "index"])
def test_diffstat_leaf_never_overwrites_private_authority(record, target, monkeypatch):
    victim = (
        record.path
        if target == "record"
        else next(record.path.parent.glob("*.index.json"))
    )
    before = authority_bytes(record)
    path = record.directory / "merge_opportunities/lane-a--lane-b.diffstat"
    path.parent.mkdir()
    path.symlink_to(victim)
    accepted = [
        lanes.LaneState(
            lane=name,
            status="awaiting_selection",
            iteration=1,
            branch=name,
            candidate_sha=name,
        )
        for name in ("lane-a", "lane-b")
    ]
    monkeypatch.setattr(select, "_changed_files", lambda root, base, sha: {sha})
    monkeypatch.setattr(select, "_diff_stat", lambda *_: "safe diff")
    from types import SimpleNamespace

    select._emit_merge_opportunities(
        record.root,
        SimpleNamespace(run_id=record.run_id, reflection_baseline_commit_sha="base"),
        {},
        accepted,
    )
    assert authority_bytes(record) == before
    assert not path.is_symlink()
    assert "safe diff" in path.read_text()


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "hardlink"])
def test_owned_leaf_is_repaired_and_run_can_continue(record, kind, tmp_path, capsys):
    record.write("results/owned.json", '{"accepted": false}')
    before = authority_bytes(record)
    path = record.directory / "results/owned.json"
    path.unlink()
    victim = tmp_path / "victim"
    victim.write_text("PRECIOUS")
    if kind == "symlink":
        path.symlink_to(victim)
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    else:
        os.link(victim, path)
    assert record.read("results/owned.json") == '{"accepted": false}'
    assert path.read_text() == '{"accepted": false}'
    assert victim.read_text() == "PRECIOUS"
    assert "restored from the harness record" in capsys.readouterr().err
    assert authority_bytes(record) == before
    record.write("results/next.json", "continued")
    assert record.read("results/next.json") == "continued"


@pytest.mark.parametrize("operation", ["append", "publish"])
def test_hard_linked_public_leaf_is_replaced_without_reading_or_writing_target(
    record, operation
):
    path = record.directory / "reflector_packet.json"
    os.link(record.path, path)
    before = authority_bytes(record)
    harness_record.write_text(
        path, "NEW PUBLIC", root=record.root, append=operation == "append"
    )
    assert authority_bytes(record) == before
    assert path.read_text() == "NEW PUBLIC"
    assert path.stat().st_nlink == 1


def test_lane_lock_and_sentinel_never_follow_links(record):
    state = lanes.LaneState(lane="lane-a", status="awaiting_selection", iteration=1)
    state.save(record.root, record.run_id)
    before = authority_bytes(record)
    lock = (
        lanes.lane_state_path(record.root, record.run_id, state.lane).parent
        / ".lane.lock"
    )
    lock.symlink_to(record.path)
    with pytest.raises(typer.BadParameter, match="private record missing"):
        with lanes._lane_lock(record.root, record.run_id, state.lane):
            pytest.fail("must not follow the lock")
    directory = record.directory / "events"
    directory.mkdir()
    (directory / ".reaped").symlink_to(record.path.parent, target_is_directory=True)
    with pytest.raises(typer.BadParameter, match="private record missing"):
        events._claim_reaper_key(directory, "planted")
    assert authority_bytes(record) == before
    assert not (record.path.parent / "planted").exists()


def test_promotion_does_not_touch_redirected_source_journal(record):
    from pydantic_ai_gepa.cli import lane_repositories

    repositories = lane_repositories.initialize(record.root, record.run_id, record.root)
    journal = record.root / ".gepa/journal.jsonl"
    journal.unlink()
    journal.symlink_to(record.path)
    before = authority_bytes(record)
    repositories.promote(repositories.seed)
    assert authority_bytes(record) == before
    assert journal.is_symlink()
    assert journal.resolve() == record.path


def test_winner_gitlink_under_gepa_is_refused(record):
    from pydantic_ai_gepa.cli import lane_repositories

    sha = flows._git(record.root, "rev-parse", "HEAD")
    flows._git(
        record.root,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{sha},.gepa/linked-module",
    )
    flows._git(record.root, "commit", "-m", "Plant workspace gitlink")
    candidate = flows._git(record.root, "rev-parse", "HEAD")
    before = authority_bytes(record)
    with pytest.raises(typer.BadParameter, match="links or special files"):
        lane_repositories.checkout(
            record.root, candidate, record.root.parent / "refused-checkout", "candidate"
        )
    assert authority_bytes(record) == before


@pytest.mark.parametrize("lane", ["../../escape", "/absolute/escape"])
def test_lane_json_cannot_turn_view_paths_into_outside_paths(record, lane):
    before = authority_bytes(record)
    with pytest.raises(typer.BadParameter):
        lanes.LaneState.from_dict(
            {"lane": lane, "status": "awaiting_selection", "iteration": 1}
        )
    assert authority_bytes(record) == before


def test_run_discovery_refuses_redirected_runs_directory(record):
    from pydantic_ai_gepa.cli.layout import latest_run_id

    runs = record.directory.parent
    runs.rename(runs.with_name("original-runs"))
    runs.symlink_to(record.path.parent, target_is_directory=True)
    before = authority_bytes(record)
    with pytest.raises(typer.BadParameter, match="private record missing"):
        latest_run_id(record.root)
    assert authority_bytes(record) == before


@pytest.mark.parametrize("operation", ["state", "atomic", "comparison", "lock"])
def test_lane_helpers_refuse_redirected_parent(record, operation):
    lane = "lane-a"
    parent = record.directory / "lanes" / lane
    parent.parent.mkdir()
    parent.symlink_to(record.path.parent, target_is_directory=True)
    before = authority_bytes(record)
    with pytest.raises(typer.BadParameter, match="private record missing"):
        if operation == "state":
            lanes.LaneState(lane=lane, status="awaiting_selection", iteration=1).save(
                record.root, record.run_id
            )
        elif operation == "atomic":
            lanes._atomic_write_json(parent / "state.json", {}, root=record.root)
        elif operation == "comparison":
            lanes._write_comparison(record.root, record.run_id, lane, 1, {})
        else:
            with lanes._lane_lock(record.root, record.run_id, lane):
                pytest.fail("must not lock through a redirected directory")
    assert authority_bytes(record) == before
    assert not (record.path.parent / "state.json").exists()
    assert not (record.path.parent / ".lane.lock").exists()


def test_sentinel_uses_explicit_workspace_without_an_active_record(record, monkeypatch):
    directory = record.directory / "events"
    directory.mkdir()
    (directory / ".reaped").symlink_to(record.path.parent, target_is_directory=True)
    before = authority_bytes(record)
    monkeypatch.chdir(record.root.parent)
    with (
        harness_record.session(),
        pytest.raises(typer.BadParameter, match="private record missing"),
    ):
        events._claim_reaper_key(directory, "planted", root=record.root)
    assert authority_bytes(record) == before
    assert not (record.path.parent / "planted").exists()


def test_lane_comparison_cannot_name_a_private_file(record):
    lane = lanes.LaneState(
        lane="lane-a",
        status="awaiting_selection",
        iteration=1,
        comparison_path=str(record.path),
    )
    before = authority_bytes(record)
    with pytest.raises(typer.BadParameter, match="private record missing"):
        select._load_comparison(lane)
    assert authority_bytes(record) == before


def test_winner_check_refuses_mutable_git_names(record):
    from pydantic_ai_gepa.cli import lane_repositories

    repositories = lane_repositories.initialize(record.root, record.run_id, record.root)
    with pytest.raises(typer.BadParameter, match="Invalid retained candidate SHA"):
        repositories.candidate("HEAD")

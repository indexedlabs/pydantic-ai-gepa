"""Public path redirection must never redirect harness authority or writes."""

import fcntl
import json
from pathlib import Path
import shutil

import pytest
import typer

from pydantic_ai_gepa.cli import harness_record, layout
from pydantic_ai_gepa.cli.reflector import run_lock
from tests.cli import test_harness_record as records
from tests.cli import test_harness_scoring as harness
from tests.cli.test_git_candidate_cli import _run

git_repo = records.git_repo
isolated_environment = records.isolated_environment
prepared = records.prepared


def test_redirect_during_publication_does_not_redirect_temporary_cleanup(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    directory = workspace / ".gepa/runs/run/results"
    directory.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    original = harness_record.os.replace

    def redirect(source, destination, **kwargs):
        directory.rename(directory.with_name("original-results"))
        directory.symlink_to(victim, target_is_directory=True)
        (victim / source).write_text("keep")
        return original(source, destination, **kwargs)

    monkeypatch.setattr(harness_record.os, "replace", redirect)
    harness_record._atomic_text(directory / "result.json", "content", root=workspace)
    assert [file.read_text() for file in victim.iterdir()] == ["keep"]
    assert (
        directory.with_name("original-results") / "result.json"
    ).read_text() == "content"


def _snapshot(path):
    return {
        str(file.relative_to(path)): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file() and not file.name.endswith(".lock")
    }


@pytest.mark.parametrize("component", ["results", "run"])
@pytest.mark.parametrize("target", ["victim", "private"])
def test_redirected_view_directories_refuse_without_touching_targets(
    prepared, monkeypatch, tmp_path, component, target
):
    dataset, directory, _, _ = prepared
    private = dataset.parent / ".gepa-heldout"
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.json").write_text('{"keep": true}')
    destination = private if target == "private" else victim
    link = directory / "results" if component == "results" else directory
    if link.exists():
        shutil.rmtree(link)
    link.symlink_to(destination, target_is_directory=True)
    expected_private = _snapshot(private)
    expected_victim = _snapshot(victim)
    monkeypatch.setattr(
        records.run_module, "run_eval_once", lambda **_: pytest.fail("must not score")
    )
    monkeypatch.setattr(
        "pydantic_ai_gepa.cli.safe_git.SafeGit.bind",
        lambda *_, **__: pytest.fail("must refuse before updating the Git cache"),
    )
    served = harness._serve(directory.name, dataset, monkeypatch)
    assert served.exit_code == 2, served.output
    assert "private record missing or unreadable" in served.output
    assert _snapshot(private) == expected_private
    assert _snapshot(victim) == expected_victim
    assert any(name.endswith(".record.json") for name in expected_private)
    assert any(
        name.endswith(".json") and "." not in name[:-5] for name in expected_private
    )  # The original pin survives, too.


def test_gepa_swap_cannot_downgrade_harness_commands(prepared, monkeypatch):
    dataset, directory, _, _ = prepared
    run_id = directory.name
    workspace = directory.parents[2]
    public = workspace / ".gepa"
    moved = workspace / ".gepa-real"
    public.rename(moved)
    public.symlink_to(moved, target_is_directory=True)
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text())
    state.update(
        best_candidate_id="forged", best_mean_score=999, heldout_required=False
    )
    state_path.write_text(json.dumps(state))
    before = _snapshot(moved)
    private_before = _snapshot(dataset.parent)
    monkeypatch.setattr(
        records.run_module, "run_eval_once", lambda **_: pytest.fail("must not score")
    )
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(dataset))
        env.setenv("GEPA_DIR", str(public))
        for command in [
            ("run", "status"),
            ("harness", "serve", "--once"),
            ("run", "select"),
        ]:
            result = _run(*command, "--run-id", run_id)
            assert result.exit_code == 2, result.output
            assert "private record missing or unreadable" in result.output, (
                result.output
            )
            assert "forged" not in result.output
    assert _snapshot(moved) == before
    assert _snapshot(dataset.parent) == private_before


@pytest.mark.parametrize("damage", ["missing", "different_gepa_dir"])
def test_private_index_required_even_if_public_flag_is_false(
    prepared, monkeypatch, damage
):
    dataset, directory, _, _ = prepared
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text())
    state.update(heldout_required=False, best_candidate_id="forged")
    state_path.write_text(json.dumps(state))
    if damage == "missing":
        next((dataset.parent / ".gepa-heldout").glob("*.index.json")).unlink()
    else:
        public = directory.parents[1]
        moved = public.with_name(".gepa-real")
        public.rename(moved)
        monkeypatch.setenv("GEPA_DIR", str(moved))
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(dataset))
        result = _run("run", "status", "--run-id", directory.name)
    assert result.exit_code == 2, result.output
    assert "private record missing or unreadable" in result.output
    assert "forged" not in result.output


def test_leaf_and_parent_links_refuse_direct_record_access(prepared, monkeypatch):
    dataset, directory, _, _ = prepared
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(dataset))
        with harness_record.session():
            record = harness_record.for_run(directory.name)
            assert record is not None
            before = record.path.read_bytes()
            state = directory / "state.json"
            state.unlink()
            state.symlink_to(record.path)
            for action in [
                lambda: record.read("state.json"),
                lambda: record.write("state.json", "forged"),
            ]:
                with pytest.raises(typer.BadParameter, match="private record missing"):
                    action()
            assert record.path.read_bytes() == before
            assert state.is_symlink()
            state.unlink()
            results = directory / "results"
            results.symlink_to(record.path.parent, target_is_directory=True)
            with pytest.raises(typer.BadParameter, match="private record missing"):
                record.write("results/planted.json", "forged")
            assert record.path.read_bytes() == before
            assert not (record.path.parent / "planted.json").exists()


def test_config_pin_accepts_checkout_alias(prepared, monkeypatch, tmp_path):
    dataset, directory, _, _ = prepared
    workspace = directory.parents[2]
    config = layout.config_path(workspace)
    expected = layout.GepaConfig.load(config).stall_threshold
    config.write_text("stall_threshold = 999\n")
    alias = tmp_path.parent / f"alias-{tmp_path.name}"
    alias.symlink_to(workspace, target_is_directory=True)
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(dataset))
        with harness_record.session():
            harness_record.for_run(directory.name, workspace)
            with run_lock(directory.name, alias, wait=True, timeout=0.2):
                assert (
                    layout.GepaConfig.load(alias / ".gepa/gepa.toml").stall_threshold
                    == expected
                )
    assert config.read_text() == "stall_threshold = 999\n"


def test_reflector_resume_requires_harness_access(prepared, monkeypatch):
    dataset, directory, _, _ = prepared
    before = _snapshot(directory)
    refused = _run("run", "resume", "--run-id", directory.name)
    assert refused.exit_code == 2, refused.output
    assert "GEPA_HELDOUT_DATASET" in refused.output
    assert _snapshot(directory) == before
    with monkeypatch.context() as env:
        env.setenv("GEPA_HELDOUT_DATASET", str(dataset))
        resumed = _run("run", "resume", "--run-id", directory.name)
    assert resumed.exit_code == 0, resumed.output
    assert json.loads((directory / "state.json").read_text())["reflector"]["epoch"] == 2


def test_harness_also_coordinates_with_public_lock_and_releases_on_error(
    prepared, monkeypatch
):
    dataset, directory, _, _ = prepared
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(dataset))
    with harness_record.session():
        record = harness_record.for_run(directory.name)
        assert record is not None
        public = directory / "run.lock"
        with public.open("a") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(TimeoutError):
                with run_lock(directory.name, wait=True, timeout=0.02):
                    pytest.fail("must respect the reflector lock")
        with pytest.raises(RuntimeError, match="interrupted"):
            with run_lock(directory.name, wait=True, timeout=0.2):
                with run_lock(directory.name):  # Reentrancy must not deadlock.
                    with public.open("a") as contender:
                        with pytest.raises(BlockingIOError):
                            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    raise RuntimeError("interrupted")
        for path in (public, record.path.with_suffix(".lock")):
            with Path(path).open("a") as contender:
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not harness_record._locks.get()

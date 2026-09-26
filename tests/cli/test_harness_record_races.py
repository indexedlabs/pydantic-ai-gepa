"""A racing reflector cannot redirect descriptor-relative view operations."""

from contextlib import contextmanager
import json
import multiprocessing
import errno
import os
from pathlib import Path
import time

import pytest
import typer

from pydantic_ai_gepa.cli import harness_record, layout
from pydantic_ai_gepa.cli.reflector import run_lock
from pydantic_ai_gepa.cli.validation import pin_heldout, validation_dataset_path
from tests.cli.test_git_candidate_cli import _git
from tests.cli import test_harness_record as records

git_repo = records.git_repo
isolated_environment = records.isolated_environment


@pytest.fixture
def record(git_repo, monkeypatch):
    private = git_repo.parent / (git_repo.name + "-private")
    private.mkdir()
    dataset = private / "heldout.jsonl"
    dataset.write_text("PRIVATE DATA NOT IN GIT\n")
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(dataset))
    with harness_record.session():
        pin_heldout(git_repo, "race")
        harness_record.initialize(git_repo, "race")
        record = harness_record.for_run("race", git_repo)
        assert record is not None
        yield record


def _flip(path, victim, directory, stop, ready, flips):
    """Only manipulate public names; never traverse the victim link."""
    parked = path.with_name(path.name + "-parked")
    temporary = path.with_name(path.name + "-link")
    if directory:
        parked.mkdir()
    sequence = 0
    while not stop.is_set():
        sequence += 1
        saved = parked / str(sequence)
        try:
            if directory:
                try:
                    path.rename(saved)
                except FileNotFoundError:
                    pass
                path.symlink_to(victim, target_is_directory=True)
            else:
                temporary.symlink_to(victim)
                os.replace(temporary, path)
            with flips.get_lock():
                flips.value += 1
            ready.set()
            path.unlink()
        except (
            FileExistsError,
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
        ):
            pass
        finally:
            if directory:
                if path.is_symlink():
                    path.unlink(missing_ok=True)
                if saved.exists() and not path.exists():
                    try:
                        saved.rename(path)
                    except OSError as exc:
                        if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                            raise
            temporary.unlink(missing_ok=True)


@contextmanager
def flipping(path, victim, *, directory=False):
    ctx = multiprocessing.get_context("spawn")
    stop, ready, flips = ctx.Event(), ctx.Event(), ctx.Value("i", 0)
    worker = ctx.Process(
        target=_flip, args=(path, victim, directory, stop, ready, flips)
    )
    worker.start()
    try:
        assert ready.wait(10), "flipper did not start"
        yield flips
    finally:
        stop.set()
        worker.join(5)
        if worker.is_alive():
            worker.kill()
            worker.join(5)
        assert not worker.is_alive()
        assert worker.exitcode == 0


def race(action):
    deadline = time.monotonic() + 8
    attempts = successes = 0
    while attempts < 2000 and time.monotonic() < deadline:
        attempts += 1
        try:
            action()
            successes += 1
        except (typer.BadParameter, FileNotFoundError):
            pass  # Refusing a moving or absent view is safe.
    assert attempts >= 100
    return successes


@pytest.mark.parametrize("target", ["outside", "record"])
def test_public_lock_race_never_changes_victim(record, target, tmp_path):
    victim = record.path if target == "record" else tmp_path / "outside"
    if target == "outside":
        victim.write_text("PRECIOUS")
    expected = victim.read_bytes()

    def lock():
        with run_lock(record.run_id, record.root, wait=True, timeout=0.2):
            pass

    with flipping(record.directory / "run.lock", victim) as flips:
        race(lock)
        assert flips.value > 10
    assert victim.read_bytes() == expected
    assert json.loads(record.path.read_text())["run_id"] == record.run_id
    # A cooperative existing lock's bytes are never read, truncated, or written.
    public = record.directory / "run.lock"
    public.unlink(missing_ok=True)
    public.write_text("reflector PID")
    lock()
    assert public.read_text() == "reflector PID"


@pytest.mark.parametrize("operation", ["restore", "list", "publish"])
@pytest.mark.parametrize("target", ["outside", "record"])
def test_results_directory_race_preserves_targets(record, tmp_path, operation, target):
    results = record.directory / "results"
    results.mkdir()
    victim = record.path.parent if target == "record" else tmp_path / "outside"
    victim.mkdir(exist_ok=True)
    keep = victim / "keep.json"
    keep.write_text("PRIVATE SENTINEL")
    expected = {
        p.name: p.read_bytes()
        for p in victim.iterdir()
        if p.is_file() and not p.name.endswith(".lock")
    }
    private_before = record.path.read_bytes()

    def action():
        if operation == "restore":
            with harness_record.SafeDir.open(
                record.root, results, create=True
            ) as opened:
                opened.write_text("keep.json", "planted")
            record._restore("results/keep.json", None)
        elif operation == "list":
            record.restore_views()
        else:
            harness_record._atomic_text(
                results / "keep.json", "public", root=record.root
            )

    with flipping(results, victim, directory=True) as flips:
        race(action)
        assert flips.value > 10
    assert {
        p.name: p.read_bytes()
        for p in victim.iterdir()
        if p.is_file() and not p.name.endswith(".lock")
    } == expected
    assert record.path.read_bytes() == private_before
    assert not any(
        "PRIVATE SENTINEL" in p.read_text()
        for p in record.directory.rglob("*.json")
        if not p.is_symlink()
    )
    assert not list(victim.glob("*.tmp"))


def test_fresh_eval_lookup_only_allows_absent_run(record):
    assert harness_record.for_run("fresh", record.root) is None
    layout.run_dir("fresh", record.root).mkdir()
    with pytest.raises(typer.BadParameter, match="private record missing"):
        harness_record.for_run("fresh", record.root)


def test_private_bookkeeping_is_not_subject_to_git_history(record, monkeypatch):
    index = next(record.path.parent.glob("*.index.json"))
    pin = next(p for p in record.path.parent.glob("*.json") if "." not in p.stem)
    for path in (index, pin, record.path):
        (record.root / path.name).write_bytes(path.read_bytes())
    _git(record.root, "add", ".")
    _git(record.root, "commit", "-m", "Predictable private bookkeeping blobs")
    with harness_record.session():
        loaded = harness_record.for_run(record.run_id, record.root)
        assert loaded is not None
        assert loaded.load() == record.load()
    # Dataset isolation still includes the Git object database.
    dataset = Path(record.dataset)
    (record.root / "dataset-copy").write_bytes(dataset.read_bytes())
    _git(record.root, "add", "dataset-copy")
    _git(record.root, "commit", "-m", "Held-out blob must still be refused")
    with pytest.raises(typer.BadParameter, match="recoverable from Git objects"):
        validation_dataset_path(str(dataset), project_root=record.root)


def test_external_gepa_dir_trusts_system_alias_ancestors(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    parent = tmp_path / "system-parent"
    parent.mkdir()
    alias = tmp_path / "system-alias"
    alias.symlink_to(parent, target_is_directory=True)
    base = alias / "gepa"
    monkeypatch.setenv("GEPA_DIR", str(base))
    with harness_record.SafeDir.open(
        workspace, base / "runs/run", create=True
    ) as opened:
        opened.write_text("state.json", "safe")
    assert (parent / "gepa/runs/run/state.json").read_text() == "safe"
    (parent / "gepa").rename(parent / "moved")
    (parent / "gepa").symlink_to(parent / "moved", target_is_directory=True)
    with pytest.raises(typer.BadParameter, match="private record missing"):
        with harness_record.SafeDir.open(workspace, base / "runs/run"):
            pytest.fail("GEPA_DIR itself must never follow a symlink")


@pytest.mark.parametrize(
    "operation", ["read", "append", "publish", "events", "nominations", "notes"]
)
def test_auxiliary_public_io_cannot_redirect_into_private_storage(record, operation):
    from pydantic_ai_gepa.cli.events import emit, EventDraft
    from pydantic_ai_gepa.cli.harness import _requests
    from pydantic_ai_gepa.cli.notes import notes_index

    directory = record.directory / operation
    directory.symlink_to(record.path.parent, target_is_directory=True)
    path = directory / record.path.name
    before = record.path.read_bytes()
    with pytest.raises(typer.BadParameter, match="private record missing"):
        if operation == "nominations":
            _requests(record.run_id)
        elif operation == "events":
            emit(
                record.run_id,
                "harness",
                EventDraft(
                    type="run_done", lane=None, payload={"final_report_path": None}
                ),
                root=record.root,
            )
        elif operation == "notes":
            notes_index(directory)
        elif operation == "read":
            harness_record.read_text(path, root=record.root)
        else:
            harness_record.write_text(
                path, "public", root=record.root, append=operation == "append"
            )
    assert record.path.read_bytes() == before


def test_opened_leaf_is_not_reread_by_path_after_redirection(record, monkeypatch):
    path = record.directory / "reflector_packet.json"
    path.write_text("PUBLIC")
    original = harness_record.os.open

    def flip_after_open(name, flags, *args, **kwargs):
        fd = original(name, flags, *args, **kwargs)
        if name == path.name and kwargs.get("dir_fd") is not None:
            path.unlink()
            path.symlink_to(record.path)
        return fd

    monkeypatch.setattr(harness_record.os, "open", flip_after_open)
    assert harness_record.read_text(path, root=record.root) == "PUBLIC"

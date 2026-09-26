"""Real Git evidence for repository ownership and object transfer."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

import pytest
import typer

from pydantic_ai_gepa.cli import lane_repositories as repos
from pydantic_ai_gepa.cli import layout


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "export"
    root.mkdir()
    (root / ".gitignore").write_text("__pycache__/\n")
    (root / "api").mkdir()
    (root / "api/pyproject.toml").write_text(
        '[project]\nname="example"\nversion="0.0.0"\n'
    )
    (root / "api/prompt.txt").write_text("training seed\n")
    git(root, "init", "--quiet")
    git(root, "config", "user.email", "tests@example.com")
    git(root, "config", "user.name", "Tests")
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "Seed")
    monkeypatch.setattr(layout, "_explicit_gepa_dirname", str(tmp_path / "public"))
    monkeypatch.setattr(
        "pydantic_ai_gepa.cli.validation.heldout_dataset", lambda **kwargs: None
    )
    return root


def digest_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_retain_refan_and_reload_without_shared_objects(seed: Path) -> None:
    record = repos.initialize(seed / "api", "run1", seed / "api")
    first = record.create_lane("lane-1", record.seed, "gepa/lane-1/1")
    second = record.create_lane("lane-2", record.seed, "gepa/lane-2/1")
    before = {
        root: digest_tree(root / ".git") for root in (seed, second, record.repository)
    }
    for lane in (first, second):
        assert (lane / ".git").is_dir()
        assert not (lane / ".git/objects/info/alternates").exists()
        assert not git(lane, "remote")
        assert git(lane, "rev-list", "--all", "--count") == "1"
    # Every regular object has a distinct inode, including seed pack files.
    inodes = [
        {p.stat().st_ino for p in (root / ".git/objects").rglob("*") if p.is_file()}
        for root in (seed, first, second, record.repository)
    ]
    assert all(
        not a.intersection(b) for i, a in enumerate(inodes) for b in inodes[i + 1 :]
    )
    (first / "api/prompt.txt").write_text("candidate\n")
    git(first, "add", ".")
    git(first, "commit", "--quiet", "-m", "Candidate")
    sha = git(first, "rev-parse", "HEAD")
    for root, digest in before.items():
        assert digest_tree(root / ".git") == digest
    record.import_lane("lane-1", sha)
    record.remove_lane("lane-1")
    record = repos.load(seed / "api", "run1")
    assert (record.candidate(sha) / "prompt.txt").read_text() == "candidate\n"
    record.promote(sha)
    next_lane = record.create_lane("lane-1", sha, "gepa/lane-1/2")
    assert git(next_lane, "rev-parse", "HEAD") == sha
    assert git(seed, "rev-parse", "HEAD") == record.seed
    assert git(second, "rev-parse", "HEAD") == record.seed
    # Enumerate every retained object, including packed/dangling objects.
    objects = git(
        next_lane, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)"
    ).splitlines()
    assert objects
    for oid in objects:
        subprocess.run(
            [
                "git",
                "-C",
                str(next_lane),
                "show",
                "--no-ext-diff",
                "--no-textconv",
                oid,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def test_import_never_executes_source_configuration(seed: Path, tmp_path: Path) -> None:
    record = repos.initialize(seed / "api", "run1", seed / "api")
    lane = record.create_lane("lane-1", record.seed, "gepa/lane-1")
    marker = tmp_path / "executed"
    hook = tmp_path / "evil-hook"
    hook.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    hook.chmod(0o755)
    for key in (
        "core.fsmonitor",
        "core.hooksPath",
        "core.sshCommand",
        "uploadpack.packObjectsHook",
        "diff.external",
    ):
        git(lane, "config", key, str(hook))
    (lane / ".git/hooks").mkdir(exist_ok=True)
    (lane / ".git/hooks/reference-transaction").write_bytes(hook.read_bytes())
    (lane / ".git/hooks/reference-transaction").chmod(0o755)
    record.import_lane("lane-1", record.seed)
    assert not marker.exists()


@pytest.mark.parametrize(
    "attack",
    [
        "objects",
        "alternates",
        "gitdir",
        "commondir",
        "lane",
        "loose-object",
        "hardlink-object",
    ],
)
def test_import_refuses_redirected_metadata(
    seed: Path, tmp_path: Path, attack: str
) -> None:
    record = repos.initialize(seed / "api", "run1", seed / "api")
    lane = record.create_lane("lane-1", record.seed, "gepa/lane-1")
    if attack == "objects":
        (lane / ".git/objects").rename(lane / ".git/old-objects")
        (lane / ".git/objects").symlink_to(seed / ".git/objects")
    elif attack == "alternates":
        (lane / ".git/objects/info/alternates").write_text(
            str(seed / ".git/objects") + "\n"
        )
    elif attack == "gitdir":
        (lane / ".git").rename(lane / "old-git")
        (lane / ".git").write_text(f"gitdir: {seed / '.git'}\n")
    elif attack == "commondir":
        (lane / ".git/commondir").symlink_to(seed / "api/prompt.txt")
    elif attack == "lane":
        lane.rename(lane.with_name("old-lane"))
        lane.symlink_to(seed)
    else:
        (lane / ".git/objects/aa").mkdir()
        target = lane / (".git/objects/aa/" + "a" * 38)
        if attack == "hardlink-object":
            os.link(seed / "api/prompt.txt", target)
        else:
            target.symlink_to(seed / "api/prompt.txt")
    before = digest_tree(seed / ".git")
    with pytest.raises((OSError, typer.BadParameter)):
        record.import_lane("lane-1", record.seed)
    assert digest_tree(seed / ".git") == before


def test_missing_layout_and_linked_runs_refuse(seed: Path) -> None:
    with pytest.raises(typer.BadParameter, match="start a new run"):
        repos.load(seed / "api", "legacy")


@pytest.mark.parametrize("operation", ["promote", "refan"])
def test_repository_replacement_recovers_after_interruption(
    seed: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    record = repos.initialize(seed / "api", "run1", seed / "api")
    lane = record.create_lane("lane-1", record.seed, "gepa/lane-1/1")
    (lane / "api/prompt.txt").write_text("next iteration\n")
    git(lane, "add", ".")
    git(lane, "commit", "--quiet", "-m", "Candidate")
    sha = git(lane, "rev-parse", "HEAD")
    record.import_lane("lane-1", sha)
    destination = record.repository if operation == "promote" else lane
    rename = Path.rename

    def interrupt(path: Path, target: Path) -> Path:
        if target == destination:
            raise OSError("Simulated interruption between renames")
        return rename(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", interrupt)
        with pytest.raises(OSError, match="Simulated interruption"):
            if operation == "promote":
                record.promote(sha)
            else:
                record.create_lane("lane-1", sha, "gepa/lane-1/2", replace=True)
    record = repos.load(seed / "api", "run1")
    if operation == "promote":
        record.promote(sha)
    else:
        record.create_lane("lane-1", sha, "gepa/lane-1/2", replace=True)
    assert git(destination, "rev-parse", "HEAD") == sha
    assert (destination / "api/prompt.txt").read_text() == "next iteration\n"
    assert git(seed, "rev-parse", "HEAD") == record.seed


def test_heldout_store_is_outside_public_and_lanes(
    seed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "private"
    private.mkdir()
    dataset = private / "validation.jsonl"
    dataset.write_text("")
    monkeypatch.setattr(
        "pydantic_ai_gepa.cli.validation.heldout_dataset", lambda **kwargs: str(dataset)
    )
    record = repos.initialize(seed / "api", "run1", seed / "api")
    assert record.directory.is_relative_to(private)
    assert not record.directory.is_relative_to(layout.gepa_dir(seed))
    lane = record.create_lane("lane-1", record.seed, "gepa/lane-1")
    assert not record.directory.is_relative_to(lane)
    assert not lane.is_relative_to(layout.gepa_dir(seed))


def test_candidate_root_and_training_need_no_scorer_reads(
    seed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from typer.testing import CliRunner
    from pydantic_ai_gepa.cli import app
    from pydantic_ai_gepa.cli.lanes import load_lane_state

    module = seed / "api/lane_example"
    module.mkdir()
    (module / "__init__.py").touch()
    (module / "evaluate.py").write_text(
        'async def evaluate(case):\n    return "answer"\n'
    )
    git(seed, "add", ".")
    git(seed, "commit", "--quiet", "-m", "Evaluator")
    scorer = tmp_path / "scorer"
    scorer.mkdir()
    import shutil

    shutil.copytree(module, scorer / "lane_example")
    (scorer / "pyproject.toml").write_text(
        '[project]\nname="scorer"\nversion="0.0.0"\n'
    )
    git(scorer, "init", "--quiet")
    git(scorer, "config", "user.email", "tests@example.com")
    git(scorer, "config", "user.name", "Tests")
    git(scorer, "add", ".")
    git(scorer, "commit", "--quiet", "-m", "Scorer")
    scorer_head = git(scorer, "rev-parse", "HEAD")
    monkeypatch.chdir(scorer)
    runner = CliRunner()
    public = tmp_path / "public"
    args = ["-G", str(public)]
    result = runner.invoke(
        app,
        [
            *args,
            "init",
            "--candidate-source",
            "git",
            "--evaluate",
            "lane_example.evaluate:evaluate",
        ],
    )
    assert result.exit_code == 0, result.output
    (public / "dataset.jsonl").write_text(
        json.dumps({"name": "training", "inputs": "x", "expected_output": "different"})
        + "\n"
    )
    result = runner.invoke(
        app,
        [
            *args,
            "run",
            "start",
            "--candidate-root",
            str(seed / "api"),
            "--lanes",
            "1",
            "--size",
            "1",
            "--max-iterations",
            "10",
            "--acceptance-repetitions",
            "1",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    run_id = layout.latest_run_id(scorer)
    assert run_id is not None
    lane = load_lane_state(scorer, run_id, "lane-1")
    lane_project = Path(str(lane.candidate_project_path))
    assert git(Path(str(lane.worktree_path)), "rev-parse", "HEAD") == git(
        seed, "rev-parse", "HEAD"
    )
    assert git(scorer, "rev-parse", "HEAD") == scorer_head
    monkeypatch.chdir(lane_project)
    original_open = Path.open
    original_stat = Path.stat

    def refuse_open(path, *args, **kwargs):
        if path.is_relative_to(scorer):
            raise AssertionError("Reflector attempted a scorer read")
        return original_open(path, *args, **kwargs)

    def refuse_stat(path, *args, **kwargs):
        if path.is_relative_to(scorer):
            raise AssertionError("Reflector attempted a scorer stat")
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", refuse_open)
        patch.setattr(Path, "stat", refuse_stat)
        result = runner.invoke(
            app,
            [*args, "lane", "continue", "lane-1", "--run-id", run_id, "--foreground"],
        )
    if result.exception is not None:
        raise result.exception
    assert result.exit_code == 0, (result.output, result.exception)
    assert load_lane_state(scorer, run_id, "lane-1").status == "awaiting_selection"


def test_candidate_checkout_failure_is_never_published(seed: Path, monkeypatch) -> None:
    record = repos.initialize(seed / "api", "run1", seed / "api")
    original = repos.checkout

    def interrupted(source, sha, destination, branch):
        destination.mkdir()
        (destination / "partial").touch()
        raise OSError("interrupted checkout")

    monkeypatch.setattr(repos, "checkout", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        record.candidate(record.seed)
    assert not (record.directory / ("candidate-" + record.seed)).exists()
    monkeypatch.setattr(repos, "checkout", original)
    assert (
        record.candidate(record.seed) / "prompt.txt"
    ).read_text() == "training seed\n"


def test_ignore_entries_are_not_duplicated(seed: Path, monkeypatch) -> None:
    from pydantic_ai_gepa.cli.lanes import ensure_worktrees_ignored

    monkeypatch.setattr(layout, "_explicit_gepa_dirname", str(seed / "api/.gepa"))
    ensure_worktrees_ignored(seed / "api")
    before = (seed / ".git/info/exclude").read_bytes()
    ensure_worktrees_ignored(seed / "api")
    assert (seed / ".git/info/exclude").read_bytes() == before


def test_remove_lane_ignores_malformed_git_metadata(seed: Path) -> None:
    record = repos.initialize(seed / "api", "run1", seed / "api")
    lane = repos.lane_path(seed / "api", "run1", "lane-1")
    lane.mkdir(parents=True)
    (lane / ".git").symlink_to(seed / ".git", target_is_directory=True)
    record.remove_lane("lane-1")
    assert not lane.exists()
    assert (seed / ".git").is_dir()

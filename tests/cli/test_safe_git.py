"""Executable Git configuration must never cross the held-out boundary."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest
import typer

from pydantic_ai_gepa.cli.candidates import git_candidate_state
from pydantic_ai_gepa.cli.layout import candidate_identity_exempt_paths, git_root
from pydantic_ai_gepa.cli.runs import current_commit_sha
from pydantic_ai_gepa.cli.safe_git import Repository, run_git, safe_repository
from pydantic_ai_gepa.cli.scoring_sandbox import (
    ScoringSandboxError,
    _checkout_entries,
    candidate_components,
    private_checkout,
)
from pydantic_ai_gepa.cli.validation import validation_dataset_path
from tests.cli.test_git_candidate_cli import _git, _run, _run_payload
from tests.cli import test_git_candidate_cli, test_scoring_sandbox
from tests.cli.test_scoring_sandbox import commit_evaluator

git_repo = test_git_candidate_cli.git_repo
private = test_scoring_sandbox.private
protocol_backend = test_scoring_sandbox.protocol_backend
clean_environment = test_scoring_sandbox.clean_environment


def poison(repo: Path, sentinel: Path, *, worktree: bool = False) -> None:
    """Install every executable class without invoking Git after installation."""
    metadata = Repository.discover(repo)
    script = sentinel.with_suffix(".sh")
    script.write_text(
        f"#!/bin/sh\nprintf '%s\\n' executed >> {shlex.quote(str(sentinel))}\nexit 1\n"
    )
    script.chmod(0o755)
    command = str(script)
    hooks = sentinel.with_suffix(".hooks")
    hooks.mkdir(exist_ok=True)
    for hook in (
        "applypatch-msg",
        "pre-applypatch",
        "post-applypatch",
        "pre-commit",
        "pre-merge-commit",
        "prepare-commit-msg",
        "commit-msg",
        "post-commit",
        "pre-rebase",
        "post-checkout",
        "post-merge",
        "pre-push",
        "pre-receive",
        "update",
        "proc-receive",
        "post-receive",
        "post-update",
        "reference-transaction",
        "push-to-checkout",
        "pre-auto-gc",
        "post-rewrite",
        "sendemail-validate",
        "fsmonitor-watchman",
        "p4-changelist",
        "p4-prepare-changelist",
        "p4-post-changelist",
        "p4-pre-submit",
        "post-index-change",
    ):
        (hooks / hook).write_bytes(script.read_bytes())
        (hooks / hook).chmod(0o755)
    included = sentinel.with_suffix(".config")
    included.write_text(
        f'[filter "x"]\n clean = {command}\n smudge = {command}\n process = {command}\n'
        f'[diff "x"]\n textconv = {command}\n'
        f"[diff]\n external = {command}\n"
        f"[core]\n fsmonitor = {command}\n hooksPath = {hooks}\n pager = {command}\n sshCommand = {command}\n attributesFile = {sentinel.with_suffix('.attributes')}\n"
        f"[log]\n showSignature = true\n[gpg]\n program = {command}\n"
        "[extensions]\n partialClone = hostile\n"
        f'[remote "hostile"]\n promisor = true\n url = ext::{command}\n'
        "[protocol]\n allow = always\n"
    )
    sentinel.with_suffix(".attributes").write_text("* filter=x diff=x\n")
    with (metadata.common_dir / "config").open("a") as file:
        file.write(
            f'\n[include]\n path = {included}\n[includeIf "gitdir:**"]\n path = {included}\n'
        )
        if worktree:
            file.write("[extensions]\n worktreeConfig = true\n")
    if worktree:
        (metadata.git_dir / "config.worktree").write_bytes(included.read_bytes())
    (metadata.common_dir / "info").mkdir(exist_ok=True)
    (metadata.common_dir / "info/attributes").write_text("* filter=x diff=x\n")


def attributes(repo):
    (repo / ".gitattributes").write_text("* filter=x diff=x\n")
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-m", "Attribute names with no executable drivers")


@pytest.mark.parametrize("missing_blob", [False, True])
def test_trusted_scorer_reads_ignore_hostile_git_config(
    git_repo, private, missing_blob
):
    scorer = _git(git_repo, "rev-parse", "HEAD")
    (git_repo / "score.txt").write_text("trusted component input")
    _git(git_repo, "add", "score.txt")
    _git(git_repo, "commit", "-m", "Candidate text")
    candidate = _git(git_repo, "rev-parse", "HEAD")
    blob = _git(git_repo, "rev-parse", "HEAD:score.txt")
    if missing_blob:
        (git_repo / ".git/objects" / blob[:2] / blob[2:]).unlink()
    sentinel = private.parent / "git-config-executed"
    poison(git_repo, sentinel)
    if missing_blob:
        with pytest.raises(ScoringSandboxError, match="blob response"):
            candidate_components(git_repo, candidate, ("score.txt",))
    else:
        assert candidate_components(git_repo, candidate, ("score.txt",)) == {
            "score.txt": "trusted component input"
        }
    with private_checkout(git_repo, scorer) as (_, checkout, _):
        assert (checkout / "score.txt").read_text() == "bad\n"
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "attack",
    [
        "clean",
        "process",
        "textconv",
        "fsmonitor",
        "lazy-fetch",
        "external",
        "smudge",
        "hook",
    ],
)
def test_plain_git_control_executes_local_commands(git_repo, tmp_path, attack):
    attributes(git_repo)
    sentinel = tmp_path.parent / (tmp_path.name + "-control")
    script = sentinel.with_suffix(".sh")
    script.write_text(
        f"#!/bin/sh\necho {attack} >> {shlex.quote(str(sentinel))}\nexit 1\n"
    )
    script.chmod(0o755)
    command = str(script)
    (git_repo / "score.txt").write_text("changed\n")
    args = ["diff", "HEAD"]
    if attack in {"clean", "process", "smudge"}:
        _git(git_repo, "config", f"filter.x.{attack}", command)
        if attack == "smudge":
            args = ["checkout", "--", "score.txt"]
    elif attack == "textconv":
        _git(git_repo, "config", "diff.x.textconv", command)
    elif attack == "external":
        _git(git_repo, "config", "diff.external", command)
    elif attack == "fsmonitor":
        _git(git_repo, "config", "core.fsmonitor", command)
        args = ["status", "--porcelain"]
    elif attack == "hook":
        hooks = git_repo / ".git/hooks"
        (hooks / "post-checkout").write_bytes(script.read_bytes())
        (hooks / "post-checkout").chmod(0o755)
        args = ["checkout", "--", "score.txt"]
    else:
        _git(git_repo, "config", "extensions.partialClone", "hostile")
        _git(git_repo, "config", "remote.hostile.promisor", "true")
        _git(git_repo, "config", "remote.hostile.url", f"ext::{command}")
        _git(git_repo, "config", "protocol.allow", "always")
        args = ["cat-file", "-p", "a" * 40]
    subprocess.run(["git", "-C", str(git_repo), *args], capture_output=True, timeout=10)
    assert set(sentinel.read_text().splitlines()) == {attack}


@pytest.mark.parametrize("linked", [False, True])
def test_hostile_repository_reads_never_execute(git_repo, private, tmp_path, linked):
    attributes(git_repo)
    repository = git_repo
    if linked:
        repository = tmp_path.parent / (tmp_path.name + "-linked")
        _git(git_repo, "worktree", "add", "-b", "linked", str(repository))
    clean = git_candidate_state(repository)
    (repository / "score.txt").write_text("changed\n")
    # Assert byte-for-byte compatibility with the old identity algorithm.
    diff = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "diff",
            "--binary",
            "--full-index",
            "HEAD",
            "--",
            ".",
        ],
        capture_output=True,
        check=True,
    ).stdout
    expected = (
        clean.short_commit_sha
        + "-dirty-"
        + hashlib.sha256(b"tracked-diff\0" + diff).hexdigest()[:12]
    )
    sentinel = tmp_path.parent / (tmp_path.name + "-hostile")
    poison(repository, sentinel, worktree=linked)
    state = git_candidate_state(repository)
    assert state.dirty and state.candidate_id == expected
    assert git_candidate_state(repository) == state
    assert git_root(repository) == repository
    assert current_commit_sha(repository) == clean.commit_sha[:10]
    assert validation_dataset_path(str(private), project_root=repository) == private
    with private_checkout(repository, clean.commit_sha) as (_, checkout, _):
        assert (checkout / "score.txt").read_text() == "bad\n"
        assert not (checkout / ".git").exists()
    assert (
        run_git(
            repository,
            "cat-file",
            "--batch-check",
            input=b"a" * 40 + b"\n",
            capture_output=True,
            check=True,
        ).stdout
        == b"a" * 40 + b" missing\n"
    )
    # Neither attributes, the dirty source, nor source index get changed.
    assert (repository / "score.txt").read_text() == "changed\n"
    assert not sentinel.exists()


def test_environment_is_replaced_and_index_is_private(git_repo, monkeypatch, tmp_path):
    sentinel = tmp_path.parent / (tmp_path.name + "-env")
    poison(git_repo, sentinel)
    for key in (
        "GIT_EXTERNAL_DIFF",
        "GIT_SSH_COMMAND",
        "GIT_CONFIG_PARAMETERS",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_EXEC_PATH",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_ATTR_SOURCE",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        monkeypatch.setenv(key, str(sentinel))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(sentinel.with_suffix(".sh")))
    index = (git_repo / ".git/index").read_bytes()
    assert not git_candidate_state(git_repo).dirty
    assert (git_repo / ".git/index").read_bytes() == index
    with safe_repository(git_repo) as git:
        assert git.env["GIT_NO_LAZY_FETCH"] == "1"
        assert not git.directory.stat().st_mode & 0o077
        assert not Path(git.env["HOME"]).stat().st_mode & 0o077
        assert git.run(
            "rev-parse", "--git-dir", capture_output=True
        ).stdout.decode().strip() == str(git.directory)
    assert not sentinel.exists()


def test_nested_repository_and_submodule_config_never_execute(git_repo, tmp_path):
    nested = git_repo / "nested"
    nested.mkdir()
    _git(nested, "init")
    (nested / "content").write_text("one")
    sentinel = tmp_path.parent / (tmp_path.name + "-nested")
    poison(nested, sentinel)
    first = git_candidate_state(git_repo)
    assert first.dirty
    (nested / "content").write_text("two")
    assert git_candidate_state(git_repo).candidate_id != first.candidate_id
    assert not sentinel.exists()
    # Also test a tracked gitlink: diff/status must not descend into it.
    sha = _git(git_repo, "rev-parse", "HEAD")
    _git(git_repo, "update-index", "--add", "--cacheinfo", f"160000,{sha},nested")
    _git(git_repo, "commit", "-m", "Gitlink")
    assert not git_candidate_state(git_repo).dirty
    run_git(git_repo, "status", "--porcelain", check=True, capture_output=True)
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "name",
    [
        b".git",
        b".GIT",
        b".git.",
        b".git ",
        b"sub/.Git/config",
        ".g\u200cit".encode(),
        b"git~1",
        b".git::$INDEX_ALLOCATION",
        b"sub\\.git\\config",
    ],
)
def test_checkout_refuses_git_equivalent_paths_before_writing(tmp_path, name):
    listing = (
        b"100644 blob "
        + b"a" * 40
        + b"\tgood\0"
        + b"100644 blob "
        + b"b" * 40
        + b"\t"
        + name
        + b"\0"
    )
    with pytest.raises(ScoringSandboxError, match="checkout path"):
        _checkout_entries(tmp_path, listing)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("redirect_refs", [False, True])
def test_harness_once_scores_with_hostile_git_config(
    git_repo, private, protocol_backend, monkeypatch, tmp_path, redirect_refs
):
    attributes(git_repo)
    commit_evaluator(
        git_repo,
        "from pathlib import Path\nasync def evaluate(case): return Path('score.txt').read_text().strip()\n",
    )
    started = _run(
        "run",
        "start",
        "--size",
        "1",
        "--max-iterations",
        "16",
        "--acceptance-repetitions",
        "1",
        "--acceptance-paired-min-cases",
        "2",
    )
    assert started.exit_code == 0, (started.output, started.exception)
    run_id = _run_payload(started.output)["run_id"]
    monkeypatch.delenv("GEPA_HELDOUT_DATASET")
    (git_repo / "score.txt").write_text("good")
    _git(git_repo, "add", "score.txt")
    _git(git_repo, "commit", "-m", "Improved candidate")
    before = git_candidate_state(
        git_repo, exclude_paths=candidate_identity_exempt_paths(git_repo)
    )
    sentinel = tmp_path.parent / (tmp_path.name + "-harness")
    poison(git_repo, sentinel)
    queued = _run("run", "continue", "--run-id", run_id, "--wait-secs", "0")
    assert queued.exit_code == 0, (queued.output, queued.exception)
    if redirect_refs:
        redirected = private.parent / "redirected-refs"
        redirected.mkdir()
        (redirected / "untouched").write_text("private data")
        refs = git_repo / ".git/refs/gepa"
        refs.rename(git_repo / ".git/original-gepa-refs")
        refs.symlink_to(redirected, target_is_directory=True)
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(private))
    served = _run("harness", "serve", "--run-id", run_id, "--once")
    assert served.exit_code == 0, (served.output, served.exception)
    monkeypatch.delenv("GEPA_HELDOUT_DATASET")
    delivered = _run("run", "continue", "--run-id", run_id, "--wait-secs", "0")
    if redirect_refs:
        assert delivered.exit_code == 2, delivered.output
        assert "Could not retain the validated candidate commit" in delivered.output
        assert str(private) not in delivered.output
        assert list(redirected.iterdir()) == [redirected / "untouched"]
        assert (redirected / "untouched").read_text() == "private data"
        assert not sentinel.exists()
        return
    assert delivered.exit_code == 0, (delivered.output, delivered.exception)
    assert _run_payload(delivered.output)["best_mean_score"] == 1
    assert _run_payload(delivered.output)["last_reflector_comparison"][
        "validation_improved"
    ]
    assert (
        git_candidate_state(
            git_repo, exclude_paths=candidate_identity_exempt_paths(git_repo)
        )
        == before
    )
    assert not sentinel.exists()


@pytest.mark.parametrize("object_format", ["sha1", "sha256"])
@pytest.mark.parametrize("head_kind", ["loose", "packed", "detached"])
def test_object_formats_and_head_layouts(tmp_path, object_format, head_kind):
    _git(tmp_path, "init", f"--object-format={object_format}")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.com")
    (tmp_path / "file").write_text("original\n")
    _git(tmp_path, "add", "file")
    _git(tmp_path, "commit", "-m", "Seed")
    oid = _git(tmp_path, "rev-parse", "HEAD")
    if head_kind == "packed":
        _git(tmp_path, "pack-refs", "--all", "--prune")
    elif head_kind == "detached":
        _git(tmp_path, "checkout", "--detach", "HEAD")
    assert git_candidate_state(tmp_path).commit_sha == oid
    assert not git_candidate_state(tmp_path).dirty
    (tmp_path / "file").write_text("changed\n")
    assert git_candidate_state(tmp_path).dirty


def test_missing_promised_blob_fails_without_fetch(git_repo, private, tmp_path):
    sha = _git(git_repo, "rev-parse", "HEAD")
    oid = _git(git_repo, "rev-parse", "HEAD:score.txt")
    (git_repo / ".git/objects" / oid[:2] / oid[2:]).unlink()
    sentinel = tmp_path.parent / (tmp_path.name + "-missing")
    poison(git_repo, sentinel)
    with pytest.raises(ScoringSandboxError, match="blob response"):
        with private_checkout(git_repo, sha):
            pytest.fail("Missing objects must fail closed")
    assert not sentinel.exists()
    assert not list((private.parent / ".gepa-heldout/work").iterdir())


def test_validation_rejects_historical_copy_under_hostile_config(
    git_repo, private, tmp_path
):
    # This writes only the fake test dataset into a local Git object database.
    subprocess.run(
        ["git", "-C", str(git_repo), "hash-object", "-w", "--stdin"],
        input=private.read_bytes(),
        check=True,
        capture_output=True,
    )
    sentinel = tmp_path.parent / (tmp_path.name + "-historical")
    poison(git_repo, sentinel)
    with pytest.raises(typer.BadParameter, match="recoverable from Git objects"):
        validation_dataset_path(str(private), project_root=git_repo)
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "args",
    [
        ("worktree", "add", "unused"),
        ("worktree", "remove", "unused"),
        ("worktree", "prune"),
        ("reset", "--hard", "HEAD"),
        ("checkout", "unused"),
        ("branch", "-f", "unused", "HEAD"),
        ("branch", "-D", "unused"),
        ("clean", "-fd"),
        ("add", "-A"),
        ("commit", "-m", "unused"),
    ],
)
def test_heldout_lane_mutations_refused_before_git(
    git_repo, private, monkeypatch, args
):
    from pydantic_ai_gepa.cli.lanes import _git as lane_git

    def unexpected(*args, **kwargs):
        pytest.fail("A refused mutation must never start Git")

    monkeypatch.setattr(subprocess, "run", unexpected)
    with pytest.raises(typer.BadParameter, match="single-checkout") as error:
        lane_git(git_repo, *args)
    assert str(private) not in str(error.value)
    assert str(git_repo) not in str(error.value)


def test_heldout_lane_start_uses_owned_repositories(
    git_repo, private, protocol_backend, monkeypatch
):
    from pydantic_ai_gepa.cli.lane_repositories import load, lane_path

    result = _run(
        "run",
        "start",
        "--lanes",
        "2",
        "--size",
        "1",
        "--max-iterations",
        "20",
        "--acceptance-repetitions",
        "1",
    )
    assert result.exit_code == 0, (result.output, result.exception)
    run_id = str(_run_payload(result.output)["run_id"])
    from pydantic_ai_gepa.cli.validation import harness_environment

    with harness_environment():
        record = load(git_repo, run_id)
        assert record.directory.is_relative_to(private.parent)
    source_sha = _git(git_repo, "rev-parse", "HEAD")
    candidate_sha = None
    for lane in ("lane-1", "lane-2"):
        path = lane_path(git_repo, run_id, lane)
        assert (path / ".git").is_dir()
        assert not (path / ".git/objects/info/alternates").exists()
        if lane == "lane-1":
            (path / "score.txt").write_text("good\n")
            _git(path, "add", "score.txt")
            _git(path, "commit", "-m", "Improve candidate")
            candidate_sha = _git(path, "rev-parse", "HEAD")
        with monkeypatch.context() as patch:
            patch.delenv("GEPA_HELDOUT_DATASET")
            patch.chdir(path)
            result = _run(
                "-G",
                str(git_repo / ".gepa"),
                "lane",
                "continue",
                lane,
                "--run-id",
                run_id,
                "--foreground",
            )
        assert result.exit_code == 0, (result.output, result.exception)
        poison(path, git_repo.parent / (git_repo.name + "-" + lane + "-hook"))
    # Reflector evidence is outside the harness-owned ledgers and survives select.
    lane_ledgers = {
        lane: git_repo / ".gepa/runs" / run_id / "lanes" / lane / "pareto.jsonl"
        for lane in ("lane-1", "lane-2")
    }
    before = {lane: path.read_bytes() for lane, path in lane_ledgers.items()}
    assert all(len(raw.splitlines()) == 3 for raw in before.values())
    harness_rows = (git_repo / ".gepa/runs" / run_id / "pareto.jsonl").read_text()
    assert all(json.loads(row)["lane"] is None for row in harness_rows.splitlines())
    result = _run("-G", str(git_repo / ".gepa"), "run", "select", "--run-id", run_id)
    assert result.exit_code == 0, (result.output, result.exception)
    assert "restored from the harness record" not in result.output
    assert {lane: path.read_bytes() for lane, path in lane_ledgers.items()} == before
    assert _run_payload(result.output)["best_commit_sha"] == candidate_sha
    # Reflector-written training rows cannot exhaust the private harness budget.
    assert _run_payload(result.output)["status"] == "running"
    from pydantic_ai_gepa.cli.run import _load_state
    from pydantic_ai_gepa.cli.select import _phase_finalize

    with harness_environment(), record.route():
        state, _, _ = _phase_finalize(git_repo, _load_state(run_id), {})
        assert state.status == "done"
    assert _git(record.repository, "rev-parse", "HEAD") == candidate_sha
    assert _git(git_repo, "rev-parse", "HEAD") == source_sha
    for lane in ("lane-1", "lane-2"):
        assert not (git_repo.parent / (git_repo.name + "-" + lane + "-hook")).exists()
        assert not lane_path(git_repo, run_id, lane).exists()
    assert str(private) not in result.output


def test_trusted_scorer_uses_original_repository_with_independent_seed(
    git_repo, private, protocol_backend, monkeypatch
):
    from pydantic_ai_gepa.cli.lane_repositories import lane_path

    config = git_repo / ".gepa/gepa.toml"
    config.write_text(
        config.read_text()
        + '\n[acceptance]\npinned_scorer = true\ntrusted_scorer = true\ncomponent_files = ["score.txt"]\n'
    )
    (git_repo / "task_pkg/evaluation.py").write_text(
        "import json, os\n"
        "async def evaluate(case):\n"
        "    return json.loads(os.environ['GEPA_CANDIDATE_COMPONENTS_JSON'])['score.txt']\n"
    )
    _git(git_repo, "add", ".")
    _git(git_repo, "commit", "-m", "Trusted scorer")
    revision = _git(git_repo, "rev-parse", "HEAD")
    monkeypatch.setenv("GEPA_HARNESS_SCORER_REVISION", revision)
    export = git_repo.parent / (git_repo.name + "-export")
    (export / "api").mkdir(parents=True)
    (export / "api/score.txt").write_text("bad")
    (export / "api/pyproject.toml").write_text(
        '[project]\nname="export"\nversion="0.0.0"\n'
    )
    _git(export, "init")
    _git(export, "config", "user.name", "Tests")
    _git(export, "config", "user.email", "tests@example.com")
    _git(export, "add", ".")
    _git(export, "commit", "-m", "History-free candidate")
    result = _run(
        "run",
        "start",
        "--candidate-root",
        str(export / "api"),
        "--lanes",
        "1",
        "--size",
        "1",
        "--max-iterations",
        "20",
        "--acceptance-repetitions",
        "1",
    )
    assert result.exit_code == 0, (result.output, result.exception)
    state = _run_payload(result.output)
    assert state["best_commit_sha"] == _git(export, "rev-parse", "HEAD")
    lane = lane_path(git_repo, str(state["run_id"]), "lane-1")
    assert not (lane / "task_pkg").exists()
    with pytest.raises(subprocess.CalledProcessError):
        _git(lane, "cat-file", "-e", revision)
    assert _git(git_repo, "rev-parse", "HEAD") == revision
    assert str(private) not in result.output


def test_heldout_select_refuses_legacy_state(git_repo, private, monkeypatch):
    from types import SimpleNamespace
    from pydantic_ai_gepa.cli.select import _run_select_locked

    state = SimpleNamespace(status="paused_for_reflection", heldout_required=False)
    with pytest.raises(typer.BadParameter, match="start a new run") as error:
        _run_select_locked(git_repo, state)
    assert str(private) not in str(error.value)


def test_cached_index_tracks_edits_and_head_tree_changes(git_repo, monkeypatch):
    import os
    from pydantic_ai_gepa.cli.safe_git import SafeGit

    original = SafeGit._run
    operations = []

    def record(self, args, **kwargs):
        operations.append(args[0])
        return original(self, args, **kwargs)

    monkeypatch.setattr(SafeGit, "_run", record)
    initial = git_candidate_state(git_repo)
    with safe_repository(git_repo) as git:
        private_dir, index = git.directory, git.index.path
    operations.clear()
    assert git_candidate_state(git_repo) == initial
    assert operations == ["rev-parse", "update-index", "diff", "ls-files"]
    with safe_repository(git_repo) as git:
        assert (git.directory, git.index.path) == (private_dir, index)
    path = git_repo / "score.txt"
    before = path.stat()
    path.write_text("new\n")  # Same size and restored mtime cannot hide an edit.
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert git_candidate_state(git_repo).dirty
    path.write_text("bad\n")
    assert not git_candidate_state(git_repo).dirty
    (git_repo / "untracked").write_text("untracked")
    assert git_candidate_state(git_repo).dirty
    (git_repo / "untracked").unlink()
    # A new commit with the same tree preserves the private stat cache.
    _git(git_repo, "commit", "--allow-empty", "-m", "Same tree")
    operations.clear()
    same_tree = git_candidate_state(git_repo)
    assert not same_tree.dirty and same_tree.commit_sha != initial.commit_sha
    assert "read-tree" not in operations
    path.write_text("new\n")
    _git(git_repo, "add", "score.txt")
    _git(git_repo, "commit", "-m", "New tree")
    operations.clear()
    assert not git_candidate_state(git_repo).dirty
    assert operations.count("read-tree") == 1
    _git(git_repo, "reset", "--hard", initial.commit_sha)
    assert git_candidate_state(git_repo) == initial


def test_shared_repository_has_separate_private_worktree_indexes(git_repo, tmp_path):
    other = tmp_path.parent / (tmp_path.name + "-other")
    _git(git_repo, "worktree", "add", "-b", "other", str(other))
    with safe_repository(git_repo) as git:
        shared, first_index = git.directory, git.index.path
    with safe_repository(other) as git:
        assert git.directory == shared
        assert git.index.path != first_index
    initial = git_candidate_state(git_repo)
    (other / "score.txt").write_text("other\n")
    assert git_candidate_state(other).dirty
    assert git_candidate_state(git_repo) == initial
    assert git_candidate_state(other).dirty


def test_cached_repository_refreshes_excludes_and_ignores_late_poison(
    git_repo, tmp_path
):
    assert not git_candidate_state(git_repo).dirty
    sentinel = tmp_path.parent / (tmp_path.name + "-late-poison")
    poison(git_repo, sentinel)
    (git_repo / "ignored-file").write_text("ordinary data")
    assert git_candidate_state(git_repo).dirty
    exclude = git_repo / ".git/info/exclude"
    exclude.write_text("ignored-file\n")
    assert not git_candidate_state(git_repo).dirty
    exclude.write_text("")
    assert git_candidate_state(git_repo).dirty
    assert not sentinel.exists()


def test_cached_index_invalidates_when_builtin_attributes_change(git_repo):
    import os

    (git_repo / ".gitattributes").write_text("* text\n")
    text = git_repo / "crlf.txt"
    text.write_bytes(b"text\r\n")
    _git(git_repo, "add", ".gitattributes", "crlf.txt")
    _git(git_repo, "commit", "-m", "Built-in text conversion")
    os.utime(text, (1000000000, 1000000000))  # Non-racy Git stat cache entry.
    assert not git_candidate_state(git_repo).dirty
    (git_repo / ".gitattributes").write_text("")
    warm = git_candidate_state(git_repo)
    with safe_repository(git_repo) as git:
        git.index.path.unlink()
        git.index.initialized = False
    cold = git_candidate_state(git_repo)
    assert warm == cold


def test_harness_git_storage_is_beside_heldout_not_in_public_temp(
    git_repo, monkeypatch
):
    import tempfile

    # The synthetic protected root is outside pytest's public temp storage and
    # outside the reflector's fixture checkout, like a real harness-private set.
    with tempfile.TemporaryDirectory(
        prefix="safe-git-private-test-", dir=Path(__file__).resolve().parents[2]
    ) as directory:
        protected = Path(directory).resolve()
        dataset = protected / "heldout.jsonl"
        dataset.write_text("{}\n")
        monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(dataset))
        with safe_repository(git_repo) as git:
            assert git.private.parent == protected / ".gepa-heldout/git"
            for path in (
                git.private,
                git.directory,
                git.index.path,
                Path(git.env["HOME"]),
                Path(git.env["XDG_CONFIG_HOME"]),
            ):
                assert path.is_relative_to(protected)
                assert not path.is_relative_to(Path(tempfile.gettempdir()).resolve())
                assert not path.is_relative_to(Path("/tmp").resolve())
            assert not git.private.stat().st_mode & 0o077
            assert not git.private.parent.stat().st_mode & 0o077
            assert not git.private.parent.parent.stat().st_mode & 0o077


def test_harness_never_reuses_poisoned_reflector_temp_cache(
    git_repo, private, monkeypatch
):
    attributes(git_repo)
    monkeypatch.delenv("GEPA_HELDOUT_DATASET")
    git_candidate_state(git_repo)
    with safe_repository(git_repo) as git:
        public_storage = git.private
        public_config = git.directory / "config"
    sentinel = private.parent / "temp-cache-executed"
    script = private.parent / "temp-cache-hook.sh"
    script.write_text(
        f"#!/bin/sh\necho executed >> {shlex.quote(str(sentinel))}\ncat\n"
    )
    script.chmod(0o755)
    with public_config.open("a") as config:
        config.write(
            f'\n[core]\n fsmonitor = {script}\n[filter "x"]\n clean = {script}\n'
        )
    (git_repo / "score.txt").write_text("changed\n")
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(private))
    assert git_candidate_state(git_repo).dirty
    assert git_candidate_state(git_repo).dirty
    with safe_repository(git_repo) as git:
        assert git.private != public_storage
        assert git.private.is_relative_to(private.parent / ".gepa-heldout")
    assert not sentinel.exists()


def test_harness_private_storage_failure_has_no_temp_fallback(
    git_repo, private, monkeypatch
):
    import tempfile
    from pydantic_ai_gepa.cli.candidates import GitCandidateError

    # Warm a reflector-side cache first; a failure must not reuse that either.
    monkeypatch.delenv("GEPA_HELDOUT_DATASET")
    git_candidate_state(git_repo)
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(private))
    (private.parent / ".gepa-heldout").write_text("not a directory")

    def unexpected(*args, **kwargs):
        pytest.fail("Private-directory failure must not fall back to temporary storage")

    monkeypatch.setattr(tempfile, "mkdtemp", unexpected)
    with pytest.raises(GitCandidateError):
        git_candidate_state(git_repo)


@pytest.mark.parametrize("object_format", ["sha1", "sha256"])
@pytest.mark.parametrize("linked", [False, True])
def test_retention_survives_gc_without_running_reference_hook(
    tmp_path, monkeypatch, object_format, linked
):
    from pydantic_ai_gepa.cli.front import _pin_git_candidate

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", f"--object-format={object_format}")
    _git(repo, "config", "gc.auto", "0")
    _git(repo, "config", "maintenance.auto", "false")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "file").write_text("seed\n")
    _git(repo, "add", "file")
    _git(repo, "commit", "-m", "Seed")
    seed = _git(repo, "rev-parse", "HEAD")
    root = repo
    if linked:
        root = tmp_path / "linked"
        _git(repo, "worktree", "add", "-b", "candidate", str(root))
    (root / "file").write_text("candidate\n")
    _git(root, "commit", "-am", "Candidate")
    sha = _git(root, "rev-parse", "HEAD")
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    sentinel = tmp_path / "executed"
    hook = hooks / "reference-transaction"
    hook.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(sentinel))}\n")
    hook.chmod(0o755)
    _git(root, "config", "core.hooksPath", str(hooks))
    _git(root, "update-ref", "refs/control", sha)
    assert sentinel.exists(), "Ordinary update-ref must execute the hostile hook"
    _git(root, "update-ref", "-d", "refs/control")
    sentinel.unlink()
    dataset = tmp_path / "private/heldout.jsonl"
    dataset.parent.mkdir()
    dataset.write_text("{}\n")
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(dataset))
    _pin_git_candidate(root, "public-run", sha[:12], sha)
    assert not sentinel.exists()
    ref = f"refs/gepa/public-run/{sha[:12]}"
    assert _git(root, "show-ref", "--verify", ref).split()[0] == sha
    # Exercise both packed refs and an already existing retention pin.
    _git(root, "-c", "core.hooksPath=/dev/null", "pack-refs", "--all")
    _pin_git_candidate(root, "public-run", sha[:12], sha)
    _pin_git_candidate(root, "public-run", sha[:12], sha)
    assert not sentinel.exists()
    _git(root, "-c", "core.hooksPath=/dev/null", "reset", "--hard", seed)
    _git(root, "reflog", "expire", "--expire=now", "--all")
    _git(root, "-c", "core.hooksPath=/dev/null", "gc", "--prune=now")
    assert _git(root, "cat-file", "-t", sha) == "commit"
    assert _git(root, "show", f"{sha}:file") == "candidate"


@pytest.mark.parametrize(
    "redirect",
    [
        "refs",
        "refs/gepa",
        "refs/gepa/run",
        "refs/gepa/run/candidate",
        "refs/gepa/run/candidate.lock",
    ],
)
def test_retention_refuses_symlink_components(git_repo, private, redirect):
    from pydantic_ai_gepa.cli.front import _pin_git_candidate

    sha = _git(git_repo, "rev-parse", "HEAD")
    metadata = git_repo / ".git"
    path = metadata / redirect
    target = private.parent / "redirect"
    target.mkdir()
    (target / "untouched").write_text("private")
    if path.exists():
        path.rename(metadata / "original-refs")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=True)
    with pytest.raises(typer.BadParameter, match="Could not retain") as error:
        _pin_git_candidate(git_repo, "run", "candidate", sha)
    assert str(private) not in str(error.value)
    assert list(target.iterdir()) == [target / "untouched"]
    assert (target / "untouched").read_text() == "private"


@pytest.mark.parametrize("layout", ["reftable", "gitdir-redirect", "gitdir-symlink"])
def test_retention_refuses_unsupported_layout(git_repo, private, layout):
    from pydantic_ai_gepa.cli.front import _pin_git_candidate

    sha = _git(git_repo, "rev-parse", "HEAD")
    marker = git_repo / ".git"
    common = marker
    if layout == "reftable":
        (marker / "reftable").mkdir()
    else:
        moved = private.parent / "redirected-git"
        marker.rename(moved)
        common = moved
        if layout == "gitdir-redirect":
            marker.write_text(f"gitdir: {moved}\n")
        else:
            marker.symlink_to(moved, target_is_directory=True)
    with pytest.raises(typer.BadParameter, match="Could not retain"):
        _pin_git_candidate(git_repo, "run", "candidate", sha)
    assert not (common / "refs/gepa").exists()


@pytest.mark.parametrize("kind", ["missing", "blob", "locked"])
def test_retention_refuses_invalid_objects_and_lock_conflicts(git_repo, private, kind):
    from pydantic_ai_gepa.cli.front import _pin_git_candidate

    sha = _git(git_repo, "rev-parse", "HEAD")
    ref = git_repo / ".git/refs/gepa/run/candidate"
    if kind == "missing":
        sha = "0" * 40
    elif kind == "blob":
        sha = _git(git_repo, "rev-parse", "HEAD:score.txt")
    else:
        ref.parent.mkdir(parents=True)
        ref.with_suffix(".lock").write_text("another writer")
    with pytest.raises(typer.BadParameter, match="Could not retain"):
        _pin_git_candidate(git_repo, "run", "candidate", sha)
    assert not ref.exists()
    if kind == "locked":
        assert ref.with_suffix(".lock").read_text() == "another writer"


@pytest.fixture
def fake_git_filesystem(monkeypatch):
    from pydantic_ai_gepa.cli import safe_git

    candidate = Path("/trusted/bin/git")
    entries = {candidate: SimpleNamespace(st_uid=0, st_mode=stat.S_IFREG | 0o755)}
    aliases = {}
    candidates = [candidate]
    locations = safe_git._git_locations
    safe_git._git_executable.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(Path, "resolve", lambda path, **_: aliases.get(path, path))
        patch.setattr(
            Path,
            "stat",
            lambda path, **_: entries.get(
                path, SimpleNamespace(st_uid=0, st_mode=stat.S_IFDIR | 0o755)
            ),
        )
        patch.setattr(Path, "lstat", lambda path: path.stat())
        patch.setattr(os, "access", lambda *_: True)
        patch.setattr(safe_git, "_git_locations", lambda: candidates)
        patch.setattr(safe_git.sys, "platform", "darwin")
        try:
            yield safe_git, entries, aliases, candidates, locations
        finally:
            safe_git._git_executable.cache_clear()


@pytest.mark.parametrize(
    "unsafe",
    [
        "user",
        "group",
        "other",
        "parent-user",
        "parent-group",
        "directory",
        "not-executable",
        "symlink",
        "shim",
        "shim-symlink",
    ],
)
def test_git_executable_refuses_unsafe_installations(fake_git_filesystem, unsafe):
    safe_git, entries, aliases, candidates, _ = fake_git_filesystem
    candidate = candidates[0]
    if unsafe == "user":
        entries[candidate].st_uid = 501
    elif unsafe in {"group", "other"}:
        entries[candidate].st_mode |= 0o020 if unsafe == "group" else 0o002
    elif unsafe.startswith("parent-"):
        entries[candidate.parent] = SimpleNamespace(
            st_uid=501 if unsafe == "parent-user" else 0,
            st_mode=stat.S_IFDIR | (0o775 if unsafe == "parent-group" else 0o755),
        )
    elif unsafe == "directory":
        entries[candidate].st_mode = stat.S_IFDIR | 0o755
    elif unsafe == "not-executable":
        entries[candidate].st_mode = stat.S_IFREG | 0o644
    elif unsafe == "symlink":
        target = Path("/user/bin/git")
        aliases[candidate] = target
        entries[target] = SimpleNamespace(st_uid=501, st_mode=stat.S_IFREG | 0o755)
    elif unsafe == "shim":
        candidates[0] = Path("/usr/bin/git")
        entries[candidates[0]] = entries[candidate]
    else:
        aliases[candidate] = Path("/usr/bin/git")
        entries[aliases[candidate]] = entries[candidate]
    with pytest.raises(safe_git.GitExecutableError, match="No trusted Git") as error:
        safe_git._git_executable(123)
    assert str(candidate) not in str(error.value)


def test_git_executable_is_pinned_once_per_process(fake_git_filesystem):
    safe_git, entries, _, candidates, _ = fake_git_filesystem
    assert safe_git._git_executable(123) == str(candidates[0])
    entries[candidates[0]].st_uid = 501
    assert safe_git._git_executable(123) == str(candidates[0])
    with pytest.raises(safe_git.GitExecutableError):
        safe_git._git_executable(124)


def test_git_executable_falls_back_to_a_trusted_installation(fake_git_filesystem):
    safe_git, entries, _, candidates, _ = fake_git_filesystem
    entries[candidates[0]].st_mode |= 0o020
    fallback = Path("/fallback/bin/git")
    candidates.append(fallback)
    entries[fallback] = SimpleNamespace(st_uid=0, st_mode=stat.S_IFREG | 0o755)
    assert safe_git._git_executable(123) == str(fallback)


def test_git_locations_ignore_user_tool_selection(fake_git_filesystem, monkeypatch):
    safe_git, entries, _, _, locations = fake_git_filesystem
    monkeypatch.setenv("DEVELOPER_DIR", "/user/developer")
    monkeypatch.setenv("PATH", "/user/bin")
    monkeypatch.setattr(os, "readlink", lambda path: "/root-developer")
    selection = Path("/var/db/xcode_select_link")
    entries[selection] = SimpleNamespace(st_uid=0, st_mode=stat.S_IFLNK | 0o777)
    fallback = Path("/Library/Developer/CommandLineTools/usr/bin/git")
    assert locations() == [Path("/root-developer/usr/bin/git"), fallback]
    entries[selection].st_uid = 501
    assert locations() == [fallback]
    # Undo before the fixture restores the real platform. A test-level
    # monkeypatch would restore the fixture's "darwin" last and leak it.
    with monkeypatch.context() as patch:
        patch.setattr(safe_git.sys, "platform", "linux")
        assert locations() == [Path("/usr/bin/git"), Path("/bin/git")]


def test_real_git_binary_is_pinned_and_not_the_macos_shim(git_repo):
    from pydantic_ai_gepa.cli.safe_git import _git_executable, _root_owned_path

    path = Path(_git_executable(os.getpid()))
    assert path.is_absolute()
    assert _root_owned_path(path, executable=True) == path
    if sys.platform == "darwin":
        assert path != Path("/usr/bin/git")
    with safe_repository(git_repo) as git:
        assert git.command[0] == str(path)
        assert "GIT_EXEC_PATH" not in git.env


@pytest.mark.parametrize("kind", ["symlink", "directory-symlink", "fifo", "hardlink"])
def test_unsafe_exclude_is_empty_and_never_reads_heldout(
    git_repo, private, monkeypatch, kind
):
    (git_repo / "guess-row-17").write_text("guess")
    baseline = git_candidate_state(git_repo)
    assert baseline.dirty
    exclude = git_repo / ".git/info/exclude"
    secret = private.parent / "exclude"
    secret.write_text("guess-row-17\n")
    exclude.unlink()
    if kind == "symlink":
        exclude.symlink_to(secret)
    elif kind == "directory-symlink":
        exclude.parent.rename(git_repo / ".git/original-info")
        exclude.parent.symlink_to(private.parent, target_is_directory=True)
    elif kind == "hardlink":
        os.link(secret, exclude)
    else:
        os.mkfifo(exclude)
    forbidden = secret.stat()
    original = os.read

    def read(fd, size):
        info = os.fstat(fd)
        assert (info.st_dev, info.st_ino) != (forbidden.st_dev, forbidden.st_ino)
        return original(fd, size)

    monkeypatch.setattr(os, "read", read)
    assert git_candidate_state(git_repo) == baseline
    secret.write_text("different-heldout-line\n")
    assert git_candidate_state(git_repo) == baseline


@pytest.mark.parametrize(
    "metadata",
    ["HEAD", "loose", "packed-refs", "commondir", "config", ".git", "gitdir"],
)
def test_metadata_symlinks_are_refused_without_reading_targets(
    git_repo, private, monkeypatch, metadata
):
    from pydantic_ai_gepa.cli.safe_git import pin_commit

    repository = Repository.discover(git_repo)
    branch, sha = repository.head()
    assert branch and sha
    root = git_repo
    target = repository.git_dir / metadata
    if metadata == "loose":
        target = repository.common_dir / branch
    elif metadata == "packed-refs":
        (repository.common_dir / branch).unlink()
    elif metadata == "config":
        (repository.git_dir / "HEAD").write_text("ref: refs/heads/unborn\n")
    elif metadata in {".git", "gitdir"}:
        root = private.parent / "linked"
        _git(git_repo, "worktree", "add", "-b", "linked", str(root))
        linked = Repository.discover(root)
        target = root / ".git" if metadata == ".git" else linked.git_dir / "gitdir"
    target.unlink(missing_ok=True)
    secret = private.parent / "metadata-target"
    secret.write_text("must not be read\n")
    target.symlink_to(secret)
    forbidden = secret.stat()
    original = os.read

    def read(fd, size):
        info = os.fstat(fd)
        assert (info.st_dev, info.st_ino) != (forbidden.st_dev, forbidden.st_ino)
        return original(fd, size)

    monkeypatch.setattr(os, "read", read)
    with pytest.raises(OSError):
        if metadata == "gitdir":
            pin_commit(root, "refs/gepa/run/candidate", sha)
        else:
            with safe_repository(root):
                pytest.fail("Unsafe metadata must be refused")


@pytest.mark.parametrize("metadata", ["HEAD", "commondir", "config", ".git"])
def test_metadata_fifo_fails_without_hanging(git_repo, metadata):
    if metadata == ".git":
        (git_repo / ".git").rename(git_repo / "original-git")
        path = git_repo / ".git"
    else:
        path = git_repo / ".git" / metadata
        if metadata == "config":
            (git_repo / ".git/HEAD").write_text("ref: refs/heads/unborn\n")
        path.unlink(missing_ok=True)
    os.mkfifo(path)
    # A separate process and timeout keep a regression from wedging pytest.
    script = """from pathlib import Path
from pydantic_ai_gepa.cli.safe_git import safe_repository
try:
    with safe_repository(Path.cwd()):
        raise AssertionError('FIFO metadata accepted')
except OSError:
    pass
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=git_repo, capture_output=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


def test_metadata_size_is_bounded_before_reading(git_repo, monkeypatch):
    from pydantic_ai_gepa.cli.safe_git import SafeGitError, _REF_LIMIT

    (git_repo / ".git/HEAD").write_bytes(b"a" * (_REF_LIMIT + 1))
    monkeypatch.setattr(os, "read", lambda *_: pytest.fail("Oversized metadata read"))
    with pytest.raises(SafeGitError):
        Repository.discover(git_repo).head()


@pytest.mark.parametrize("object_format", ["sha1", "sha256"])
def test_unborn_object_format_is_parsed_from_safe_stdin(tmp_path, object_format):
    _git(tmp_path, "init", f"--object-format={object_format}")
    with safe_repository(tmp_path) as git:
        assert git.object_format == object_format
        assert git.head_oid is None


@pytest.mark.parametrize("source", ["xdg", "core.excludesFile"])
def test_global_excludes_do_not_hide_git_candidates(
    git_repo, tmp_path, monkeypatch, source
):
    home = tmp_path.parent / (tmp_path.name + "-home")
    (home / "git").mkdir(parents=True)
    ignore = home / "git/ignore"
    ignore.write_text("globally-ignored\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    if source == "core.excludesFile":
        config = home / ".gitconfig"
        config.write_text(f"[core]\n excludesFile = {ignore}\n")
        # Use a non-default location so this case exercises the config setting.
        ignore.rename(home / "custom-ignore")
        config.write_text(f"[core]\n excludesFile = {home / 'custom-ignore'}\n")
    (git_repo / "globally-ignored").write_text("candidate content")
    assert not _git(git_repo, "status", "--porcelain")
    assert git_candidate_state(git_repo).dirty
    exclude = git_repo / ".git/info/exclude"
    exclude.write_text("globally-ignored\n")
    assert not git_candidate_state(git_repo).dirty

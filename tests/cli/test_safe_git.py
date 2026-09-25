"""Executable Git configuration must never cross the held-out boundary."""

import hashlib
from pathlib import Path
import shlex
import subprocess

import pytest
import typer

from pydantic_ai_gepa.cli.candidates import git_candidate_state
from pydantic_ai_gepa.cli.layout import candidate_identity_exempt_paths, git_root
from pydantic_ai_gepa.cli.runs import current_commit_sha
from pydantic_ai_gepa.cli.safe_git import Repository, run_git, safe_repository
from pydantic_ai_gepa.cli.scoring_sandbox import (
    ScoringSandboxError,
    _checkout_entries,
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
    (metadata.common_dir / "info/attributes").write_text("* filter=x diff=x\n")


def attributes(repo):
    (repo / ".gitattributes").write_text("* filter=x diff=x\n")
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-m", "Attribute names with no executable drivers")


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


def test_harness_once_scores_with_hostile_git_config(
    git_repo, private, protocol_backend, monkeypatch, tmp_path
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
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(private))
    served = _run("harness", "serve", "--run-id", run_id, "--once")
    assert served.exit_code == 0, (served.output, served.exception)
    monkeypatch.delenv("GEPA_HELDOUT_DATASET")
    delivered = _run("run", "continue", "--run-id", run_id, "--wait-secs", "0")
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


def test_heldout_lane_start_refused_before_scoring(git_repo, private):
    result = _run("run", "start", "--lanes", "2")
    assert result.exit_code == 2, (result.output, result.exception)
    assert "single-checkout" in result.output
    assert str(private) not in result.output
    assert not (git_repo / "worktrees").exists()


def test_heldout_select_refused_before_state_changes(git_repo, private, monkeypatch):
    from types import SimpleNamespace
    from pydantic_ai_gepa.cli.select import _run_select_locked

    # This also refuses a training lane run if held-out access is introduced
    # later, rather than only checking its persisted heldout_required flag.
    state = SimpleNamespace(status="paused_for_reflection", heldout_required=False)
    with pytest.raises(typer.BadParameter, match="single-checkout") as error:
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

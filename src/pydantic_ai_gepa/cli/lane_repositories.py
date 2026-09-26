"""Independent lane repositories and durable, controller-owned candidate objects.

No Git process is ever started with a lane's GIT_DIR. Object files are copied
with no-follow opens before Git sees them; only the nominated reachable closure
is retained. This module owns the small private layout record, independently of
the run-state persistence layer.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import BinaryIO, Iterator

import typer

from .safe_git import (
    Repository,
    _directory_fd,
    _git_executable,
    _OPTIONS,
    run_git,
    safe_repository,
)


class CandidateAncestryError(typer.BadParameter):
    """The proposal is unrelated to the privately recorded seed."""


_ACTIVE: ContextVar[tuple[Path, Path] | None] = ContextVar(
    "lane_repository", default=None
)


def candidate_root(root: Path) -> Path:
    active = _ACTIVE.get()
    return active[1] if active and root.resolve() == active[0] else root


def _identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise typer.BadParameter("Invalid repository identity.")
    return value


def lanes_root(root: Path) -> Path:
    from .layout import gepa_dir

    public = gepa_dir(root).resolve()
    return public.with_name(public.name + ".lanes")


def lane_path(root: Path, run_id: str, lane: str) -> Path:
    return lanes_root(root) / _identifier(run_id) / _identifier(lane)


def _storage(root: Path, run_id: str) -> Path:
    from .layout import gepa_dir
    from .validation import heldout_dataset

    public = gepa_dir(root).resolve()
    dataset = heldout_dataset(required=False)
    if dataset:
        owner = hashlib.sha256(os.fsencode(public)).hexdigest()
        parent = (
            Path(dataset).resolve().parent / ".gepa-heldout" / "repositories" / owner
        )
    else:
        parent = public.with_name(public.name + ".repositories")
    return parent / _identifier(run_id)


def _command(
    root: Path,
    *args: str,
    input: bytes | None = None,
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
) -> bytes:
    """Only for our own repository, never for source/lane metadata."""
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(root),
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_DIR": str(root / ".git"),
        "GIT_WORK_TREE": str(root),
    }
    command = [_git_executable(os.getpid()), "--no-pager"]
    for option in _OPTIONS:
        command.extend(("-c", option))
    return (
        subprocess.run(
            command + list(args),
            cwd=root,
            env=env,
            input=input,
            check=True,
            stdin=stdin,
            stdout=stdout if stdout is not None else subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        or b""
    )


def _init(root: Path, object_format: str) -> None:
    root.mkdir(mode=0o700)
    _command(root, "init", "--template=", f"--object-format={object_format}")
    # Lane commits need a stable identity without access to global Git config.
    _command(root, "config", "user.name", "GEPA")
    _command(root, "config", "user.email", "gepa@localhost")


def _copy_objects(source: Path, destination: Path) -> None:
    """Snapshot bytes through directory descriptors, rejecting links/special files."""

    def copy(fd: int, target: Path) -> None:
        for name in os.listdir(fd):
            if name in {"alternates", "http-alternates"}:
                raise typer.BadParameter("Lane object alternates are forbidden.")
            child = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
            )
            try:
                info = os.fstat(child)
                mode = info.st_mode
                if stat.S_ISDIR(mode):
                    (target / name).mkdir(exist_ok=True)
                    copy(child, target / name)
                elif stat.S_ISREG(mode):
                    if info.st_nlink != 1:
                        raise typer.BadParameter(
                            "Hard-linked Git objects are forbidden."
                        )
                    with (
                        os.fdopen(os.dup(child), "rb") as reader,
                        (target / name).open("wb") as writer,
                    ):
                        shutil.copyfileobj(reader, writer)
                else:
                    raise typer.BadParameter("Unsupported Git object storage.")
            finally:
                os.close(child)

    with _directory_fd(source) as fd:
        copy(fd, destination)


def import_commit(
    source: Path, sha: str, destination: Path, *, lane: bool = True
) -> None:
    """Retain a nominated commit without executing source hooks/config/transports."""
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha):
        raise typer.BadParameter("A full candidate commit SHA is required.")
    if lane:
        # Independent lanes have exactly one conventional metadata directory.
        # Do not discover through attacker-supplied gitdir/commondir pointers,
        # even to subsequently reject them: discovery would read their targets.
        metadata = source.absolute() / ".git"
        with _directory_fd(metadata) as fd:
            if "commondir" in os.listdir(fd):
                raise typer.BadParameter(
                    "Linked lane repositories are unsupported; start a new run."
                )
        repo = Repository(source, metadata, metadata)
    else:
        repo = Repository.discover(source)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix="objects-") as temp:
        snapshot = Path(temp) / "snapshot"
        _init(snapshot, "sha256" if len(sha) == 64 else "sha1")
        _copy_objects(repo.common_dir / "objects", snapshot / ".git/objects")
        # This config-free, owned GIT_DIR sees only the copied source bytes.
        if _command(snapshot, "cat-file", "-t", sha) != b"commit\n":
            raise typer.BadParameter("Candidate object is not a commit.")
        with (Path(temp) / "transfer.pack").open("w+b") as pack:
            _command(
                snapshot,
                "pack-objects",
                "--stdout",
                "--revs",
                input=(sha + "\n").encode(),
                stdout=pack,
            )
            pack.seek(0)
            _command(destination, "index-pack", "--stdin", "--strict", stdin=pack)
        _command(destination, "rev-list", "--objects", "--missing=error", sha)
        _command(destination, "update-ref", "refs/gepa/retained/" + sha, sha)


def checkout(source: Path, sha: str, destination: Path, branch: str) -> None:
    """Create a standalone repository using raw blobs, without checkout filters."""
    from .scoring_sandbox import _checkout_entries, _write_checkout_blobs

    _init(destination, "sha256" if len(sha) == 64 else "sha1")
    try:
        import_commit(source, sha, destination, lane=False)
        with safe_repository(destination) as git:
            listing = git.run(
                "ls-tree",
                "-r",
                "-t",
                "-z",
                "--full-tree",
                sha,
                check=True,
                capture_output=True,
            ).stdout
            entries = _checkout_entries(destination, listing)
            _write_checkout_blobs(git, entries)
        _command(destination, "check-ref-format", "refs/heads/" + branch)
        _command(destination, "update-ref", "refs/heads/" + branch, sha)
        _command(destination, "symbolic-ref", "HEAD", "refs/heads/" + branch)
        _command(destination, "read-tree", sha)
    except BaseException:
        shutil.rmtree(destination)
        raise


@dataclass(frozen=True)
class Repositories:
    root: Path
    run_id: str
    directory: Path
    prefix: Path
    seed: str

    @property
    def repository(self) -> Path:
        return self.directory / "primary"

    @property
    def project(self) -> Path:
        return self.repository / self.prefix

    @contextmanager
    def route(self) -> Iterator[None]:
        token = _ACTIVE.set((self.root.resolve(), self.project))
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def import_lane(self, lane: str, sha: str) -> None:
        source = lane_path(self.root, self.run_id, lane)
        import_commit(source, sha, self.repository)
        result = run_git(self.repository, "merge-base", "--is-ancestor", self.seed, sha)
        if result.returncode:
            raise CandidateAncestryError(
                "Candidate does not descend from the run seed."
            )

    def candidate(self, sha: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha):
            raise typer.BadParameter("Invalid retained candidate SHA.")
        target = self.directory / ("candidate-" + sha)
        if not target.exists():
            checkout(self.repository, sha, target, "candidate")
        return target / self.prefix

    def create_lane(
        self, lane: str, sha: str, branch: str, *, replace: bool = False
    ) -> Path:
        path = lane_path(self.root, self.run_id, lane)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with _directory_fd(path.parent):
            pass
        if path.is_symlink() or (path.exists() and not replace):
            raise typer.BadParameter(
                "Lane destination already exists or is redirected."
            )
        # Build outside the reflector grant. An interrupted export can never
        # leave a partially initialized repository at the public lane path.
        with tempfile.TemporaryDirectory(dir=path.parent, prefix=".lane-") as temp:
            stage = Path(temp) / "repository"
            checkout(self.repository, sha, stage, branch)
            previous = path.with_name("." + path.name + ".previous")
            if previous.is_symlink():
                raise typer.BadParameter("Refusing redirected lane backup.")
            if previous.exists():
                shutil.rmtree(previous)
            if path.exists():
                path.rename(previous)
            stage.rename(path)
            if previous.exists():
                shutil.rmtree(previous)
        return path

    def remove_lane(self, lane: str) -> None:
        path = lane_path(self.root, self.run_id, lane)
        if path.is_symlink():
            raise typer.BadParameter("Refusing redirected lane directory.")
        with _directory_fd(path.parent):
            if path.exists():
                if not (path / ".git").is_dir() or (path / ".git").is_symlink():
                    raise typer.BadParameter(
                        "Linked lane repositories are unsupported; start a new run."
                    )
                shutil.rmtree(path)

    def promote(self, sha: str) -> None:
        if run_git(
            self.repository, "merge-base", "--is-ancestor", self.seed, sha
        ).returncode:
            raise CandidateAncestryError(
                "Candidate does not descend from the run seed."
            )
        # Our store is never reflector-writable. Replace the checkout without
        # loading candidate config, attributes, hooks or executable filters.
        stage = self.directory / "promoting"
        if stage.exists():
            shutil.rmtree(stage)
        checkout(self.repository, sha, stage, "gepa-primary")
        # Preserve every retained proposal for replay/adoption and merge advice.
        for ref in (
            _command(
                self.repository,
                "for-each-ref",
                "--format=%(objectname)",
                "refs/gepa/retained",
            )
            .decode()
            .splitlines()
        ):
            import_commit(self.repository, ref, stage, lane=False)
        old = self.directory / "previous"
        if old.exists():
            shutil.rmtree(old)
        self.repository.rename(old)
        stage.rename(self.repository)
        shutil.rmtree(old)


def initialize(root: Path, run_id: str, source: Path) -> Repositories:
    from .layout import git_root, project_prefix

    repository = git_root(source)
    prefix = project_prefix(source, repository)
    sha = run_git(
        repository, "rev-parse", "HEAD", check=True, capture_output=True, text=True
    ).stdout.strip()
    directory = _storage(root, run_id)
    directory.mkdir(parents=True, mode=0o700)
    with _directory_fd(directory):
        pass
    record = Repositories(root, run_id, directory, prefix, sha)
    checkout(repository, sha, record.repository, "gepa-primary")
    (directory / "layout.json").write_text(
        json.dumps({"version": 1, "prefix": str(prefix), "seed": sha})
    )
    return record


def load(root: Path, run_id: str) -> Repositories:
    directory = _storage(root, run_id)
    try:
        with _directory_fd(directory):
            data = json.loads((directory / "layout.json").read_text())
        prefix = Path(data["prefix"])
        if data["version"] != 1 or prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError
        # A crash between promotion renames leaves the previous owned repo.
        # Restoring it allows the checkpointed select to retry promotion.
        if not (directory / "primary").exists() and (directory / "previous").is_dir():
            (directory / "previous").rename(directory / "primary")
        return Repositories(root, run_id, directory, prefix, data["seed"])
    except (OSError, ValueError, KeyError):
        raise typer.BadParameter(
            "Independent lane repository record missing; start a new run (linked lanes cannot resume)."
        ) from None

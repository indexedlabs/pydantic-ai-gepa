"""Read untrusted Git objects through a private, configuration-free repository.

Only the listed plumbing/read commands are supported. Repository discovery and
HEAD resolution read files, never invoke Git in the source repository. In
particular, neither config.worktree nor config includes enter the private repo.
Public retention refs are published as data using no-follow directory handles.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import threading
from typing import Any, Iterator


class SafeGitError(OSError):
    """Repository metadata cannot be safely interpreted."""


@dataclass(frozen=True)
class Repository:
    root: Path
    git_dir: Path
    common_dir: Path

    @classmethod
    def discover(cls, start: Path) -> Repository:
        start = start.resolve()
        for root in (start, *start.parents):
            marker = root / ".git"
            if marker.is_dir():
                git_dir = marker.resolve()
            elif marker.is_file():
                value = marker.read_text().strip()
                if not value.startswith("gitdir: ") or "\n" in value:
                    raise SafeGitError("Invalid Git directory pointer.")
                git_dir = (root / value[8:]).resolve()
            else:
                continue
            common_file = git_dir / "commondir"
            common = (
                (git_dir / common_file.read_text().strip()).resolve()
                if common_file.exists()
                else git_dir
            )
            if not (common / "objects").is_dir():
                raise SafeGitError("Invalid Git object directory.")
            if (common / "reftable").exists():
                raise SafeGitError("Reftable repositories are not supported.")
            return cls(root, git_dir, common)
        raise FileNotFoundError("not a git repository")

    def head(self) -> tuple[str | None, str | None]:
        value = (self.git_dir / "HEAD").read_text().strip()
        branch = None
        seen: set[str] = set()
        while value.startswith("ref: "):
            ref = value[5:]
            if not re.fullmatch(r"refs/[\w./-]+", ref, flags=re.ASCII) or any(
                p in {"", ".", ".."} for p in ref.split("/")
            ):
                raise SafeGitError("Invalid symbolic Git ref.")
            if ref in seen or len(seen) >= 16:
                raise SafeGitError("Cyclic symbolic Git ref.")
            seen.add(ref)
            branch = branch or ref
            loose = self.common_dir / ref
            if loose.exists():
                value = loose.read_text().strip()
                continue
            packed = self.common_dir / "packed-refs"
            for line in packed.read_text().splitlines() if packed.exists() else ():
                oid, _, name = line.partition(" ")
                if name == ref:
                    value = oid
                    break
            else:
                return branch, None  # unborn HEAD
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value):
            raise SafeGitError("Invalid Git HEAD.")
        return branch, value


_READ_COMMANDS = {
    "rev-parse",
    "diff",
    "ls-files",
    "status",
    "ls-tree",
    "cat-file",
    "hash-object",
    "merge-base",
}
_OPTIONS = (
    "core.fsmonitor=false",
    "core.hooksPath=/dev/null",
    "core.attributesFile=/dev/null",
    "diff.external=",
    "core.untrackedCache=false",
    "submodule.recurse=false",
    "core.quotePath=true",
    "color.ui=false",
    "maintenance.auto=false",
    "gc.auto=0",
    "protocol.allow=never",
)


@dataclass
class _IndexState:
    path: Path
    initialized: bool = False
    head: str | None = None
    tree: str | None = None
    attribute_paths: tuple[Path, ...] = ()
    attributes_digest: bytes | None = None


class SafeGit:
    """Process-owned Git storage, with a separate index for each worktree.

    Access is serialized by safe_repository. Each context rebinds HEAD and the
    worktree; cached state consists only of harness-owned files and Git stat
    data. No source index, config, hooks or attributes are ever copied.
    """

    def __init__(
        self,
        repository: Repository,
        private: Path,
        cwd: Path,
        object_format: str | None,
    ):
        self.repository = repository
        self.private = private
        self.cwd = cwd.resolve()
        self.indexes: dict[tuple[Path, Path], _IndexState] = {}
        self.directory = private / "repo"
        self.directory.mkdir(mode=0o700)
        home = private / "home"
        home.mkdir(mode=0o700)
        self.env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home),
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_DIR": str(self.directory),
            "GIT_WORK_TREE": str(repository.root),
            "GIT_INDEX_FILE": str(private / "index"),
        }
        self.command = ["git", "--no-pager"]
        for option in _OPTIONS:
            self.command.extend(("-c", option))
        (self.directory / "objects/info").mkdir(parents=True)
        (self.directory / "refs").mkdir()
        (self.directory / "HEAD").write_text("ref: refs/heads/unborn\n")
        # Committed repositories advertise their format through the HEAD OID.
        # Only unborn repositories need explicit-file, no-includes config
        # parsing. It reads data in our owned repo, never source configuration.
        if object_format is None:
            fmt = self._run(
                [
                    "config",
                    "--no-includes",
                    "--file",
                    str(repository.common_dir / "config"),
                    "--get",
                    "extensions.objectformat",
                ],
                capture_output=True,
            )
            if fmt.returncode not in (0, 1) or fmt.stdout.strip() not in (
                b"",
                b"sha1",
                b"sha256",
            ):
                raise SafeGitError("Unsupported Git object format.")
            object_format = fmt.stdout.strip().decode() or "sha1"
        self.object_format = object_format
        (self.directory / "config").write_text(
            "[core]\n\trepositoryformatversion = 1\n\tbare = false\n"
            f"[extensions]\n\tobjectformat = {object_format}\n"
        )
        objects = os.fsencode(repository.common_dir / "objects")
        if b"\n" in objects or objects.startswith(b'"'):
            raise SafeGitError("Unsupported Git object directory path.")
        (self.directory / "objects/info/alternates").write_bytes(objects + b"\n")
        (self.directory / "info").mkdir()

    def bind(
        self, repository: Repository, cwd: Path, branch: str | None, oid: str | None
    ) -> None:
        self.repository = repository
        self.cwd = cwd.resolve()
        self.env["GIT_WORK_TREE"] = str(repository.root)
        if oid and len(oid) != (64 if self.object_format == "sha256" else 40):
            raise SafeGitError("Git HEAD object format mismatch.")
        if branch and oid:
            ref = self.directory / branch
            ref.parent.mkdir(parents=True, exist_ok=True)
            _write_changed(ref, (oid + "\n").encode())
            head = f"ref: {branch}\n"
        else:
            head = oid + "\n" if oid else "ref: refs/heads/unborn\n"
            if oid is None:
                unborn = self.directory / "refs/heads/unborn"
                if unborn.is_file():
                    unborn.unlink()
        _write_changed(self.directory / "HEAD", head.encode())
        self.head_oid = oid
        self.index = self.indexes.setdefault(
            (repository.root, repository.git_dir),
            _IndexState(self.private / f"index-{len(self.indexes)}"),
        )
        self.env["GIT_INDEX_FILE"] = str(self.index.path)
        self.index_ready = False
        # Excludes are inert patterns. Refresh them on every context so changes
        # to ignore rules cannot be hidden by the persistent private index.
        exclude = repository.common_dir / "info/exclude"
        _write_changed(
            self.directory / "info/exclude",
            exclude.read_bytes() if exclude.is_file() else b"",
        )

    def prepare_index(self) -> None:
        index = self.index
        if not index.initialized or index.head != self.head_oid:
            tree = (
                self._run(
                    ["rev-parse", "--verify", "HEAD^{tree}"],
                    check=True,
                    capture_output=True,
                )
                .stdout.decode()
                .strip()
                if self.head_oid
                else None
            )
            if not index.initialized or index.tree != tree:
                self._run(
                    ["read-tree", tree] if tree else ["read-tree", "--empty"],
                    check=True,
                    capture_output=True,
                )
                index.tree = tree
                index.attribute_paths = self.attribute_paths(tree)
            index.head = self.head_oid
            index.initialized = True
        digest = hashlib.sha256()
        for path in index.attribute_paths:
            # Git does not follow a symlinked .gitattributes file. A directory
            # replaced by a symlink cannot supply attributes for tracked files.
            if path.is_symlink() or not path.resolve().is_relative_to(
                self.repository.root
            ):
                continue
            if path.is_file():
                digest.update(os.fsencode(path))
                digest.update(b"\0")
                digest.update(path.read_bytes())
                digest.update(b"\0")
        attributes_digest = digest.digest()
        if (
            index.attributes_digest is not None
            and index.attributes_digest != attributes_digest
        ):
            # A stat match alone cannot validate a file when its built-in
            # conversion rules changed. read-tree may retain matching stats,
            # so discard the private index before repopulating it.
            index.initialized = False
            index.path.unlink(missing_ok=True)
            self._run(
                ["read-tree", index.tree] if index.tree else ["read-tree", "--empty"],
                check=True,
                capture_output=True,
            )
            index.initialized = True
        index.attributes_digest = attributes_digest
        # GIT_OPTIONAL_LOCKS=0 prevents diff/status from saving refreshed stats.
        # This explicit write updates only our private index, without trusting
        # reflector index extensions or descending into submodules.
        self._run(
            ["update-index", "-q", "--ignore-submodules", "--refresh"],
            check=True,
            capture_output=True,
        )
        self.index_ready = True

    def attribute_paths(self, tree: str | None) -> tuple[Path, ...]:
        if tree is None:
            return ()
        # Only ancestors of tracked files can affect their conversions. Avoid
        # walking ignored directories (venvs, dependencies, nested repos).
        listing = self._run(
            ["ls-tree", "-r", "-z", "--full-tree", "--name-only", tree],
            check=True,
            capture_output=True,
        ).stdout
        directories = {self.repository.root}
        for raw in listing.split(b"\0"):
            if not raw:
                continue
            relative = Path(os.fsdecode(raw))
            if relative.is_absolute() or ".." in relative.parts:
                raise SafeGitError("Invalid Git tree path.")
            directories.update(
                self.repository.root / parent for parent in relative.parents
            )
        return tuple(directory / ".gitattributes" for directory in sorted(directories))

    def _run(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        return subprocess.run(
            [*self.command, *args], cwd=self.cwd, env=self.env, **kwargs
        )

    def arguments(self, args: tuple[str, ...]) -> list[str]:
        if not args or args[0] not in _READ_COMMANDS:
            raise SafeGitError("Unsupported harness Git command.")
        if args[0] in {"diff", "ls-files", "status"} and not self.index_ready:
            self.prepare_index()
        result = list(args)
        if args[0] == "diff":
            result[1:1] = ["--no-textconv", "--no-ext-diff", "--ignore-submodules=all"]
        elif args[0] == "status":
            result.insert(1, "--ignore-submodules=all")
        return result

    def run(self, *args: str, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        return self._run(self.arguments(args), **kwargs)

    def popen(self, *args: str, **kwargs: Any) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [*self.command, *self.arguments(args)],
            cwd=self.cwd,
            env=self.env,
            **kwargs,
        )


def _write_changed(path: Path, content: bytes) -> None:
    if not path.exists() or path.read_bytes() != content:
        path.write_bytes(content)


_RepositoryOwner = tuple[int, Path, Path | None]
_repositories: dict[tuple[int, Path, Path | None, str], SafeGit] = {}
_repository_formats: dict[_RepositoryOwner, str] = {}
_repository_lock = threading.RLock()


def _after_fork() -> None:
    global _repository_lock
    _repository_lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _cleanup_repositories() -> None:
    # A forked child must not remove its parent's private storage.
    for (pid, _, _, _), git in tuple(_repositories.items()):
        if pid == os.getpid():
            shutil.rmtree(git.private, ignore_errors=True)


atexit.register(_cleanup_repositories)


def _private_storage() -> Path | None:
    from .validation import heldout_dataset

    dataset = heldout_dataset(required=False)
    if dataset is None:
        return None
    # /tmp is normally writable by the same-uid reflector. Permissions alone
    # cannot protect config/index/alternates there. Use the same protected
    # parent as private_checkout, and never fall back if creation fails.
    storage = Path(dataset).resolve().parent / ".gepa-heldout" / "git"
    storage.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(storage.parent, 0o700)
    os.chmod(storage, 0o700)
    return storage


@contextmanager
def safe_repository(root: Path) -> Iterator[SafeGit]:
    repository = Repository.discover(root)
    branch, oid = repository.head()
    object_format = ("sha256" if len(oid) == 64 else "sha1") if oid else None
    storage = _private_storage()
    # HEAD/index binding and every call using it form one serialized operation.
    # A new OS process always gets new storage, even after fork.
    with _repository_lock:
        # Never promote a reflector-writable temp cache into harness use when
        # held-out access is introduced later in the same process.
        owner = (os.getpid(), repository.common_dir, storage)
        object_format = object_format or _repository_formats.get(owner)
        key = (*owner, object_format or "unborn")
        git = _repositories.get(key)
        if git is None:
            private = Path(
                tempfile.mkdtemp(
                    prefix="gepa-git-", dir=storage if storage is not None else "/tmp"
                )
            )
            try:
                git = SafeGit(repository, private, root, object_format)
            except BaseException:
                shutil.rmtree(private)
                raise
            _repository_formats[owner] = git.object_format
            _repositories[(*owner, git.object_format)] = git
        git.bind(repository, root, branch, oid)
        yield git


def run_git(root: Path, *args: str, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    with safe_repository(root) as git:
        return git.run(*args, **kwargs)


@contextmanager
def _directory_fd(path: Path) -> Iterator[int]:
    """Open an absolute directory without following any symlink component."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def pin_commit(root: Path, ref: str, sha: str) -> None:
    """Publish a public retention ref as data, never invoke source Git hooks.

    Git's files backend recognizes loose refs during GC. Directory descriptors
    and no-follow opens confine the write even if source paths are replaced.
    Only the refs/gepa namespace is supported; no symbolic ref is followed.
    """
    if not re.fullmatch(
        r"refs/gepa/[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*", ref
    ) or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", sha):
        raise SafeGitError("Invalid retention ref.")
    sha = sha.lower()
    with safe_repository(root) as git:
        repository = git.repository
        marker = repository.root / ".git"
        marker_mode = marker.lstat().st_mode
        if stat.S_ISDIR(marker_mode):
            supported = repository.git_dir == repository.common_dir == marker
        elif stat.S_ISREG(marker_mode):
            # Accept only Git's conventional linked-worktree layout, including
            # its backlink. Arbitrary gitdir/commondir redirects are read-only.
            backlink = repository.git_dir / "gitdir"
            supported = (
                repository.git_dir.parent.name == "worktrees"
                and repository.git_dir.parent.parent == repository.common_dir
                and not backlink.is_symlink()
                and backlink.read_text().strip() == str(marker)
            )
        else:
            supported = False
        if not supported:
            raise SafeGitError("Unsupported retention repository layout.")
        if (
            len(sha) != (64 if git.object_format == "sha256" else 40)
            or git.run("cat-file", "-t", sha, check=True, capture_output=True).stdout
            != b"commit\n"
        ):
            raise SafeGitError("Retention requires a commit object.")
        with _directory_fd(git.repository.common_dir) as common:
            fd = os.dup(common)
            try:
                parts = ref.split("/")
                for part in parts[:-1]:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                    child = os.open(
                        part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                    )
                    os.close(fd)
                    fd = child
                name = parts[-1]
                try:
                    existing = os.stat(name, dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    if not stat.S_ISREG(existing.st_mode):
                        raise SafeGitError("Unsupported retention ref layout.")
                lock = name + ".lock"
                handle = os.open(
                    lock,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=fd,
                )
                try:
                    with os.fdopen(handle, "wb") as file:
                        file.write((sha + "\n").encode("ascii"))
                        file.flush()
                        os.fsync(file.fileno())
                    os.replace(lock, name, src_dir_fd=fd, dst_dir_fd=fd)
                except BaseException:
                    try:
                        os.unlink(lock, dir_fd=fd)
                    except FileNotFoundError:
                        pass
                    raise
                os.fsync(fd)
            finally:
                os.close(fd)


def unsafe_checkout_component(name: bytes) -> bool:
    """Git's HFS ignorable set and NTFS dotgit aliases (including ADS)."""
    ignored = {
        *range(0x200C, 0x2010),
        *range(0x202A, 0x202F),
        *range(0x206A, 0x2070),
        0xFEFF,
    }
    text = "".join(c for c in os.fsdecode(name) if ord(c) not in ignored).casefold()
    return any(
        part.split(":", 1)[0].rstrip(" .") in {".git", "git~1"}
        for part in text.split("\\")
    )


def refuse_heldout_git_mutations() -> None:
    """Lane lifecycle mutations need a separate, config-free mutation design."""
    import typer
    from .validation import heldout_dataset

    if heldout_dataset(required=False) is not None:
        raise typer.BadParameter(
            "Held-out lane Git mutations are unsupported; use a single-checkout run "
            "(--lanes 0). Lane start/select, worktree and branch changes are refused."
        )

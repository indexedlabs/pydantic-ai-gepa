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
import errno
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from typing import Any, Iterator


class SafeGitError(OSError):
    """Repository metadata cannot be safely interpreted."""


class GitExecutableError(SafeGitError):
    """No trusted, non-shim Git executable is installed."""


def _root_owned_path(path: Path, *, executable: bool = False) -> Path:
    resolved = path.resolve(strict=True)
    for directory in reversed(resolved.parents):
        info = directory.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise GitExecutableError("Unsafe Git installation.")
    info = resolved.stat()
    kind = stat.S_ISREG if executable else stat.S_ISDIR
    if info.st_uid != 0 or info.st_mode & 0o022 or not kind(info.st_mode):
        raise GitExecutableError("Unsafe Git installation.")
    if executable and (not info.st_mode & 0o111 or not os.access(resolved, os.X_OK)):
        raise GitExecutableError("Git is not executable.")
    return resolved


def _git_locations() -> list[Path]:
    if sys.platform != "darwin":
        return [Path("/usr/bin/git"), Path("/bin/git")]
    candidates = []
    selection = Path("/var/db/xcode_select_link")
    try:
        _root_owned_path(selection.parent)
        if selection.lstat().st_uid == 0:
            developer = Path(os.readlink(selection))
            if developer.is_absolute():
                candidates.append(developer / "usr/bin/git")
    except OSError:
        pass
    candidates.append(Path("/Library/Developer/CommandLineTools/usr/bin/git"))
    return candidates


@lru_cache(maxsize=1)
def _git_executable(pid: int) -> str:
    # The PID makes a fork resolve its own binary. Never consult xcrun, PATH,
    # DEVELOPER_DIR or a per-user tool lookup cache, even during discovery.
    for candidate in _git_locations():
        try:
            resolved = _root_owned_path(candidate, executable=True)
            if sys.platform == "darwin" and Path("/usr/bin/git") in (
                candidate,
                resolved,
            ):
                continue
            return str(resolved)
        except (OSError, RuntimeError):
            continue
    raise GitExecutableError(
        "No trusted Git executable is available; a root-owned installation "
        "with no group/other-writable path components is required."
    )


_METADATA_LIMIT = 1024 * 1024
_REF_LIMIT = 4096
_PACKED_REFS_LIMIT = 16 * 1024 * 1024


def _read_metadata_at(directory: int, name: str, limit: int) -> bytes:
    """Read bounded regular metadata through no-follow directory handles."""
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SafeGitError("Invalid Git metadata path.")
    parent = os.dup(directory)
    try:
        for part in parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
            )
            os.close(parent)
            parent = child
        fd = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        try:
            info = os.fstat(fd)
            # A hard link can alias a file the reflector can't read, such as
            # the held-out set, so only a single-link regular file is data.
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise OSError(errno.EINVAL, "Git metadata must be a regular file.")
            if info.st_size > limit:
                raise SafeGitError("Git metadata exceeds the size limit.")
            content = bytearray()
            while block := os.read(fd, min(65536, limit + 1 - len(content))):
                content.extend(block)
                if len(content) > limit:
                    raise SafeGitError("Git metadata exceeds the size limit.")
            return bytes(content)
        finally:
            os.close(fd)
    finally:
        os.close(parent)


def _read_metadata(
    directory: Path,
    name: str,
    *,
    limit: int = _METADATA_LIMIT,
    optional: bool = False,
    unsafe_empty: bool = False,
) -> bytes | None:
    try:
        with _directory_fd(directory) as fd:
            return _read_metadata_at(fd, name, limit)
    except FileNotFoundError:
        if optional:
            return None
        raise
    except OSError as exc:
        if unsafe_empty and exc.errno in {
            errno.ELOOP,
            errno.ENOTDIR,
            errno.EINVAL,
            errno.ENXIO,
            errno.EISDIR,
        }:
            return None
        raise SafeGitError("Cannot safely read Git metadata.") from None


# Keys emitted by lane_repositories._init: template-free Git init (including
# platform filesystem probes and SHA-256) followed by the two identity writes.
_GIT_BOOLEAN = frozenset({"true", "false", "yes", "no", "on", "off", "1", "0"})
_LANE_CONFIG: dict[str, frozenset[str] | None] = {
    "core.repositoryformatversion": frozenset({"0", "1"}),
    "core.bare": frozenset({"false", "no", "off", "0"}),
    "core.filemode": _GIT_BOOLEAN,
    "core.logallrefupdates": _GIT_BOOLEAN | {"always"},
    "core.ignorecase": _GIT_BOOLEAN,
    "core.precomposeunicode": _GIT_BOOLEAN,
    "extensions.objectformat": frozenset({"sha1", "sha256"}),
    "user.name": None,
    "user.email": None,
}
_LANE_REDIRECTS = (
    "commondir",
    "config.worktree",
    "info/attributes",
    "objects/info/alternates",
    "objects/info/http-alternates",
    "worktrees",
    "modules",
    "remotes",
    "branches",
)


def _check_lane_config(raw: bytes) -> None:
    """Accept only plain sections and explicit, unquoted ASCII values."""
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise SafeGitError("Lane .git/config is not plain ASCII.") from None
    if "\r" in text or "\0" in text:
        raise SafeGitError("Lane .git/config contains CR or NUL bytes.")
    section = None
    for line in text.split("\n"):
        line = line.strip(" \t")
        if not line or line.startswith(("#", ";")):
            continue
        if header := re.fullmatch(r"\[([A-Za-z]+)\]", line):
            section = header[1].lower()
            if section not in {"core", "extensions", "user"}:
                raise SafeGitError("Lane .git/config has an unsupported section.")
            continue
        entry = re.fullmatch(
            r"([A-Za-z][A-Za-z0-9-]*)[ \t]*=[ \t]*([\x20-\x7e]*)", line
        )
        if entry is None or section is None or any(c in entry[2] for c in '"\\#;'):
            raise SafeGitError("Lane .git/config is not a plain entry.")
        key, value = f"{section}.{entry[1].lower()}", entry[2].rstrip(" \t")
        allowed = _LANE_CONFIG.get(key, frozenset())
        if allowed is not None and value.lower() not in allowed:
            raise SafeGitError(f"Lane .git/config has unsupported key or value: {key}.")


def refuse_executable_lane_git(project: Path) -> Path:
    """Check a project's repository before an unsandboxed reflector launch.

    Read files only, with no-follow metadata handles. Gitfiles, redirects,
    executable configuration and nested repositories are never accepted.
    """
    start = project.absolute()
    for root in (start, *start.parents):
        with _directory_fd(root) as directory:
            try:
                marker = os.stat(".git", dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(marker.st_mode):
                raise SafeGitError(
                    "Lane .git must be a directory, not a gitfile or link."
                )
        break
    else:
        raise SafeGitError("No lane Git repository found.")
    metadata = root / ".git"
    _check_lane_config(_read_metadata(metadata, "config") or b"")
    with _directory_fd(metadata) as directory:
        for name in ("hooks", "info", "objects", "objects/info"):
            try:
                info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(info.st_mode):
                raise SafeGitError(f"Lane .git/{name} must be a directory, not a link.")
        for name in _LANE_REDIRECTS:
            try:
                os.stat(name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise SafeGitError(f"Lane has unsupported Git metadata: .git/{name}.")
    hooks = metadata / "hooks"
    if hooks.exists():
        with _directory_fd(hooks) as directory:
            for name in os.listdir(directory):
                if not name.endswith(".sample"):
                    raise SafeGitError("Lane has an active Git hook.")
                _read_metadata_at(directory, name, _METADATA_LIMIT)

    def unreadable(error: OSError) -> None:
        raise error

    for directory, directories, files in os.walk(root, onerror=unreadable):
        if Path(directory) == root:
            directories.remove(".git")
        if ".git" in directories or ".git" in files:
            raise SafeGitError("Lane contains a nested Git repository.")
    return root


@dataclass(frozen=True)
class Repository:
    root: Path
    git_dir: Path
    common_dir: Path

    @classmethod
    def discover(cls, start: Path) -> Repository:
        start = start.resolve()
        for root in (start, *start.parents):
            with _directory_fd(root) as fd:
                try:
                    marker = os.stat(".git", dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                value = (
                    None
                    if stat.S_ISDIR(marker.st_mode)
                    else os.fsdecode(_read_metadata_at(fd, ".git", _REF_LIMIT)).strip()
                )
            if value is None:
                git_dir = root / ".git"
            else:
                if not value.startswith("gitdir: ") or "\n" in value or "\0" in value:
                    raise SafeGitError("Invalid Git directory pointer.")
                git_dir = Path(os.path.abspath(root / value[8:]))
            common_data = _read_metadata(
                git_dir, "commondir", limit=_REF_LIMIT, optional=True
            )
            common = git_dir
            if common_data is not None:
                pointer = os.fsdecode(common_data).strip()
                if not pointer or "\n" in pointer or "\0" in pointer:
                    raise SafeGitError("Invalid Git common directory pointer.")
                common = Path(os.path.abspath(git_dir / pointer))
            with _directory_fd(common / "objects"):
                pass
            if (common / "reftable").exists():
                raise SafeGitError("Reftable repositories are not supported.")
            return cls(root, git_dir, common)
        raise FileNotFoundError("not a git repository")

    def head(self) -> tuple[str | None, str | None]:
        value = os.fsdecode(
            _read_metadata(self.git_dir, "HEAD", limit=_REF_LIMIT) or b""
        ).strip()
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
            loose = _read_metadata(
                self.common_dir, ref, limit=_REF_LIMIT, optional=True
            )
            if loose is not None:
                value = os.fsdecode(loose).strip()
                continue
            packed = _read_metadata(
                self.common_dir, "packed-refs", limit=_PACKED_REFS_LIMIT, optional=True
            )
            for line in os.fsdecode(packed or b"").splitlines():
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
        self.command = [_git_executable(os.getpid()), "--no-pager"]
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
                    "-",
                    "--get",
                    "extensions.objectformat",
                ],
                input=_read_metadata(repository.common_dir, "config"),
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
        _write_changed(
            self.directory / "info/exclude",
            _read_metadata(
                repository.common_dir, "info/exclude", optional=True, unsafe_empty=True
            )
            or b"",
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
            supported = (
                repository.git_dir.parent.name == "worktrees"
                and repository.git_dir.parent.parent == repository.common_dir
                and os.fsdecode(
                    _read_metadata(repository.git_dir, "gitdir", limit=_REF_LIMIT)
                    or b""
                ).strip()
                == str(marker)
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

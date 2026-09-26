"""Authoritative held-out run records; GEPA_DIR files are disposable views.

Only a process with the harness dataset can open a record. Public writes are
never imported, including writes made by an older harness (rollback). Records
are initialized only at run creation; missing legacy records fail closed.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import hashlib
import errno
import os
from pathlib import Path
import stat
import tempfile
from threading import RLock
from typing import Any, Callable, Iterator, TypeVar
from uuid import uuid4

from ..vector_acceptance import (
    VectorRecord,
    VectorRecordStore as PublicVectorRecordStore,
)

import typer

from .validation import _pin_path, heldout_dataset, public_echo, validation_dataset_path

_active: ContextVar[Record | None] = ContextVar("harness_record", default=None)
_locks: ContextVar[frozenset[str]] = ContextVar(
    "harness_record_locks", default=frozenset()
)
_mutex = RLock()


@contextmanager
def session() -> Iterator[None]:
    token = _active.set(None)
    try:
        yield
    finally:
        _active.reset(token)


def _atomic_text(
    path: Path, content: str, *, private: bool = False, root: Path | None = None
) -> None:
    if not private:
        assert root is not None
        with SafeDir.open(root, path.parent, create=True) as directory:
            directory.write_text(path.name, content)
        return
    # Only harness-owned private storage uses pathnames.
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _unavailable() -> typer.BadParameter:
    return typer.BadParameter(
        "Harness private record missing or unreadable; refusing GEPA_DIR state. "
        "Restore the original private record or start a new run (legacy runs cannot be adopted)."
    )


class _PlantedLeaf(typer.BadParameter):
    def __init__(self):
        super().__init__(_unavailable().message)


def _public_path(root: Path, path: Path) -> Path:
    workspace = root.resolve()
    # Normalize a checkout alias (including macOS /var) without resolving any
    # reflector-controlled component inside GEPA_DIR.
    lexical_root = Path(os.path.abspath(root))
    target = Path(os.path.abspath(path))
    for ancestor in (lexical_root, *reversed(target.parents)):
        if target.is_relative_to(ancestor) and ancestor.resolve() == workspace:
            return workspace / target.relative_to(ancestor)
    return target


class SafeDir:
    """Public I/O stays relative to an opened, non-link directory inode.

    Trust the workspace root and its ancestors; for an external GEPA_DIR,
    trust its parent and ancestors (including system /tmp and /var aliases).
    Every component inside the workspace, or from the external GEPA_DIR's
    final component onward, is opened with O_NOFOLLOW.
    """

    def __init__(self, fd: int):
        self.fd = fd

    @classmethod
    @contextmanager
    def open(
        cls, root: Path, path: Path, *, create: bool = False, base: Path | None = None
    ) -> Iterator[SafeDir]:
        from .layout import gepa_dir

        workspace = root.resolve()
        base = _public_path(root, base or gepa_dir(workspace))
        target = _public_path(root, path)
        if not target.is_relative_to(base):
            raise _unavailable()
        if base.is_relative_to(workspace):
            anchor = workspace
            parts = target.relative_to(workspace).parts
        else:
            anchor = base.parent.resolve()
            parts = (base.name, *target.relative_to(base).parts)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        fd = None
        try:
            fd = os.open(anchor, flags)
            for part in parts:
                if create:
                    try:
                        os.mkdir(part, 0o755, dir_fd=fd)
                    except FileExistsError:
                        pass
                child = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = child
        except FileNotFoundError:
            if fd is not None:
                os.close(fd)
            raise
        except OSError:
            if fd is not None:
                os.close(fd)
            raise _unavailable() from None
        try:
            yield cls(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _name(name: str) -> None:
        if not name or name in {".", ".."} or "/" in name or "\0" in name:
            raise _unavailable()

    def check_leaf(self, name: str) -> bool:
        self._name(name)
        try:
            info = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        except OSError:
            raise _unavailable() from None
        if stat.S_ISLNK(info.st_mode):
            raise _unavailable()
        return True

    @contextmanager
    def file(self, name: str, flags: int = os.O_RDONLY) -> Iterator[Any]:
        self._name(name)
        if self.planted_leaf(name):
            raise _PlantedLeaf()
        try:
            fd = os.open(
                name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=self.fd
            )
        except FileNotFoundError:
            raise
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise _PlantedLeaf() from None
            raise _unavailable() from None
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
                raise _PlantedLeaf()
            handle = os.fdopen(fd, "r+" if flags & os.O_RDWR else "r", encoding="utf-8")
        except BaseException:
            os.close(fd)
            raise
        with handle:
            yield handle

    def read_text(self, name: str, *, errors: str = "strict") -> str:
        try:
            with self.file(name) as handle:
                handle.reconfigure(errors=errors)
                return handle.read()
        except FileNotFoundError:
            raise
        except OSError:
            raise _unavailable() from None

    def append_text(self, name: str, content: str) -> None:
        # Never modify an existing inode: a public hard link must not turn an
        # append into a write to its private/outside source.
        try:
            previous = self.read_text(name)
        except (FileNotFoundError, _PlantedLeaf):
            previous = ""
        self.write_text(name, previous + content)

    def discard_leaf(self, name: str) -> None:
        self._name(name)
        try:
            os.unlink(name, dir_fd=self.fd)
        except FileNotFoundError:
            pass
        except OSError as exc:
            # Empty planted leaf directories can be removed without walking
            # them. Nonempty directories fail closed; never recursively repair.
            if exc.errno not in {errno.EISDIR, errno.EPERM}:
                raise _unavailable() from None
            try:
                os.rmdir(name, dir_fd=self.fd)
            except OSError:
                raise _unavailable() from None

    def planted_leaf(self, name: str) -> bool:
        self._name(name)
        try:
            info = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        except OSError:
            raise _unavailable() from None
        return not stat.S_ISREG(info.st_mode) or info.st_nlink > 1

    def repair_leaf(self, name: str) -> None:
        if self.planted_leaf(name):
            self.discard_leaf(name)
            public_echo(
                "Harness record: GEPA_DIR view was changed outside the harness; "
                "replaced the planted leaf.",
                err=True,
            )

    def names(self) -> list[str]:
        try:
            return os.listdir(self.fd)
        except OSError:
            raise _unavailable() from None

    def unlink(self, name: str) -> None:
        self._name(name)
        try:
            os.unlink(name, dir_fd=self.fd)
        except FileNotFoundError:
            pass
        except OSError:
            raise _unavailable() from None

    def write_text(self, name: str, content: str, *, exclusive: bool = False) -> None:
        self._name(name)
        self.repair_leaf(name)
        temporary = f".{uuid4().hex}.tmp"
        try:
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=self.fd,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                if exclusive:
                    os.link(
                        temporary,
                        name,
                        src_dir_fd=self.fd,
                        dst_dir_fd=self.fd,
                        follow_symlinks=False,
                    )
                else:
                    os.replace(temporary, name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
                os.fsync(self.fd)
            finally:
                self.unlink(temporary)
        except FileExistsError:
            if exclusive:
                raise
            raise _unavailable() from None
        except OSError:
            raise _unavailable() from None


def check_view_path(root: Path, path: Path, *, base: Path | None = None) -> None:
    """Early refusal only; subsequent I/O must still use SafeDir, never a path."""
    from .layout import gepa_dir

    base = base or gepa_dir(root)
    try:
        if _public_path(root, path) == _public_path(root, base):
            with SafeDir.open(root, path, base=base):
                pass
        else:
            with SafeDir.open(root, path.parent, base=base) as directory:
                directory.check_leaf(path.name)
    except FileNotFoundError:
        pass  # No operation follows from this probe; SafeDir reopens safely.


@contextmanager
def view_file(path: Path, *, root: Path | None = None) -> Iterator[Any]:
    """Open a cooperative public lock without following links or modifying it."""
    from .layout import repo_root

    active = _active.get()
    workspace = root or (active.root if active else repo_root())
    with SafeDir.open(workspace, path.parent, create=True) as directory:
        with directory.file(path.name, os.O_RDWR | os.O_CREAT) as handle:
            yield handle


def _index_path(dataset: str, root: Path, run_id: str) -> Path:
    # Workspace identity is canonical; GEPA_DIR is deliberately NOT resolved.
    # Moving or redirecting GEPA_DIR cannot change which private index we read.
    key = hashlib.sha256(f"index\0{root.resolve()}\0{run_id}".encode()).hexdigest()
    return validation_dataset_path(
        str(Path(dataset).parent / ".gepa-heldout" / f"{key}.index.json"),
        project_root=root,
        allow_missing=True,
        check_history=False,
    )


def register_run(root: Path, run_id: str, pin: Path) -> None:
    """Register immutable run identity at pin creation, before public state."""
    from .layout import gepa_dir, run_dir

    check_view_path(root, run_dir(run_id, root))
    dataset = heldout_dataset()
    assert dataset is not None
    index = _index_path(dataset, root, run_id)
    entry = {
        "workspace": str(root.resolve()),
        "gepa_dir": os.path.abspath(gepa_dir(root.resolve())),
        "run_id": run_id,
        "pin": pin.name,
    }
    # Exclusive creation prevents a second GEPA_DIR from taking over the same
    # workspace/run identity. A partially created index fails closed.
    fd = os.open(index, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(entry, handle)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(index.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _indexed_record_path(dataset: str, root: Path, run_id: str) -> Path | None:
    from .layout import gepa_dir, run_dir

    check_view_path(root, run_dir(run_id, root))
    index = _index_path(dataset, root, run_id)
    try:
        entry = json.loads(index.read_text(encoding="utf-8"))
    except FileNotFoundError:
        try:
            with SafeDir.open(root, run_dir(run_id, root)):
                pass
        except FileNotFoundError:
            return None
        raise _unavailable() from None
    except (OSError, ValueError):
        raise _unavailable() from None
    try:
        if (
            entry["workspace"] != str(root.resolve())
            or entry["run_id"] != run_id
            or entry["gepa_dir"] != os.path.abspath(gepa_dir(root.resolve()))
            or entry["pin"] != _pin_path(dataset, root, run_id).name
        ):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        raise _unavailable() from None
    return (index.parent / entry["pin"]).with_suffix(".record.json")


class Record:
    def __init__(self, root: Path, run_id: str, path: Path):
        from .layout import run_dir

        self.root, self.run_id, self.path = root, run_id, path
        self.dataset = heldout_dataset()
        self.directory = run_dir(run_id, root)

    @contextmanager
    def locked(self) -> Iterator[None]:
        from .reflector import run_lock

        with run_lock(self.run_id, self.root, wait=True), _mutex:
            yield

    def load(self) -> dict[str, Any]:
        check_view_path(self.root, self.directory)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                data["version"] != 1
                or data["run_id"] != self.run_id
                or not isinstance(data["files"], dict)
                or not all(isinstance(v, str) for v in data["files"].values())
            ):
                raise ValueError
            return data
        except (OSError, ValueError, KeyError, TypeError):
            raise _unavailable() from None

    def _restore(self, key: str, content: str | None) -> None:
        if key.startswith("@"):
            return  # Private bookkeeping has no whole-file public counterpart.
        path = self.directory / key
        with SafeDir.open(self.root, path.parent, create=True) as directory:
            self._restore_at(directory, path.name, content)

    def _restore_at(self, directory: SafeDir, name: str, content: str | None) -> None:
        try:
            actual = directory.read_text(name)
        except _PlantedLeaf:
            actual = None
            self._refusal()
            directory.discard_leaf(name)
            if content is not None:
                directory.write_text(name, content)
            return
        except (FileNotFoundError, UnicodeError):
            actual = None
        if actual == content and (
            content is not None or not directory.check_leaf(name)
        ):
            return
        self._refusal()
        if content is None:
            directory.unlink(name)
        else:
            directory.write_text(name, content)

    def _refusal(self) -> None:
        public_echo(
            f"Harness record: GEPA_DIR state for run {self.run_id} was changed "
            "outside the harness; restored from the harness record.",
            err=True,
        )

    def read(self, key: str) -> str | None:
        with self.locked():
            content = self.load()["files"].get(key)
            self._restore(key, content)
            return content

    def write(self, key: str, content: str, *, append: bool = False) -> None:
        with self.locked():
            if not key.startswith("@"):
                with SafeDir.open(
                    self.root, (self.directory / key).parent, create=True
                ):
                    pass
            data = self.load()
            if append:
                content = data["files"].get(key, "") + content
            data["files"][key] = content
            # Commit authority before publishing the view. An interrupted view
            # write is repaired on the next read; it never authorizes a retry.
            _atomic_text(self.path, json.dumps(data), private=True)
            if not key.startswith("@"):
                _atomic_text(self.directory / key, content, root=self.root)

    def restore_views(self) -> None:
        with self.locked():
            check_view_path(self.root, self.directory / "results")
            files = self.load()["files"]
            keys = {key for key in files if not key.startswith("@")}
            with SafeDir.open(
                self.root, self.directory / "results", create=True
            ) as results:
                for name in results.names():
                    if name.endswith(".json"):
                        self._restore_at(results, name, files.get(f"results/{name}"))
            for key in sorted(keys):
                self._restore(key, files.get(key))


def for_run(run_id: str, root: Path | None = None) -> Record | None:
    from .layout import repo_root, run_dir

    dataset = heldout_dataset(required=False)
    if dataset is None:
        return None
    workspace = (root or repo_root()).resolve()
    check_view_path(workspace, run_dir(run_id, workspace))
    active = _active.get()
    if (
        active
        and active.root == workspace
        and active.run_id == run_id
        and active.dataset == dataset
    ):
        return active
    path = _indexed_record_path(dataset, workspace, run_id)
    if path is None:
        return None
    record = Record(workspace, run_id, path)
    record.load()
    _active.set(record)
    return record


def initialize(root: Path, run_id: str) -> None:
    from .layout import config_path

    dataset = heldout_dataset()
    assert dataset is not None
    path = _indexed_record_path(dataset, root, run_id)
    if path is None:
        raise _unavailable()
    record = Record(root.resolve(), run_id, path)
    with record.locked():
        if path.exists():
            raise typer.BadParameter("Harness record already exists; start a new run.")
        _atomic_text(
            path,
            json.dumps(
                {
                    "version": 1,
                    "run_id": run_id,
                    "files": {"@config": _config_view(root, config_path(root))},
                }
            ),
            private=True,
        )
    _active.set(record)


def _view(path: Path, root: Path | None = None) -> tuple[Record, str] | None:
    from .layout import repo_root, runs_dir

    if not heldout_dataset(required=False):
        return None
    active = _active.get()
    workspace = root or repo_root()
    if (
        root is None
        and active is not None
        and runs_dir(workspace) == runs_dir(active.root)
    ):
        # An absolute GEPA_DIR still belongs to the primary workspace while
        # candidate evaluation temporarily changes cwd to a lane checkout.
        workspace = active.root
    try:
        parts = (
            _public_path(workspace, path)
            .relative_to(runs_dir(workspace.resolve()))
            .parts
        )
    except ValueError:
        if root is not None or active is None:
            return None
        workspace = active.root
        try:
            parts = (
                _public_path(workspace, path)
                .relative_to(runs_dir(workspace.resolve()))
                .parts
            )
        except ValueError:
            return None
    if len(parts) < 2:
        return None
    key = "/".join(parts[1:])
    if parts[1] not in {
        "state.json",
        "pareto.jsonl",
        "vectors.jsonl",
        "spend.jsonl",
        "spend-reservations.json",
        "validation-spend-registered",
        "validation-spend-owners.json",
        "minibatches",
        "results",
        "final_report.md",
    }:
        return None
    record = for_run(parts[0], workspace)
    return (record, key) if record else None


def _public_workspace(path: Path, root: Path | None = None) -> Path | None:
    from .layout import gepa_dir, repo_root

    if not heldout_dataset(required=False):
        return None
    active = _active.get()
    workspace = root or (active.root if active else repo_root())
    if _public_path(workspace, path).is_relative_to(
        _public_path(workspace, gepa_dir(workspace))
    ):
        return workspace
    return None


def list_paths(path: Path, *, root: Path | None = None) -> list[Path]:
    workspace = _public_workspace(path, root)
    if workspace is None:
        return list(path.iterdir()) if path.is_dir() else []
    try:
        with SafeDir.open(workspace, path) as directory:
            return [path / name for name in directory.names()]
    except FileNotFoundError:
        return []


def read_text(path: Path, *, root: Path | None = None, errors: str = "strict") -> str:
    view = _view(path, root)
    if view is None:
        workspace = _public_workspace(path, root)
        if workspace is not None:
            with SafeDir.open(workspace, path.parent) as directory:
                return directory.read_text(path.name, errors=errors)
        return path.read_text(encoding="utf-8", errors=errors)
    content = view[0].read(view[1])
    if content is None:
        raise FileNotFoundError(f"No harness-written view at {path.name}")
    return content


def read_bytes(path: Path, *, root: Path | None = None) -> bytes:
    view = _view(path, root)
    if view is not None:
        return read_text(path, root=root).encode("utf-8")
    workspace = _public_workspace(path, root)
    if workspace is None:
        return path.read_bytes()
    with SafeDir.open(workspace, path.parent) as directory:
        with directory.file(path.name) as handle:
            return handle.buffer.read()


def exists(path: Path, *, root: Path | None = None) -> bool:
    view = _view(path, root)
    if view:
        return view[0].read(view[1]) is not None
    workspace = _public_workspace(path, root)
    if workspace is not None:
        try:
            with SafeDir.open(workspace, path.parent) as directory:
                return directory.check_leaf(path.name)
        except FileNotFoundError:
            return False
    return path.exists()


def write_text(
    path: Path, content: str, *, root: Path | None = None, append: bool = False
) -> bool:
    """Publish through the record; return False for an ordinary public file."""
    view = _view(path, root)
    if view is None:
        workspace = _public_workspace(path, root)
        if workspace is not None:
            with SafeDir.open(workspace, path.parent, create=True) as directory:
                if append:
                    directory.append_text(path.name, content)
                else:
                    directory.write_text(path.name, content)
            return True
        return False
    view[0].write(view[1], content, append=append)
    return True


def _config_view(root: Path, path: Path) -> str:
    with SafeDir.open(root, path.parent, base=path.parent) as directory:
        return directory.read_text(path.name)


def config_text(path: Path) -> str | None:
    from .layout import config_path, repo_root

    if not heldout_dataset(required=False):
        return None
    record = _active.get()
    root = repo_root(path.parent)
    check_view_path(root, path, base=path.parent)
    if record and _public_path(record.root, path) == config_path(record.root):
        content = record.read("@config")
        if content is None:
            raise _unavailable()
        return content
    # Startup/candidate config reads also stay bound to the opened directory.
    return _config_view(root, path)


class VectorRecordStore(PublicVectorRecordStore):
    """Use private training vectors when the harness rebaselines a lane run."""

    def append(self, record: VectorRecord) -> None:
        if not write_text(
            self.path, json.dumps(record.to_dict(), sort_keys=True) + "\n", append=True
        ):
            super().append(record)

    def records(self) -> list[VectorRecord]:
        if not exists(self.path):
            return []
        return [
            VectorRecord.from_dict(json.loads(line))
            for line in read_text(self.path).splitlines()
            if line.strip()
        ]


def private_lock_path(root: Path | None, run_id: str) -> Path | None:
    """A writable public lock inode cannot serialize authoritative updates."""
    from .layout import repo_root

    dataset = heldout_dataset(required=False)
    if dataset is None:
        return None
    workspace = (root or repo_root()).resolve()
    active = _active.get()
    if (
        active
        and active.root == workspace
        and active.run_id == run_id
        and active.dataset == dataset
    ):
        return active.path.with_suffix(".lock")
    path = _indexed_record_path(dataset, workspace, run_id)
    return path.with_suffix(".lock") if path is not None else None


_ResultT = TypeVar("_ResultT")


def serialized_eval(evaluate: Callable[..., _ResultT]) -> Callable[..., _ResultT]:
    """Keep a held-out run's budget admission and paid publication under its lock."""
    from functools import wraps

    @wraps(evaluate)
    def wrapped(**kwargs: Any) -> _ResultT:
        from .layout import latest_run_id, repo_root
        from .reflector import run_lock

        root = kwargs.get("workspace_root") or repo_root()
        run_id = kwargs.get("run_id") or latest_run_id(root)
        record = for_run(run_id, root) if run_id else None
        if record is None:
            return evaluate(**kwargs)
        with run_lock(record.run_id, record.root, wait=True):
            return evaluate(**kwargs)

    return wrapped


@contextmanager
def before_spend_lock(path: Path) -> Iterator[None]:
    """Keep run -> spend -> record mutex ordering for concurrent status/eval."""
    from .reflector import run_lock

    view = _view(path.parent / "state.json")
    if view is None:
        yield
    else:
        record = view[0]
        with run_lock(record.run_id, record.root, wait=True):
            yield

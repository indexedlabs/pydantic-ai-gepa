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
import os
from pathlib import Path
import stat
import tempfile
from threading import RLock
from typing import Any, Callable, Iterator, TypeVar

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
        check_view_path(root, path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    if private:
        path.parent.chmod(0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if not private:
            assert root is not None
            check_view_path(root, path)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        safe_cleanup = True
        if not private:
            assert root is not None
            try:
                check_view_path(root, Path(temporary))
            except typer.BadParameter:
                safe_cleanup = False
        if safe_cleanup and os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _unavailable() -> typer.BadParameter:
    return typer.BadParameter(
        "Harness private record missing or unreadable; refusing GEPA_DIR state. "
        "Restore the original private record or start a new run (legacy runs cannot be adopted)."
    )


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


def check_view_path(root: Path, path: Path, *, base: Path | None = None) -> None:
    """Refuse redirected public paths before reading, listing, or changing them."""
    from .layout import gepa_dir

    workspace = root.resolve()
    base = _public_path(root, base or gepa_dir(workspace))
    target = _public_path(root, path)
    if not target.is_relative_to(base):
        raise _unavailable()
    anchor = workspace if base.is_relative_to(workspace) else Path(base.anchor)
    current = anchor
    for component in target.relative_to(anchor).parts:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError:
            raise _unavailable() from None
        if stat.S_ISLNK(mode) or (current != target and not stat.S_ISDIR(mode)):
            raise _unavailable()


def _index_path(dataset: str, root: Path, run_id: str) -> Path:
    # Workspace identity is canonical; GEPA_DIR is deliberately NOT resolved.
    # Moving or redirecting GEPA_DIR cannot change which private index we read.
    key = hashlib.sha256(f"index\0{root.resolve()}\0{run_id}".encode()).hexdigest()
    return validation_dataset_path(
        str(Path(dataset).parent / ".gepa-heldout" / f"{key}.index.json"),
        project_root=root,
        allow_missing=True,
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


def _indexed_record_path(dataset: str, root: Path, run_id: str) -> Path:
    from .layout import gepa_dir, run_dir

    check_view_path(root, run_dir(run_id, root))
    index = _index_path(dataset, root, run_id)
    try:
        entry = json.loads(index.read_text(encoding="utf-8"))
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
        check_view_path(self.root, path)
        try:
            actual = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            actual = None
        if (
            actual == content
            and not path.is_symlink()
            and (content is not None or not path.exists())
        ):
            return
        public_echo(
            f"Harness record: GEPA_DIR state for run {self.run_id} was changed "
            "outside the harness; restored from the harness record.",
            err=True,
        )
        if content is None:
            check_view_path(self.root, path)
            path.unlink(missing_ok=True)
        else:
            check_view_path(self.root, path)
            _atomic_text(path, content, root=self.root)

    def read(self, key: str) -> str | None:
        with self.locked():
            content = self.load()["files"].get(key)
            self._restore(key, content)
            return content

    def write(self, key: str, content: str, *, append: bool = False) -> None:
        with self.locked():
            if not key.startswith("@"):
                check_view_path(self.root, self.directory / key)
            data = self.load()
            if append:
                content = data["files"].get(key, "") + content
            data["files"][key] = content
            # Commit authority before publishing the view. An interrupted view
            # write is repaired on the next read; it never authorizes a retry.
            _atomic_text(self.path, json.dumps(data), private=True)
            if not key.startswith("@"):
                check_view_path(self.root, self.directory / key)
                _atomic_text(self.directory / key, content, root=self.root)

    def restore_views(self) -> None:
        with self.locked():
            check_view_path(self.root, self.directory / "results")
            files = self.load()["files"]
            keys = {key for key in files if not key.startswith("@")}
            keys.update(
                f"results/{path.name}"
                for path in (self.directory / "results").glob("*.json")
            )
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
    record = Record(workspace, run_id, path)
    record.load()
    _active.set(record)
    return record


def initialize(root: Path, run_id: str) -> None:
    from .layout import config_path

    dataset = heldout_dataset()
    assert dataset is not None
    path = _indexed_record_path(dataset, root, run_id)
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
                    "files": {"@config": config_path(root).read_text(encoding="utf-8")},
                }
            ),
            private=True,
        )
    _active.set(record)


def _view(path: Path, root: Path | None = None) -> tuple[Record, str] | None:
    from .layout import repo_root, runs_dir

    active = _active.get() if heldout_dataset(required=False) else None
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


def read_text(path: Path, *, root: Path | None = None) -> str:
    view = _view(path, root)
    if view is None:
        return path.read_text(encoding="utf-8")
    content = view[0].read(view[1])
    if content is None:
        raise FileNotFoundError(f"No harness-written view at {path.name}")
    return content


def exists(path: Path, *, root: Path | None = None) -> bool:
    view = _view(path, root)
    return view[0].read(view[1]) is not None if view else path.exists()


def write_text(
    path: Path, content: str, *, root: Path | None = None, append: bool = False
) -> bool:
    """Publish through the record; return False for an ordinary public file."""
    view = _view(path, root)
    if view is None:
        return False
    view[0].write(view[1], content, append=append)
    return True


def config_text(path: Path) -> str | None:
    from .layout import config_path, repo_root

    if not heldout_dataset(required=False):
        return None
    # Candidate config validation may read a checkout-local copy outside the
    # primary absolute GEPA_DIR. Check its ancestors without treating that
    # untrusted candidate copy as the harness's pinned configuration.
    check_view_path(repo_root(path.parent), path, base=path.parent)
    record = _active.get()
    if (
        not record
        or not heldout_dataset(required=False)
        or path.resolve() != config_path(record.root).resolve()
    ):
        return None
    content = record.read("@config")
    if content is None:
        raise _unavailable()
    # Configuration can be tracked candidate material: use the pinned value
    # without rewriting the reflector's commit or working tree.
    return content


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
    return _indexed_record_path(dataset, workspace, run_id).with_suffix(".lock")


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

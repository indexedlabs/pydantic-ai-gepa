"""Authoritative held-out run records; GEPA_DIR files are disposable views.

Only a process with the harness dataset can open a record. Public writes are
never imported, including writes made by an older harness (rollback). Records
are initialized only at run creation; missing legacy records fail closed.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import tempfile
from threading import RLock
from typing import Any, Callable, Iterator, TypeVar

from ..vector_acceptance import (
    VectorRecord,
    VectorRecordStore as PublicVectorRecordStore,
)

import typer

from .validation import _pin_path, heldout_dataset, public_echo

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


def _atomic_text(path: Path, content: str, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    if private:
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
            path.unlink(missing_ok=True)
        else:
            _atomic_text(path, content)

    def read(self, key: str) -> str | None:
        with self.locked():
            content = self.load()["files"].get(key)
            self._restore(key, content)
            return content

    def write(self, key: str, content: str, *, append: bool = False) -> None:
        with self.locked():
            data = self.load()
            if append:
                content = data["files"].get(key, "") + content
            data["files"][key] = content
            # Commit authority before publishing the view. An interrupted view
            # write is repaired on the next read; it never authorizes a retry.
            _atomic_text(self.path, json.dumps(data), private=True)
            if not key.startswith("@"):
                _atomic_text(self.directory / key, content)

    def restore_views(self) -> None:
        with self.locked():
            files = self.load()["files"]
            keys = {key for key in files if not key.startswith("@")}
            keys.update(
                f"results/{path.name}"
                for path in (self.directory / "results").glob("*.json")
            )
            for key in sorted(keys):
                self._restore(key, files.get(key))


def for_run(run_id: str, root: Path | None = None) -> Record | None:
    from .layout import repo_root, run_state_path

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
        return active
    pin = _pin_path(dataset, workspace, run_id)
    path = pin.with_suffix(".record.json")
    if not pin.exists() and not path.exists():
        # Non-held-out/ad-hoc runs retain their original behavior. The private
        # pin, not a reflector-controlled flag, identifies established runs.
        try:
            required = json.loads(run_state_path(run_id, workspace).read_text()).get(
                "heldout_required", False
            )
        except (OSError, ValueError, AttributeError):
            required = False
        if required:
            raise _unavailable()
        return None
    record = Record(workspace, run_id, path)
    record.load()
    _active.set(record)
    return record


def initialize(root: Path, run_id: str) -> None:
    from .layout import config_path

    dataset = heldout_dataset()
    assert dataset is not None
    path = _pin_path(dataset, root, run_id).with_suffix(".record.json")
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
    try:
        parts = Path(os.path.abspath(path)).relative_to(runs_dir(workspace)).parts
    except ValueError:
        if root is not None or active is None:
            return None
        workspace = active.root
        try:
            parts = Path(os.path.abspath(path)).relative_to(runs_dir(workspace)).parts
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
    from .layout import config_path

    record = _active.get()
    if (
        not record
        or not heldout_dataset(required=False)
        or path != config_path(record.root)
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
    pin = _pin_path(dataset, workspace, run_id)
    if pin.exists():
        return pin.with_suffix(".record.lock")
    return None


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

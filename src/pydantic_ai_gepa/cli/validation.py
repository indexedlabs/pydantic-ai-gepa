"""Keep held-out datasets outside every reflector checkout and its Git objects."""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable, ParamSpec, TypeVar

import typer


def validation_dataset_path(
    configured_path: str,
    *,
    project_root: Path,
    candidate_root: Path | None = None,
    allow_missing: bool = False,
) -> Path:
    """Resolve a harness-owned dataset, refusing checkout and historical copies."""
    lexical_path = Path(os.path.abspath(project_root / configured_path))
    path = lexical_path.resolve()
    fix = (
        "Keep held-out data outside every checkout and GEPA_DIR; set "
        "GEPA_HELDOUT_DATASET only in the harness environment. "
        "Start from Git history that never contained the held-out data."
    )

    def run_git(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        try:
            return subprocess.run(command, env={**os.environ, "LC_ALL": "C"}, **kwargs)
        except OSError as exc:
            raise typer.BadParameter(
                "Cannot verify validation dataset isolation. " + fix
            ) from exc

    from .layout import gepa_dir

    roots = {project_root.resolve(), (candidate_root or project_root).resolve()}
    repositories: set[Path] = set()
    for root in tuple(roots):
        result = run_git(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            repository = Path(result.stdout.strip()).resolve()
            roots.add(repository)
            repositories.add(repository)
        elif "not a git repository" not in result.stderr:
            raise typer.BadParameter(
                "Cannot verify validation dataset isolation. " + fix
            )
    roots.add(gepa_dir(project_root).resolve())
    if any(
        path.is_relative_to(root) or lexical_path.is_relative_to(root) for root in roots
    ):
        raise typer.BadParameter(
            "Held-out validation must be outside the reflector checkout. " + fix
        )
    if allow_missing and not path.exists():
        return path
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise typer.BadParameter(
            "Cannot read held-out validation dataset. " + fix
        ) from exc
    for repository in repositories:
        # Checking the object database also catches staged files, deleted files,
        # renamed files, other branches and reflogs. Moving a file is not enough.
        hashed = run_git(
            ["git", "-C", str(repository), "hash-object", "--stdin"],
            input=contents,
            capture_output=True,
        )
        if hashed.returncode:
            raise typer.BadParameter(
                "Cannot verify validation dataset Git history. " + fix
            )
        found = run_git(
            [
                "git",
                "-C",
                str(repository),
                "cat-file",
                "--batch-check",
            ],
            input=hashed.stdout,
            capture_output=True,
        )
        if found.returncode:
            raise typer.BadParameter(
                "Cannot verify validation dataset Git history. " + fix
            )
        if found.stdout.strip() != hashed.stdout.strip() + b" missing":
            raise typer.BadParameter(
                "Held-out validation is recoverable from Git objects. " + fix
            )
    return path


def validation_evidence_path(dataset: str, *, project_root: Path, run_id: str) -> Path:
    """Keep paired evidence beside the harness-owned dataset, outside checkouts."""
    configured = heldout_dataset()
    if Path(dataset).resolve() != Path(str(configured)).resolve():
        raise typer.BadParameter(
            "Held-out evidence requires the harness's pinned dataset."
        )
    dataset_path = validation_dataset_path(dataset, project_root=project_root)
    key = hashlib.sha256(
        f"{project_root.resolve()}\0{run_id}\0{dataset_path}".encode()
    ).hexdigest()
    return validation_dataset_path(
        str(dataset_path.parent / ".gepa-validation-evidence" / f"{key}.json"),
        project_root=project_root,
        allow_missing=True,
    )


def write_validation_evidence(
    dataset: str,
    *,
    project_root: Path,
    run_id: str,
    identity: dict[str, Any],
    scores: dict[str, float],
) -> None:
    path = validation_evidence_path(dataset, project_root=project_root, run_id=run_id)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"identity": identity, "scores": scores}, handle)
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def read_validation_evidence(
    dataset: str | None,
    *,
    project_root: Path,
    run_id: str,
    identity: dict[str, Any],
) -> dict[str, float]:
    if dataset is None:
        return {}
    path = validation_evidence_path(dataset, project_root=project_root, run_id=run_id)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("identity") != identity:
            return {}
        return {str(key): float(value) for key, value in raw["scores"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        # A missing or interrupted private write can never authorize promotion.
        return {}


def refuse_legacy_validation() -> None:
    """Reject public configuration without ever repeating its private value."""
    raise typer.BadParameter(
        "Remove validation_dataset / --validation-dataset (and any held-out .env entry). "
        "Move the path to GEPA_HELDOUT_DATASET in the harness's environment."
    )


def heldout_dataset(*, required: bool = True) -> str | None:
    """The sole environment lookup for the harness's private dataset path."""
    value = os.environ.get("GEPA_HELDOUT_DATASET")
    if not value:
        if required:
            raise typer.BadParameter(
                "Held-out scoring requires GEPA_HELDOUT_DATASET in the harness's environment."
            )
        return None
    if not Path(value).is_absolute():
        raise typer.BadParameter("GEPA_HELDOUT_DATASET must be an absolute path.")
    return value


def heldout_identity(root: Path) -> tuple[str, str]:
    configured = heldout_dataset()
    assert configured is not None
    path = validation_dataset_path(configured, project_root=root)
    try:
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise typer.BadParameter("Cannot read held-out dataset.") from None


def _pin_path(dataset: str, root: Path, run_id: str) -> Path:
    from .layout import gepa_dir

    key = hashlib.sha256(f"{gepa_dir(root).resolve()}\0{run_id}".encode()).hexdigest()
    return validation_dataset_path(
        str(Path(dataset).parent / ".gepa-heldout" / f"{key}.json"),
        project_root=root,
        allow_missing=True,
    )


def pin_heldout(root: Path, run_id: str) -> None:
    dataset, digest = heldout_identity(root)
    path = _pin_path(dataset, root, run_id)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        json.dump({"dataset": dataset, "digest": digest}, handle)


def check_heldout_pin(root: Path, run_id: str) -> tuple[str, str]:
    dataset, digest = heldout_identity(root)
    try:
        pin = json.loads(_pin_path(dataset, root, run_id).read_text())
    except (OSError, ValueError):
        raise typer.BadParameter(
            "Held-out dataset pin missing or changed; restore the harness's original dataset."
        ) from None
    if pin != {"dataset": dataset, "digest": digest}:
        raise typer.BadParameter("Held-out validation dataset changed after run start.")
    return dataset, digest


_P = ParamSpec("_P")
_T = TypeVar("_T")


def private_evaluation(evaluate: Callable[_P, _T]) -> Callable[_P, _T]:
    """Discard evaluator console chatter on held-out calls, including replay."""
    from contextlib import redirect_stderr, redirect_stdout
    from functools import wraps
    import io

    @wraps(evaluate)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        if kwargs.get("dataset_role") != "validation":
            return evaluate(*args, **kwargs)
        heldout_dataset()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return evaluate(*args, **kwargs)

    return wrapped

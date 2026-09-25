"""Reflectors nominate candidates; an orchestrator-owned process scores them.

The queue contains public identities and training options only. The harness
inherits private dataset access from its orchestrator, never from a request.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from contextvars import ContextVar
import io
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable
from uuid import uuid4

import typer

from .candidates import git_candidate_state
from .layout import candidate_identity_exempt_paths, repo_root, run_dir
from .reflector import _current_tree, run_lock
from .validation import check_heldout_pin, heldout_dataset

app = typer.Typer(no_args_is_help=True)
WAIT_TIMEOUT = 75
_tree_check: ContextVar[Callable[[], None] | None] = ContextVar(
    "harness_tree_check", default=None
)


class StaleNomination(typer.BadParameter):
    def __init__(self) -> None:
        super().__init__(
            "Stale nomination: reflector epoch, HEAD or tree changed. "
            "Commit a clean candidate and run continue again."
        )


def check_scoring_tree() -> None:
    """Called before state publication, including recovery checkpoints."""
    check = _tree_check.get()
    if check is not None:
        check()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def _tree(state: Any) -> dict[str, Any]:
    project = Path(state.project_root) if state.project_root else repo_root()
    tree = _current_tree(state, project)
    if state.candidate_source == "components" and tree["commit_sha"]:
        tree["dirty"] = git_candidate_state(
            project, exclude_paths=candidate_identity_exempt_paths(project)
        ).dirty
    if tree["dirty"] or tree["candidate_id"] is None:
        raise StaleNomination()
    return {key: tree[key] for key in ("candidate_id", "commit_sha")}


def _requests(run_id: str) -> list[tuple[Path, dict[str, Any]]]:
    return [
        (path, json.loads(path.read_text()))
        for path in sorted((run_dir(run_id) / "nominations").glob("*.json"))
    ]


def _result_path(request_path: Path) -> Path:
    return request_path.parent.parent / "results" / request_path.name


def _wait(path: Path, wait_secs: float) -> None:
    deadline = time.monotonic() + wait_secs
    while True:
        result_path = _result_path(path)
        if result_path.exists():
            result = json.loads(result_path.read_text())
            _atomic_json(
                path.parent.parent / "receipts" / path.name, {"received": True}
            )
            typer.echo(result["stdout"], nl=False)
            typer.echo(result["stderr"], err=True, nl=False)
            raise typer.Exit(code=result["exit_code"])
        if wait_secs == 0:
            typer.echo(json.dumps({"nomination_id": path.stem, "status": "pending"}))
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            typer.echo(
                f"Nomination {path.stem} is pending. Run the same continue command again "
                "to wait for the harness (exit 75).",
                err=True,
            )
            raise typer.Exit(code=WAIT_TIMEOUT)
        time.sleep(min(0.2, remaining))


def nominate(
    run_id: str | None, gate_case: list[str], epoch: int | None, wait_secs: float
) -> None:
    from .run import _load_state, _validate_gate_cases

    state = _load_state(run_id)
    current_epoch = state.reflector["epoch"]
    if epoch is not None and epoch != current_epoch:
        raise typer.BadParameter(
            f"Stale reflector epoch {epoch}; current epoch is {current_epoch}."
        )
    identity = {
        "run_id": state.run_id,
        "reflector_epoch": current_epoch,
        **_tree(state),
        "gate_case": gate_case,
    }

    def find_request() -> Path | None:
        requests = _requests(state.run_id)
        for path, request in requests:
            if request.get("reflector_epoch") != current_epoch:
                continue
            if not _result_path(path).exists():
                if all(request.get(key) == value for key, value in identity.items()):
                    return path
                raise typer.BadParameter(
                    "A different nomination is pending; restore its candidate and gate options "
                    "or wait for the harness to reject it as stale."
                )
        for path, request in reversed(requests):
            if all(request.get(key) == value for key, value in identity.items()):
                result_path = _result_path(path)
                if not result_path.exists():
                    continue
                result = json.loads(result_path.read_text())
                received = (path.parent.parent / "receipts" / path.name).exists()
                if (
                    not result.get("stale")
                    and result.get("state_updated_at") == state.updated_at
                    and not (result.get("retryable") and received)
                ):
                    return path
        return None

    # Re-attachment does not take the lock a live scorer holds while evaluating.
    path = find_request()
    if path is None:
        with run_lock(state.run_id):
            latest = _load_state(state.run_id)
            if latest.reflector["epoch"] != current_epoch or _tree(latest) != {
                key: identity[key] for key in ("candidate_id", "commit_sha")
            }:
                raise StaleNomination()
            path = find_request()
            if path is None:
                if gate_case and state.reflection_minibatch_id is not None:
                    _validate_gate_cases(state, gate_case)
                nomination_id = uuid4().hex
                path = run_dir(state.run_id) / "nominations" / f"{nomination_id}.json"
                _atomic_json(path, {**identity, "nomination_id": nomination_id})
    _wait(path, wait_secs)


def _score(path: Path, request: dict[str, Any], state: Any) -> None:
    from .reflector_recovery import continue_run
    from .run import _continue_impl, _load_state

    def check() -> None:
        current = _load_state(state.run_id)
        if (
            request.get("run_id") != state.run_id
            or request.get("nomination_id") != path.stem
            or request.get("reflector_epoch") != current.reflector["epoch"]
            or _tree(current)
            != {key: request.get(key) for key in ("candidate_id", "commit_sha")}
        ):
            raise StaleNomination()

    stdout, stderr = io.StringIO(), io.StringIO()
    code = 0
    stale = False
    token = _tree_check.set(check)
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                check()
                check_heldout_pin(repo_root(), state.run_id)
                continue_run(
                    state.run_id,
                    request["gate_case"],
                    request["reflector_epoch"],
                    _continue_impl,
                    locked=True,
                )
                check()
            except typer.BadParameter as exc:
                stale = isinstance(exc, StaleNomination)
                code = 2
                # Only deliberate CLI refusals cross the boundary, never raw
                # evaluator exceptions (which may contain private case data).
                typer.echo(exc.format_message(), err=True)
            except typer.Exit as exc:
                code = exc.exit_code
    except Exception:
        # Preserve the pending nomination and paid checkpoint for crash replay.
        typer.echo(
            "Harness scoring interrupted; retry serve to recover its checkpoint.",
            err=True,
        )
        raise typer.Exit(code=1) from None
    finally:
        _tree_check.reset(token)
    final_state = _load_state(state.run_id)
    _atomic_json(
        _result_path(path),
        {
            "nomination_id": path.stem,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "exit_code": code,
            "stale": stale,
            "state_updated_at": final_state.updated_at,
            "retryable": final_state.status == "paused_after_infrastructure_error"
            or code not in (0, 70),
        },
    )


@app.command()
def serve(
    run_id: str = typer.Option(..., "--run-id"),
    once: bool = typer.Option(
        False, "--once", help="Process one pending nomination, then return."
    ),
) -> None:
    """Score nominations in the harness process started by the orchestrator."""
    from .run import _load_state

    heldout_dataset()
    while True:
        with run_lock(run_id, wait=True):
            state = _load_state(run_id)
            if not state.heldout_required:
                raise typer.BadParameter(
                    "This run does not require held-out harness scoring."
                )
            check_heldout_pin(repo_root(), run_id)
            pending = [
                (path, request)
                for path, request in _requests(run_id)
                if not _result_path(path).exists()
            ]
            current = []
            for path, request in pending:
                if request.get("reflector_epoch") != state.reflector["epoch"]:
                    _score(path, request, state)
                else:
                    current.append((path, request))
            if current:
                _score(*current[0], state)
            done = _load_state(run_id).status == "done"
        if once or done:
            return
        time.sleep(0.2)

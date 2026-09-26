"""Public training ledgers that never overlap held-out harness-owned views."""

from contextvars import ContextVar
from functools import wraps
import json
from pathlib import Path
from typing import Any, Callable, TypeVar

_active: ContextVar[tuple[str, Path] | None] = ContextVar("lane_ledger", default=None)
_Result = TypeVar("_Result")


def active(run_id: str) -> bool:
    from .validation import heldout_dataset

    current = _active.get()
    return bool(
        current and current[0] == run_id and not heldout_dataset(required=False)
    )


def directory(run_id: str, root: Path | None = None) -> Path:
    from .layout import run_dir

    current = _active.get()
    return current[1] if current and active(run_id) else run_dir(run_id, root)


def training_directory(run_id: str, lane: str, root: Path | None = None) -> Path:
    from .lanes import _validate_lane_id
    from .layout import run_dir

    return run_dir(run_id, root) / "lanes" / _validate_lane_id(lane)


def lane_training_evaluation(
    evaluate: Callable[..., _Result],
) -> Callable[..., _Result]:
    """Route reflector training only; harness evaluations keep their private ledger."""

    @wraps(evaluate)
    def wrapped(**kwargs: Any) -> _Result:
        from .layout import latest_run_id, repo_root, run_state_path
        from .validation import heldout_dataset

        lane = kwargs.get("lane")
        if (
            not lane
            or heldout_dataset(required=False)
            or kwargs.get("dataset_role", "training") != "training"
        ):
            return evaluate(**kwargs)
        root = kwargs.get("workspace_root") or repo_root()
        run_id = kwargs.get("run_id") or latest_run_id(root)
        if not run_id:
            return evaluate(**kwargs)
        state = run_state_path(run_id, root)
        if not state.exists() or not json.loads(state.read_text()).get(
            "heldout_required"
        ):
            return evaluate(**kwargs)
        token = _active.set((run_id, training_directory(run_id, lane, root)))
        try:
            return evaluate(**kwargs)
        finally:
            _active.reset(token)

    return wrapped

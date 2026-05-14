"""On-disk layout under `.gepa/` plus `GepaConfig` loader and agent resolver.

This module is the single source of truth for where each artifact lives. Every
verb in the CLI reads/writes through these helpers so the directory shape stays
consistent across `init`, `propose`, `eval`, `apply`, etc.

Layout overview::

    <repo_root>/
        .gepa/
            gepa.toml                  # GepaConfig
            dataset.jsonl              # default dataset path (configurable)
            journal.jsonl              # Reflection Ledger
            components/<slot>.md       # confirmed component text values
            staged/<slot>.md           # stubs awaiting `gepa components confirm`
            runs/<run_id>/
                minibatches/<mb_id>.json
                candidates/<id>.json
                proposals/<id>.json
                traces/<case_id>/...
                pareto.jsonl           # append-only Pareto history

See pydanticaigepa-spec-973 and pydanticaigepa-dec-xd6 for the contract this
file implements.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
import tomllib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GEPA_DIRNAME = ".gepa"
CONFIG_FILENAME = "gepa.toml"
DEFAULT_DATASET_PATH = ".gepa/dataset.jsonl"
JOURNAL_FILENAME = "journal.jsonl"


class GepaConfigError(RuntimeError):
    """Raised when the GepaConfig is malformed or the referenced agent cannot be loaded."""


@dataclass(frozen=True)
class GepaConfig:
    """Configuration loaded from `.gepa/gepa.toml`."""

    agent: str
    """Module reference for the user's pydantic-ai agent, e.g. ``"mypkg.agents:my_agent"``."""

    dataset: str = DEFAULT_DATASET_PATH
    """Relative path (from repo root) to the dataset JSONL file."""

    metric: str | None = None
    """Optional ``module.path:attr`` reference for a custom metric callable."""

    defaults: dict[str, Any] = field(default_factory=dict)
    """Default knob values for `gepa eval`, etc. Optional."""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> GepaConfig:
        if "agent" not in data:
            raise GepaConfigError(
                "Missing required key 'agent' in gepa.toml (expected 'module.path:attr')."
            )
        agent = data["agent"]
        if not isinstance(agent, str) or ":" not in agent:
            raise GepaConfigError(
                f"Invalid 'agent' value: {agent!r}. Expected 'module.path:attr'."
            )
        dataset = data.get("dataset", DEFAULT_DATASET_PATH)
        metric = data.get("metric")
        if metric is not None and (not isinstance(metric, str) or ":" not in metric):
            raise GepaConfigError(
                f"Invalid 'metric' value: {metric!r}. Expected 'module.path:attr' or omit."
            )
        defaults = data.get("defaults", {}) or {}
        if not isinstance(defaults, dict):
            raise GepaConfigError(
                f"'defaults' must be a TOML table, got {type(defaults).__name__}."
            )
        return GepaConfig(
            agent=agent, dataset=dataset, metric=metric, defaults=defaults
        )

    @staticmethod
    def load(path: Path) -> GepaConfig:
        if not path.exists():
            raise GepaConfigError(
                f"No gepa.toml at {path}. Run `gepa init --agent module:attr` first."
            )
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        return GepaConfig.from_dict(raw)


def repo_root(start: Path | None = None) -> Path:
    """Return the repo root (directory containing `.gepa/` or `pyproject.toml`).

    Walks upward from `start` (or cwd) until a `.gepa/` or `pyproject.toml` is found.
    Falls back to cwd if neither marker exists.
    """
    cursor = (start or Path.cwd()).resolve()
    for candidate in [cursor, *cursor.parents]:
        if (candidate / GEPA_DIRNAME).is_dir():
            return candidate
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def gepa_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / GEPA_DIRNAME


def config_path(root: Path | None = None) -> Path:
    return gepa_dir(root) / CONFIG_FILENAME


def components_dir(root: Path | None = None) -> Path:
    return gepa_dir(root) / "components"


def staged_dir(root: Path | None = None) -> Path:
    return gepa_dir(root) / "staged"


def journal_path(root: Path | None = None) -> Path:
    return gepa_dir(root) / JOURNAL_FILENAME


def runs_dir(root: Path | None = None) -> Path:
    return gepa_dir(root) / "runs"


def run_dir(run_id: str, root: Path | None = None) -> Path:
    return runs_dir(root) / run_id


def minibatch_path(run_id: str, mb_id: str, root: Path | None = None) -> Path:
    return run_dir(run_id, root) / "minibatches" / f"{mb_id}.json"


def candidate_dir(run_id: str, root: Path | None = None) -> Path:
    return run_dir(run_id, root) / "candidates"


def proposal_dir(run_id: str, root: Path | None = None) -> Path:
    return run_dir(run_id, root) / "proposals"


def traces_dir(run_id: str, case_id: str, root: Path | None = None) -> Path:
    return run_dir(run_id, root) / "traces" / case_id


def pareto_log_path(run_id: str, root: Path | None = None) -> Path:
    return run_dir(run_id, root) / "pareto.jsonl"


def ensure_layout(root: Path | None = None) -> Path:
    """Create the standard `.gepa/` subtree if missing. Idempotent. Returns the gepa dir."""
    base = gepa_dir(root)
    for sub in (base, components_dir(root), staged_dir(root), runs_dir(root)):
        sub.mkdir(parents=True, exist_ok=True)
    journal = journal_path(root)
    if not journal.exists():
        journal.touch()
    return base


_RUN_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")


def new_run_id(now: datetime | None = None) -> str:
    """Return a sortable run id: ``YYYYMMDDThhmmssZ-<8 hex>``."""
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    suffix = uuid.uuid4().hex[:8]
    return f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{suffix}"


def is_run_id(value: str) -> bool:
    return bool(_RUN_ID_RE.match(value))


def latest_run_id(root: Path | None = None) -> str | None:
    base = runs_dir(root)
    if not base.is_dir():
        return None
    candidates = sorted(
        (p.name for p in base.iterdir() if p.is_dir() and is_run_id(p.name)),
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_agent(config: GepaConfig) -> Any:
    """Import ``config.agent`` (``module.path:attr``) and return the agent attribute."""
    return resolve_module_attr(config.agent, kind="agent")


def resolve_metric(config: GepaConfig) -> Any:
    """Import the configured metric, or return ``None`` to fall back to the CLI default."""
    if not config.metric:
        return None
    return resolve_module_attr(config.metric, kind="metric")


def resolve_module_attr(ref: str, *, kind: str = "object") -> Any:
    """Resolve a ``module.path:attr`` reference to the named attribute."""
    if ":" not in ref:
        raise GepaConfigError(
            f"Invalid {kind} ref {ref!r}: expected 'module.path:attr'."
        )
    module_path, attr = ref.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise GepaConfigError(
            f"Could not import module {module_path!r} for {kind} ref {ref!r}: {exc}"
        ) from exc
    if not hasattr(module, attr):
        raise GepaConfigError(
            f"Module {module_path!r} has no attribute {attr!r} ({kind} ref {ref!r})."
        )
    return getattr(module, attr)


def write_default_config(
    agent: str,
    dataset: str = DEFAULT_DATASET_PATH,
    *,
    metric: str | None = None,
    root: Path | None = None,
    force: bool = False,
) -> Path:
    """Write a minimal `.gepa/gepa.toml`. Returns the path written.

    ``metric`` is written as a top-level key when provided. Anything that
    might evolve into a defaults block lives outside this function — keeping
    the bootstrap template small avoids documenting features that are not yet
    wired into the CLI.
    """
    path = config_path(root)
    if path.exists() and not force:
        raise GepaConfigError(f"{path} already exists. Pass --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'agent = "{agent}"',
        f'dataset = "{dataset}"',
    ]
    if metric:
        lines.append(f'metric = "{metric}"')
    contents = "\n".join(lines) + "\n"
    path.write_text(contents, encoding="utf-8")
    return path


def insert_repo_root_on_path(root: Path | None = None) -> None:
    """Make sure ``repo_root()`` is importable as a package source.

    Coding agents often run `gepa` from the repo root where the agent module is
    in the local ``src/`` or ``./<package>/`` tree. We push the root onto
    ``sys.path`` so ``importlib.import_module`` can resolve those modules
    without requiring an editable install.
    """
    root_path = str((root or repo_root()).resolve())
    if root_path not in sys.path:
        sys.path.insert(0, root_path)


def load_dotenv(root: Path | None = None) -> dict[str, str]:
    """Load ``.env`` from the repo root into ``os.environ`` without overriding.

    Returns the dict of keys that were actually applied (empty if no .env
    exists, or if every key in .env was already set in the environment).
    Behaves like ``os.environ.setdefault`` — existing env vars always win.

    Format: ``KEY=VALUE`` per line. Blank lines and lines starting with ``#``
    are ignored. A leading ``export `` on the key is stripped. A single matched
    pair of surrounding quotes on the value is stripped. Lines without an ``=``
    are silently skipped.
    """
    base = (root or repo_root()).resolve()
    path = base / ".env"
    if not path.is_file():
        return {}

    applied: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw_value = stripped.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key:
            continue
        value = raw_value.strip()
        # Strip a single matched pair of surrounding quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied

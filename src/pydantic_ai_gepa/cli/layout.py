"""On-disk layout under `.gepa/` plus `GepaConfig` loader and agent resolver.

This module is the single source of truth for where each artifact lives. Every
verb in the CLI reads/writes through these helpers so the directory shape stays
consistent across `init`, `eval`, `apply`, etc.

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
import subprocess
import sys
import tomllib
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence


GEPA_DIRNAME = ".gepa"
GEPA_DIR_ENV = "GEPA_DIR"
CONFIG_FILENAME = "gepa.toml"
JOURNAL_FILENAME = "journal.jsonl"
CandidateSource = Literal["components", "git"]


_explicit_gepa_dirname: str | None = None


def set_gepa_dirname(name: str | None) -> None:
    """Override the active workspace dirname (used by the root Typer callback).

    Pass ``None`` to clear the override; subsequent reads fall back to the
    ``GEPA_DIR`` environment variable, then to the default ``.gepa``.
    """
    global _explicit_gepa_dirname
    _explicit_gepa_dirname = name


def current_gepa_dirname() -> str:
    """Return the active workspace dirname.

    Precedence: explicit override (``--gepa-dir`` flag) > ``GEPA_DIR`` env
    var > the default ``.gepa``. Read on every call so ``.env``-loaded
    values picked up by the root callback take effect.
    """
    if _explicit_gepa_dirname is not None:
        return _explicit_gepa_dirname
    env_value = os.environ.get(GEPA_DIR_ENV)
    if env_value:
        return env_value
    return GEPA_DIRNAME


def default_dataset_path() -> str:
    """Default dataset path (relative to repo root) for the active dirname."""
    return f"{current_gepa_dirname()}/dataset.jsonl"


DEFAULT_DATASET_PATH = ".gepa/dataset.jsonl"
"""Kept for backwards compatibility. Prefer :func:`default_dataset_path`,
which reflects the currently active ``--gepa-dir`` override."""


class GepaConfigError(RuntimeError):
    """Raised when the GepaConfig is malformed or the referenced agent cannot be loaded."""


@dataclass(frozen=True)
class GepaConfig:
    """Configuration loaded from `.gepa/gepa.toml`."""

    agent: str | None = None
    """Optional module reference for the user's pydantic-ai agent."""

    evaluate: str | None = None
    """Optional plain ``module.path:attr`` evaluation callable for git mode."""

    candidate_source: CandidateSource = "components"
    """Candidate identity and application strategy. Component mode is the default."""

    dataset: str = ".gepa/dataset.jsonl"
    """Relative path (from repo root) to the dataset JSONL file.

    When omitted from ``gepa.toml``, ``GepaConfig.from_dict`` fills this
    in from :func:`default_dataset_path` so a workspace at
    ``.gepa.personalize/`` gets ``.gepa.personalize/dataset.jsonl`` rather
    than the hardcoded default.
    """

    metric: str | None = None
    """Optional ``module.path:attr`` reference for a custom metric callable."""

    case_factory: str | None = None
    """Optional ``module.path:attr`` reference for a case-factory callable.

    The factory's signature is ``(case: Case) -> BaseModel`` (sync or
    async). When set, the adapter invokes it for every rollout to convert
    the raw dataset row into the agent's input model — useful when the
    input carries binary content (PDFs, images, audio) or other
    non-JSON-roundtrippable state. The factory runs at eval time only;
    the runtime agent's input model stays unchanged.
    """

    defaults: dict[str, Any] = field(default_factory=dict)
    """Default knob values for `gepa eval`, etc. Optional."""

    skills: str | None = None
    """Optional path (relative to repo root) to an Agent Skills directory."""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> GepaConfig:
        candidate_source = data.get("candidate_source", "components")
        if candidate_source not in {"components", "git"}:
            raise GepaConfigError(
                "Invalid 'candidate_source' value: "
                f"{candidate_source!r}. Expected 'components' or 'git'."
            )
        agent = data.get("agent")
        evaluate = data.get("evaluate")
        if candidate_source == "components" and agent is None:
            raise GepaConfigError(
                "Missing required key 'agent' in gepa.toml (expected 'module.path:attr')."
            )
        if candidate_source == "git" and agent is None and evaluate is None:
            raise GepaConfigError(
                "Git candidate mode requires either 'agent' or 'evaluate' in gepa.toml."
            )
        if agent is not None and (not isinstance(agent, str) or ":" not in agent):
            raise GepaConfigError(
                f"Invalid 'agent' value: {agent!r}. Expected 'module.path:attr'."
            )
        if evaluate is not None and (
            not isinstance(evaluate, str) or ":" not in evaluate
        ):
            raise GepaConfigError(
                f"Invalid 'evaluate' value: {evaluate!r}. "
                "Expected 'module.path:attr' or omit."
            )
        if candidate_source == "components" and evaluate is not None:
            raise GepaConfigError(
                "'evaluate' is only supported when candidate_source = \"git\"."
            )
        dataset = data.get("dataset", default_dataset_path())
        metric = data.get("metric")
        if metric is not None and (not isinstance(metric, str) or ":" not in metric):
            raise GepaConfigError(
                f"Invalid 'metric' value: {metric!r}. Expected 'module.path:attr' or omit."
            )
        case_factory = data.get("case_factory")
        if case_factory is not None and (
            not isinstance(case_factory, str) or ":" not in case_factory
        ):
            raise GepaConfigError(
                f"Invalid 'case_factory' value: {case_factory!r}. "
                "Expected 'module.path:attr' or omit."
            )
        defaults = data.get("defaults", {}) or {}
        if not isinstance(defaults, dict):
            raise GepaConfigError(
                f"'defaults' must be a TOML table, got {type(defaults).__name__}."
            )
        skills = data.get("skills")
        if skills is not None and not isinstance(skills, str):
            raise GepaConfigError(
                f"Invalid 'skills' value: {skills!r}. Expected a path string or omit."
            )
        return GepaConfig(
            agent=agent,
            evaluate=evaluate,
            candidate_source=candidate_source,
            dataset=dataset,
            metric=metric,
            case_factory=case_factory,
            defaults=defaults,
            skills=skills,
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
    """Return the repo root (directory containing the active workspace or `pyproject.toml`).

    Walks upward from ``start`` (or cwd) until the active workspace dirname
    (see :func:`current_gepa_dirname`) or a ``pyproject.toml`` is found. Falls
    back to cwd if neither marker exists.

    When the active dirname is an absolute path, the workspace itself can't
    sit at the repo root, so only ``pyproject.toml`` is used for the walk.
    """
    cursor = (start or Path.cwd()).resolve()
    dirname = current_gepa_dirname()
    workspace_marker_is_relative = not Path(dirname).is_absolute()
    for candidate in [cursor, *cursor.parents]:
        if workspace_marker_is_relative and (candidate / dirname).is_dir():
            return candidate
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def git_root(start: Path | None = None) -> Path:
    """Return the Git top-level containing ``start``.

    A Python project may be nested inside a monorepo, so this is deliberately
    separate from :func:`repo_root`, which identifies the import/config root.
    """

    cursor = (start or repo_root()).resolve()
    try:
        value = subprocess.run(
            ["git", "-C", str(cursor), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = (
            exc.stderr.strip()
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        raise GepaConfigError(
            f"Could not resolve Git top-level from {cursor}: {detail}"
        ) from exc
    return Path(value).resolve()


def project_root_for_workspace(workspace: Path) -> Path:
    """Resolve the primary Python project that owns an explicit workspace.

    Nested workspaces such as ``api/evals/foo/.gepa/rlm`` belong to the
    nearest ancestor containing ``pyproject.toml``. Flat repositories without
    a Python project marker retain the historical Git-top-level behavior.
    """

    resolved = workspace.resolve()
    for candidate in [resolved.parent, *resolved.parent.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return git_root(resolved.parent)


def project_prefix(project: Path, repository: Path | None = None) -> Path:
    """Return the project path relative to its Git top-level."""

    resolved_project = project.resolve()
    resolved_repository = (repository or git_root(resolved_project)).resolve()
    try:
        return resolved_project.relative_to(resolved_repository)
    except ValueError as exc:
        raise GepaConfigError(
            f"Python project {resolved_project} is outside Git root "
            f"{resolved_repository}."
        ) from exc


def candidate_project_root(
    primary_project_root: Path, candidate_git_root: Path
) -> Path:
    """Map a primary monorepo project into a candidate Git worktree."""

    prefix = project_prefix(primary_project_root)
    candidate = (candidate_git_root.resolve() / prefix).resolve()
    if not candidate.is_dir():
        raise GepaConfigError(
            f"Candidate project root does not exist: {candidate} "
            f"(project prefix {prefix.as_posix()!r})."
        )
    return candidate


def gepa_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / current_gepa_dirname()


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


def run_state_path(run_id: str, root: Path | None = None) -> Path:
    return run_dir(run_id, root) / "state.json"


def final_report_path(run_id: str, root: Path | None = None) -> Path:
    return run_dir(run_id, root) / "final_report.md"


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


def resolve_agent(config: GepaConfig, *, expected_root: Path | None = None) -> Any:
    """Import ``config.agent`` (``module.path:attr``) and return the agent attribute."""
    if not config.agent:
        raise GepaConfigError("No 'agent' is configured in gepa.toml.")
    return resolve_module_attr(config.agent, kind="agent", expected_root=expected_root)


def resolve_evaluate(config: GepaConfig, *, expected_root: Path | None = None) -> Any:
    """Import the optional plain git-mode evaluation callable."""
    if not config.evaluate:
        return None
    return resolve_module_attr(
        config.evaluate, kind="evaluate", expected_root=expected_root
    )


def resolve_metric(config: GepaConfig, *, expected_root: Path | None = None) -> Any:
    """Import the configured metric, or return ``None`` to fall back to the CLI default."""
    if not config.metric:
        return None
    return resolve_module_attr(
        config.metric, kind="metric", expected_root=expected_root
    )


def resolve_case_factory(
    config: GepaConfig, *, expected_root: Path | None = None
) -> Any:
    """Import the configured case factory, or return ``None`` if absent.

    The returned callable (when set) is invoked by the adapter at rollout
    time to convert a raw ``Case`` into the agent's input model — see
    ``GepaConfig.case_factory`` for the contract.
    """
    if not config.case_factory:
        return None
    return resolve_module_attr(
        config.case_factory, kind="case_factory", expected_root=expected_root
    )


def resolve_skills(config: GepaConfig, *, root: Path | None = None) -> Any:
    """Resolve the optional ``skills`` directory from ``gepa.toml``."""
    if not config.skills:
        return None
    from pydantic_ai_gepa.skills import SkillsFS

    skills_path = (root or repo_root()) / config.skills
    if not skills_path.exists():
        raise GepaConfigError(f"Skills directory does not exist: {skills_path}")
    return SkillsFS.from_disk(skills_path)


def resolve_module_attr(
    ref: str, *, kind: str = "object", expected_root: Path | None = None
) -> Any:
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
    if expected_root is not None:
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise GepaConfigError(
                f"Resolved {kind} module {module_path!r} has no source file; "
                f"expected it under candidate project {expected_root.resolve()}."
            )
        resolved_file = Path(module_file).resolve()
        try:
            resolved_file.relative_to(expected_root.resolve())
        except ValueError as exc:
            raise GepaConfigError(
                f"Resolved {kind} ref {ref!r} outside the candidate project: "
                f"{resolved_file} is not under {expected_root.resolve()}."
            ) from exc
    return getattr(module, attr)


def write_default_config(
    agent: str | None,
    dataset: str | None = None,
    *,
    evaluate: str | None = None,
    candidate_source: CandidateSource = "components",
    metric: str | None = None,
    case_factory: str | None = None,
    skills: str | None = None,
    root: Path | None = None,
    force: bool = False,
) -> Path:
    """Write a minimal `.gepa/gepa.toml`. Returns the path written.

    ``metric`` and ``case_factory`` are written as top-level keys when
    provided. Anything that might evolve into a defaults block lives
    outside this function — keeping the bootstrap template small avoids
    documenting features that are not yet wired into the CLI.
    """
    path = config_path(root)
    if path.exists() and not force:
        raise GepaConfigError(f"{path} already exists. Pass --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_dataset = dataset if dataset is not None else default_dataset_path()
    lines = []
    if agent:
        lines.append(f'agent = "{agent}"')
    if evaluate:
        lines.append(f'evaluate = "{evaluate}"')
    lines.extend(
        [
            f'candidate_source = "{candidate_source}"',
            f'dataset = "{resolved_dataset}"',
        ]
    )
    if metric:
        lines.append(f'metric = "{metric}"')
    if case_factory:
        lines.append(f'case_factory = "{case_factory}"')
    if skills:
        lines.append(f'skills = "{skills}"')
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
    project = (root or repo_root()).resolve()
    for import_root in reversed((project / "src", project)):
        if not import_root.is_dir():
            continue
        path_text = str(import_root)
        if path_text in sys.path:
            sys.path.remove(path_text)
        sys.path.insert(0, path_text)


def _module_source_paths(module: Any) -> tuple[Path, ...]:
    """Return file and namespace-package locations for an imported module."""

    raw_paths: list[Any] = []
    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        raw_paths.append(module_file)
    module_path = getattr(module, "__path__", None)
    if module_path is not None:
        try:
            raw_paths.extend(module_path)
        except TypeError:
            pass
    resolved: list[Path] = []
    for raw in raw_paths:
        try:
            path = Path(raw).resolve()
        except (OSError, RuntimeError, TypeError):
            continue
        if path not in resolved:
            resolved.append(path)
    return tuple(resolved)


def _is_below(path: Path | None, root: Path) -> bool:
    if path is None:
        return False
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _module_is_below(module: Any, root: Path) -> bool:
    return any(_is_below(path, root) for path in _module_source_paths(module))


@contextmanager
def candidate_import_context(
    *,
    primary_project_root: Path,
    candidate_project_root: Path,
    refs: Sequence[str | None],
) -> Iterator[None]:
    """Isolate configured and transitive imports to one candidate project.

    Managed lane evaluation can run in a process that previously evaluated
    the primary checkout. Evicting project-owned modules before resolution is
    therefore as important as changing ``sys.path``: otherwise Python reuses
    the primary module object and silently evaluates the wrong candidate.
    """

    primary = primary_project_root.resolve()
    candidate = candidate_project_root.resolve()
    primary_repository = git_root(primary)
    candidate_repository = git_root(candidate)
    configured_roots = {ref.split(":", 1)[0].split(".", 1)[0] for ref in refs if ref}
    removed: dict[str, Any] = {}
    for name, module in list(sys.modules.items()):
        if name == "pydantic_ai_gepa" or name.startswith("pydantic_ai_gepa."):
            continue
        top_level = name.split(".", 1)[0]
        if (
            _module_is_below(module, primary_repository)
            or _module_is_below(module, candidate_repository)
            or top_level in configured_roots
        ):
            removed[name] = module
            sys.modules.pop(name, None)

    original_path = list(sys.path)
    original_cwd = Path.cwd()
    translated_paths: list[str] = []
    external_paths: list[str] = []
    for entry in sys.path:
        try:
            resolved_entry = Path(entry or original_cwd).resolve()
            suffix = resolved_entry.relative_to(primary_repository)
        except (OSError, ValueError):
            external_paths.append(entry)
            continue
        translated = (candidate_repository / suffix).resolve()
        if translated.exists():
            translated_text = str(translated)
            if translated_text not in translated_paths:
                translated_paths.append(translated_text)
    sys.path[:] = [*translated_paths, *external_paths]
    insert_repo_root_on_path(candidate)
    os.chdir(candidate)
    try:
        yield
    finally:
        os.chdir(original_cwd)
        for name, module in list(sys.modules.items()):
            if name == "pydantic_ai_gepa" or name.startswith("pydantic_ai_gepa."):
                continue
            top_level = name.split(".", 1)[0]
            if (
                _module_is_below(module, candidate_repository)
                or top_level in configured_roots
            ):
                sys.modules.pop(name, None)
        sys.modules.update(removed)
        sys.path[:] = original_path


_DOTENV_INTERPOLATION_RE = re.compile(
    r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))"
)


def _interpolate_dotenv_value(value: str, env: dict[str, str]) -> str:
    """Expand ``$VAR`` / ``${VAR}`` references against ``env`` (missing → empty)."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return env.get(name, "")

    return _DOTENV_INTERPOLATION_RE.sub(replace, value)


def load_dotenv(root: Path | None = None) -> dict[str, str]:
    """Load ``.env`` from the repo root into ``os.environ`` without overriding.

    Returns the dict of keys that were actually applied (empty if no .env
    exists, or if every key in .env was already set in the environment).
    Behaves like ``os.environ.setdefault`` — existing env vars always win.

    Format: ``KEY=VALUE`` per line. Blank lines and lines starting with ``#``
    are ignored. A leading ``export `` on the key is stripped. A single matched
    pair of surrounding quotes on the value is stripped. Lines without an ``=``
    are silently skipped.

    ``$VAR`` and ``${VAR}`` references are expanded against the in-process
    environment (current ``os.environ`` plus any prior keys from this same
    file). Single-quoted values are kept verbatim (POSIX shell semantics);
    double-quoted and unquoted values are interpolated. Missing references
    expand to the empty string.

    Multi-line values, escape sequences, and command substitution are NOT
    supported — switch to ``python-dotenv`` if the repo needs those.
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
        quote_type: str | None = None
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            quote_type = value[0]
            value = value[1:-1]
        if quote_type != "'":
            value = _interpolate_dotenv_value(value, dict(os.environ))
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied

"""Parse-only verification of the explicit trusted text scoring scope."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from .layout import GepaConfig, gepa_dir, git_root, valid_string_symbol
from .safe_git import safe_repository
from .scoring_sandbox import (
    ScoringSandboxError,
    _relative_file,
    _verified_checkout_entries,
    candidate_components,
)


def require_text_scope(config: GepaConfig) -> None:
    acceptance = config.acceptance
    files = acceptance.component_files
    text = acceptance.text_files
    python = acceptance.python_string_symbols
    if (
        acceptance.trusted_scorer is not True
        or acceptance.pinned_scorer is not True
        or acceptance.mode != "scalar"
        or not files
        or len(set(files)) != len(files)
        or len(set(text)) != len(text)
        or set(text) & set(python)
        or set(files) != set(text) | set(python)
        or not all(_relative_file(p) for p in files)
        or any(Path(p).suffix not in {".md", ".txt"} for p in text)
        or any(
            Path(p).suffix != ".py"
            or not isinstance(names, (list, tuple))
            or not names
            or any(not valid_string_symbol(n) for n in names)
            for p, names in python.items()
        )
    ):
        raise ScoringSandboxError(
            "Trusted in-process scoring requires an explicit text-only component scope and trusted pinned scalar scoring."
        )


def _masked(source: str, symbols: list[str]) -> str:
    tree = ast.parse(source)
    pending = set(symbols)
    for node in tree.body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
            doc_symbol = node.name + ".__doc__"
            if doc_symbol in pending:
                if ast.get_docstring(node) is None:
                    raise ValueError
                node.body[0] = ast.Expr(value=ast.Constant(value="<docstring>"))
                pending.remove(doc_symbol)
        elif isinstance(node, ast.Assign):
            names = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        if not pending.intersection(names):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                child.value = "<instruction>"
        pending.difference_update(names)
    if pending:
        raise ValueError
    return ast.dump(tree, include_attributes=False)


def candidate_baseline(scorer_project: Path, meter: Any) -> str:
    from .harness_record import for_run
    from .lane_repositories import load

    record = for_run(meter.run_id, meter.root or scorer_project)
    state = json.loads(record.read("state.json") or "{}") if record else {}
    if state.get("lane_repository_version") == 1:
        return load(meter.root or scorer_project, meter.run_id).seed
    baseline = record.read("@trusted_text_base") if record else None
    if baseline:
        return baseline
    raise ScoringSandboxError(
        "Trusted scoring requires a privately recorded candidate repository seed."
    )


def verify_candidate(
    config: GepaConfig,
    project: Path,
    sha: str,
    scorer_project: Path,
    revision: str,
    baseline: str,
) -> dict[str, str]:
    """Compare verified repository trees, including paths outside the Python root."""
    require_text_scope(config)

    def tree(root: Path, commit: str) -> dict[str, tuple[bytes, bytes]]:
        with safe_repository(root) as git:
            return {
                p.relative_to(git.repository.root).as_posix(): (mode, oid)
                for p, mode, oid in _verified_checkout_entries(
                    git, git.repository.root, commit
                )
                if mode != b"040000"
            }

    before = tree(project, baseline)
    after = tree(project, sha)
    with safe_repository(project) as git:
        if git.run(
            "merge-base", "--is-ancestor", baseline, sha, capture_output=True
        ).returncode:
            raise ScoringSandboxError(
                "Candidate does not descend from its recorded baseline."
            )
        if git.run(
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            check=True,
            capture_output=True,
        ).stdout:
            raise ScoringSandboxError(
                "Trusted scoring refuses untracked candidate files."
            )
    prefix = project.resolve().relative_to(git_root(project))
    allowed = {(prefix / p).as_posix() for p in config.acceptance.component_files}
    for path in before.keys() | after.keys():
        if before.get(path) == after.get(path):
            continue
        if (
            path not in allowed
            or path not in before
            or path not in after
            or before[path][0] != after[path][0]
            or after[path][0] != b"100644"
        ):
            raise ScoringSandboxError(
                "Candidate changes exceed the declared text scope."
            )
    components = candidate_components(project, sha, config.acceptance.component_files)
    pinned = candidate_components(
        scorer_project, revision, config.acceptance.component_files
    )
    try:
        for path, symbols in config.acceptance.python_string_symbols.items():
            if _masked(pinned[path], symbols) != _masked(components[path], symbols):
                raise ValueError
    except (SyntaxError, ValueError, TypeError, RecursionError):
        raise ScoringSandboxError(
            "Candidate Python changes exceed the declared string symbols."
        ) from None
    return components


def private_roots(project: Path, scorer_project: Path, meter: Any) -> list[Path]:
    """Use the driver's project/lane exclusions and private-record lane reads."""
    from .lanes import load_all_lane_states
    from .lane_repositories import lanes_root

    roots = {
        project.resolve(),
        scorer_project.resolve(),
        git_root(project),
        git_root(scorer_project),
    }
    workspace = meter.root or scorer_project
    roots.update(
        (
            workspace.resolve(),
            gepa_dir(workspace).resolve(),
            lanes_root(workspace).resolve(),
        )
    )
    for lane in load_all_lane_states(workspace, meter.run_id):
        roots.update(
            Path(p).resolve()
            for p in (lane.worktree_path, lane.candidate_project_path)
            if p
        )
    return sorted(roots)


def check_private_storage(path: Path, roots: list[Path]) -> None:
    import os
    import stat

    resolved = path.resolve()
    info = path.lstat()
    if (
        any(resolved.is_relative_to(root.resolve()) for root in roots)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ScoringSandboxError(
            "Trusted scoring storage must be private, owned by this user, mode 0700 and outside project, GEPA_DIR and lanes."
        )

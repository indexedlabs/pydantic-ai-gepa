"""Defense-in-depth import isolation for the trusted scoring subprocess."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
from pathlib import Path
import site
import sys
from typing import Any

from .layout import _module_source_paths


class CandidateImportGuard(importlib.abc.MetaPathFinder):
    def __init__(
        self, roots: list[str], *, scorer_root: str, pinned_root: str, project: Path
    ) -> None:
        self.roots = tuple(Path(root).resolve() for root in roots)
        self.scorer_root = Path(scorer_root).resolve()
        self.pinned_root = Path(pinned_root).resolve()
        # Only interpreter installation directories qualify, never .pth targets.
        self.dependencies = tuple(Path(p).resolve() for p in site.getsitepackages())
        self.pinned_paths = [
            str(p.resolve()) for p in (project / "src", project) if p.is_dir()
        ]
        self.finders = tuple(sys.meta_path)
        self.refused = False
        sys.path[:] = [p for p in sys.path if p and not self.blocked(p)]
        sys.path[:] = self.pinned_paths + [
            p for p in sys.path if p not in self.pinned_paths
        ]
        sys.meta_path.insert(0, self)
        self.check()

    def blocked(self, value: str | Path) -> bool:
        path = Path(value).resolve()
        if path.is_relative_to(self.pinned_root) or any(
            path.is_relative_to(root) for root in self.dependencies
        ):
            return False
        return path.is_relative_to(self.scorer_root) or any(
            path.is_relative_to(root) for root in self.roots
        )

    def refuse(self) -> None:
        self.refused = True
        raise ImportError("Unpinned project imports are forbidden in trusted scoring.")

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if self.refused or any(self.blocked(p or Path.cwd()) for p in sys.path):
            self.refuse()
        # Editable-install finders can precede PathFinder. Give the pinned tree
        # first refusal, including namespace packages and their submodules.
        search = (
            self.pinned_paths
            if path is None
            else [p for p in path if Path(p).resolve().is_relative_to(self.pinned_root)]
        )
        spec = importlib.machinery.PathFinder.find_spec(fullname, search, target)
        if spec is not None:
            return spec
        for finder in self.finders:
            spec = finder.find_spec(fullname, path, target)
            if spec is None:
                continue
            locations = list(spec.submodule_search_locations or ())
            if spec.origin and spec.origin not in {"built-in", "frozen"}:
                locations.append(spec.origin)
            if any(self.blocked(p) for p in locations):
                self.refuse()
            return spec
        return None

    def check(self) -> None:
        if self.refused or any(self.blocked(p or Path.cwd()) for p in sys.path):
            self.refuse()
        for module in tuple(sys.modules.values()):
            if module is not None and any(
                self.blocked(p) for p in _module_source_paths(module)
            ):
                self.refuse()

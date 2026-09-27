"""Defense-in-depth import isolation for the trusted scoring subprocess."""

from __future__ import annotations

import importlib.abc
from pathlib import Path
import sys
from typing import Any

from .layout import _module_source_paths


class CandidateImportGuard(importlib.abc.MetaPathFinder):
    def __init__(self, roots: list[str]) -> None:
        self.roots = tuple(Path(root).resolve() for root in roots)
        self.finders = tuple(sys.meta_path)
        self.refused = False
        sys.path[:] = [p for p in sys.path if p and not self.blocked(p)]
        sys.meta_path.insert(0, self)
        self.check()

    def blocked(self, value: str | Path) -> bool:
        path = Path(value).resolve()
        return any(path.is_relative_to(root) for root in self.roots)

    def refuse(self) -> None:
        self.refused = True
        raise ImportError("Candidate imports are forbidden in trusted scoring.")

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
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

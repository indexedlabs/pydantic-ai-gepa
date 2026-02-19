"""In-memory filesystem primitives for Agent Skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


def normalize_rel_path(path: str) -> str:
    """Normalize and validate a relative path (posix separators, no '..')."""
    if not path:
        raise ValueError("path must be non-empty")
    p = Path(path)
    if p.is_absolute():
        raise ValueError("path must be relative")
    parts = list(p.parts)
    if any(part in ("..", ".", "") for part in parts):
        raise ValueError("path must not contain '.', '..', or empty segments")
    normalized = Path(*parts).as_posix().lstrip("/")
    if not normalized or normalized == ".":
        raise ValueError("path must not resolve to root")
    return normalized


@dataclass(slots=True)
class File:
    """A file node in an in-memory filesystem."""

    content: bytes

    def read_text(self, *, encoding: str = "utf-8") -> str:
        return self.content.decode(encoding)

    @classmethod
    def from_text(cls, text: str, *, encoding: str = "utf-8") -> File:
        return cls(content=text.encode(encoding))


@dataclass(slots=True)
class Directory:
    """A directory node in an in-memory filesystem."""

    entries: dict[str, Directory | File] = field(default_factory=dict)


class SkillsFS:
    """A minimal, filesystem-like container for skills.

    Paths are always relative to the FS root and use POSIX separators.
    """

    def __init__(self, root: Directory | None = None) -> None:
        self._root = root or Directory()

    @property
    def root(self) -> Directory:
        return self._root

    def exists(self, path: str) -> bool:
        try:
            self._get_node(path)
        except KeyError:
            return False
        return True

    def is_file(self, path: str) -> bool:
        try:
            return isinstance(self._get_node(path), File)
        except KeyError:
            return False

    def is_dir(self, path: str) -> bool:
        try:
            return isinstance(self._get_node(path), Directory)
        except KeyError:
            return False

    def read_bytes(self, path: str) -> bytes:
        node = self._get_node(path)
        if not isinstance(node, File):
            raise IsADirectoryError(path)
        return node.content

    def read_text(self, path: str, *, encoding: str = "utf-8") -> str:
        node = self._get_node(path)
        if not isinstance(node, File):
            raise IsADirectoryError(path)
        return node.read_text(encoding=encoding)

    def write_bytes(self, path: str, content: bytes) -> None:
        normalized = normalize_rel_path(path)
        parent, name = self._split_parent(normalized)
        directory = self._mkdirs(parent)
        directory.entries[name] = File(content=content)

    def write_text(self, path: str, text: str, *, encoding: str = "utf-8") -> None:
        self.write_bytes(path, text.encode(encoding))

    def mkdir(self, path: str) -> None:
        normalized = normalize_rel_path(path)
        self._mkdirs(normalized)

    def listdir(self, path: str) -> list[str]:
        node = self._get_node(path)
        if not isinstance(node, Directory):
            raise NotADirectoryError(path)
        return sorted(node.entries.keys())

    def iter_files(self) -> Iterator[tuple[str, File]]:
        """Yield (path, File) for all files in the filesystem."""

        def walk(prefix: str, directory: Directory) -> Iterator[tuple[str, File]]:
            for name, child in directory.entries.items():
                child_path = f"{prefix}/{name}" if prefix else name
                if isinstance(child, File):
                    yield child_path, child
                else:
                    yield from walk(child_path, child)

        yield from walk("", self._root)

    def iter_skill_dirs(self) -> Iterator[str]:
        """Yield directories that contain a SKILL.md file."""

        def walk(prefix: str, directory: Directory) -> Iterator[str]:
            if "SKILL.md" in directory.entries and isinstance(
                directory.entries["SKILL.md"], File
            ):
                yield prefix
            for name, child in directory.entries.items():
                if isinstance(child, Directory):
                    child_path = f"{prefix}/{name}" if prefix else name
                    yield from walk(child_path, child)

        yield from walk("", self._root)

    @classmethod
    def from_disk(
        cls,
        root: Path,
        *,
        include_hidden: bool = False,
        max_file_bytes: int | None = None,
        max_files: int = 10000,
    ) -> SkillsFS:
        """Load a directory tree into an in-memory filesystem."""
        if not root.exists() or not root.is_dir():
            raise ValueError(f"root must be an existing directory: {root}")

        fs = cls()
        resolved_root = root.resolve()
        file_count = 0

        for path in root.rglob("*"):
            if path.is_dir():
                continue

            try:
                resolved_path = path.resolve(strict=True)
            except OSError:
                continue

            if not resolved_path.is_relative_to(resolved_root):
                raise ValueError(
                    f"Symlink escape detected: {path} points outside of skills root {root}"
                )

            rel = path.relative_to(root).as_posix()
            if not include_hidden and any(part.startswith(".") for part in path.parts):
                continue

            file_count += 1
            if file_count > max_files:
                raise ValueError(
                    f"Filesystem traversal exceeded max_files limit ({max_files})"
                )

            content = resolved_path.read_bytes()
            if max_file_bytes is not None and len(content) > max_file_bytes:
                raise ValueError(f"file too large: {rel} ({len(content)} bytes)")
            fs.write_bytes(rel, content)
        return fs

    def _split_parent(self, normalized: str) -> tuple[str, str]:
        if "/" not in normalized:
            return "", normalized
        parent, name = normalized.rsplit("/", 1)
        return parent, name

    def _mkdirs(self, path: str) -> Directory:
        if not path:
            return self._root
        normalized = normalize_rel_path(path)
        current: Directory = self._root
        for segment in normalized.split("/"):
            existing = current.entries.get(segment)
            if existing is None:
                child = Directory()
                current.entries[segment] = child
                current = child
                continue
            if isinstance(existing, File):
                raise NotADirectoryError(f"{segment} is a file")
            current = existing
        return current

    def _get_node(self, path: str) -> Directory | File:
        if path == "":
            return self._root
        normalized = normalize_rel_path(path)
        current: Directory | File = self._root
        for segment in normalized.split("/"):
            if not isinstance(current, Directory):
                raise KeyError(path)
            current = current.entries[segment]
        return current


class OverlayFS:
    """Read-through overlay on top of a base filesystem."""

    def __init__(self, base: SkillsFS, overlay: SkillsFS | None = None) -> None:
        self.base = base
        self.overlay = overlay or SkillsFS()

    def read_text(self, path: str, *, encoding: str = "utf-8") -> str:
        if self.overlay.exists(path):
            return self.overlay.read_text(path, encoding=encoding)
        return self.base.read_text(path, encoding=encoding)

    def exists(self, path: str) -> bool:
        return self.overlay.exists(path) or self.base.exists(path)

    def write_text(self, path: str, text: str, *, encoding: str = "utf-8") -> None:
        self.overlay.write_text(path, text, encoding=encoding)

    def iter_files(self) -> Iterator[tuple[str, File]]:
        base_map = {path: file for path, file in self.base.iter_files()}
        for path, file in self.overlay.iter_files():
            base_map[path] = file
        for path in sorted(base_map.keys()):
            yield path, base_map[path]

    def iter_skill_dirs(self) -> Iterator[str]:
        """Yield directories that contain a SKILL.md file (base + overlay)."""
        dirs = set(self.base.iter_skill_dirs()) | set(self.overlay.iter_skill_dirs())
        for d in sorted(dirs):
            yield d

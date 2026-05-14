"""Shared I/O helpers for the gepa CLI.

The content-file rule (pydanticaigepa-dec-dmk) means every text-content input
goes through ``--*-file PATH`` (or ``-`` for stdin), and every text-content
output can be sent to ``--output-file PATH`` (or stdout). These helpers
centralize that convention.
"""

from __future__ import annotations

import sys
from pathlib import Path


STDIN_SENTINEL = "-"


def read_content_file(path: Path) -> str:
    """Read text content from a file path or stdin (when path == ``-``)."""
    if str(path) == STDIN_SENTINEL:
        return sys.stdin.read()
    return path.read_text(encoding="utf-8")


def write_content_file(path: Path | None, content: str) -> None:
    """Write text content to a file path or stdout (when path is None or ``-``)."""
    if path is None or str(path) == STDIN_SENTINEL:
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

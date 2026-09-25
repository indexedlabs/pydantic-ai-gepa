"""Shared helpers for CLI output assertions."""

import re


_RICH_DECORATION = re.compile(r"\x1b\[[0-9;:]*m|[\u2500-\u257f]")


def normalize_cli_output(text: str) -> str:
    """Remove ANSI styling, Rich borders, and wrapping for text comparisons."""
    return "".join(_RICH_DECORATION.sub("", text).split())

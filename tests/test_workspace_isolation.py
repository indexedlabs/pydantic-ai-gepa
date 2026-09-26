"""The test runner cannot write the developer's real GEPA workspace."""

import pytest
from pathlib import Path

from pydantic_ai_gepa.cli.safe_git import Repository

from tests.conftest import _REAL_GEPA


def test_real_home_workspace_writes_are_refused() -> None:
    with pytest.raises(AssertionError, match="real home GEPA workspace"):
        (_REAL_GEPA / "gepa.toml").write_text("must never be written")


def test_temporary_project_does_not_inherit_host_git_repo(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a git repository"):
        Repository.discover(tmp_path)

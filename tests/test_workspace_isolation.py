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


def test_isolated_home_does_not_claim_sandbox_test_home(tmp_path: Path) -> None:
    fake_home = tmp_path.parent / (tmp_path.name + "-home")
    fake_home.mkdir()
    assert Path.home() != fake_home
    assert Path.home().is_dir()
    assert Path.home().parent == tmp_path.parent

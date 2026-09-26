import asyncio
from collections.abc import Iterator
import os
from pathlib import Path
import sys
from typing import Any
import pytest


_REAL_GEPA = (Path.home() / ".gepa").resolve()


def _guard_real_workspace(event: str, args: tuple[Any, ...]) -> None:
    if event == "open":
        flags = args[2]
        if not isinstance(flags, int) or not flags & (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        ):
            return
        paths = args[:1]
    elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.truncate"}:
        paths = args[:1]
    elif event in {"os.rename", "os.link", "os.symlink"}:
        paths = args[:2]
    else:
        return
    for raw in paths:
        if isinstance(raw, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(raw)).resolve()
            if path == _REAL_GEPA or path.is_relative_to(_REAL_GEPA):
                raise AssertionError(
                    "Test attempted to write the real home GEPA workspace"
                )


sys.addaudithook(_guard_real_workspace)


@pytest.fixture(autouse=True)
def isolated_user_environment(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Keep HOME inside pytest's sandbox without adding files to candidate trees.
    home = tmp_path.parent / (tmp_path.name + "-home")
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("GEPA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    from pydantic_ai_gepa.cli.safe_git import Repository

    discover = Repository.discover
    boundary = tmp_path_factory.getbasetemp().resolve()

    def discover_test_repository(cls: type[Repository], start: Path) -> Repository:
        repository = discover(start)
        # --basetemp may be nested in a developer checkout. Tests that model
        # non-Git projects must not inherit that unrelated repository.
        if not repository.root.is_relative_to(boundary):
            raise FileNotFoundError("not a git repository")
        return repository

    monkeypatch.setattr(Repository, "discover", classmethod(discover_test_repository))


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def event_loop() -> Iterator[None]:
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    yield
    new_loop.close()

"""Directory opens preserve symlink refusal without reading lane ancestors."""

import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

from pydantic_ai_gepa.cli import safe_git


@pytest.fixture(params=["native", "other-platform", "missing-flag"])
def directory_backend(request, monkeypatch):
    if request.param == "native":
        if sys.platform != "darwin" or not hasattr(os, "O_NOFOLLOW_ANY"):
            pytest.skip("Native directory lookup requires macOS O_NOFOLLOW_ANY")
    elif request.param == "other-platform":
        monkeypatch.setattr(safe_git.sys, "platform", "linux")
    else:
        monkeypatch.setattr(safe_git.sys, "platform", "darwin")
        monkeypatch.delattr(safe_git.os, "O_NOFOLLOW_ANY", raising=False)


def test_directory_fd_refuses_relative_directory(
    tmp_path, directory_backend, monkeypatch
):
    (tmp_path / "plain").mkdir()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="must be absolute"):
        with safe_git._directory_fd(Path("plain")):
            pytest.fail("Opened a relative directory")


def test_directory_fd_opens_and_closes_plain_directory(tmp_path, directory_backend):
    directory = tmp_path.resolve() / "plain" / "child"
    directory.mkdir(parents=True)
    with safe_git._directory_fd(directory) as fd:
        assert os.fstat(fd) == directory.stat()
    with pytest.raises(OSError) as error:
        os.fstat(fd)
    assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("ancestor", [False, True], ids=["final", "ancestor"])
def test_directory_fd_refuses_symlinks(tmp_path, directory_backend, ancestor):
    root = tmp_path.resolve()
    (root / "plain" / "child").mkdir(parents=True)
    link = root / "link"
    link.symlink_to(root / "plain", target_is_directory=True)
    with pytest.raises(OSError) as error:
        with safe_git._directory_fd(link / "child" if ancestor else link):
            pytest.fail("Opened a directory through a symlink")
    assert error.value.errno in {errno.ELOOP, errno.ENOTDIR}


def test_real_reflector_sandbox_discovers_lane_without_ancestor_reads():
    if sys.platform != "darwin":
        pytest.skip("Real reflector isolation requires macOS Seatbelt")
    codex = shutil.which("codex")
    if codex is None:
        pytest.skip("Real reflector isolation requires codex on PATH")
    if not hasattr(os, "O_NOFOLLOW_ANY"):
        pytest.skip("Real reflector isolation requires Python O_NOFOLLOW_ANY")

    # :minimal reads system temp trees, so keep this layout under the checkout.
    with tempfile.TemporaryDirectory(
        prefix=".tmp-safe-git-", dir=Path(__file__).resolve().parents[2]
    ) as temporary:
        root = Path(temporary).resolve()
        run = root / "run"
        lane = root / "run.lanes" / "id" / "lane-1"
        heldout = root / "run.heldout"
        home = root / "codex-home"
        user_home = root / "user-home"
        for directory in (run, lane / ".git" / "objects", lane / "sub", heldout, home):
            directory.mkdir(parents=True)
        user_home.mkdir()
        # Load the actual stdlib-only module without importing the full package.
        shutil.copyfile(safe_git.__file__, lane / "safe_git.py")
        runtime = Path(sys.base_prefix).resolve()
        grants = {
            ":minimal": "read",
            str(lane): "write",
            str(run): "write",
            str(heldout): "deny",
            str(runtime): "read",
        }
        (home / "config.toml").write_text(
            "[permissions.gepa-reflector.filesystem]\n"
            + "".join(
                f"{json.dumps(path)} = {json.dumps(mode)}\n"
                for path, mode in grants.items()
            )
            + "\n[permissions.gepa-reflector.network]\nenabled = false\n"
        )
        # The suite's isolated HOME is under temp, which :minimal allows.
        # Probe a synthetic HOME outside that grant, never the operator's home.
        denied = [root, user_home, heldout]
        if Path("/Users").is_dir():
            denied.append(Path("/Users"))
        probe = f"""
import errno
import sys
from pathlib import Path
sys.path.insert(0, {str(lane)!r})
from safe_git import Repository, _directory_fd
lane = Path({str(lane)!r})
assert Repository.discover(lane / 'sub').root == lane
with _directory_fd(Path({str(run)!r})):
    pass
for name in {list(map(str, denied))!r}:
    try:
        with _directory_fd(Path(name)):
            pass
    except OSError as error:
        assert error.errno == errno.EPERM, (name, error)
    else:
        raise AssertionError('Sandbox allowed directory read: ' + name)
print('lane discovered; run opened; ancestor, home and heldout reads denied')
"""
        result = subprocess.run(
            [
                codex,
                "sandbox",
                "-P",
                "gepa-reflector",
                "-C",
                str(lane),
                "--",
                str(Path(sys.executable).resolve()),
                "-I",
                "-S",
                "-B",
                "-c",
                probe,
            ],
            env={**os.environ, "CODEX_HOME": str(home), "HOME": str(user_home)},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert (
            "lane discovered; run opened; ancestor, home and heldout reads denied"
            in result.stdout
        )

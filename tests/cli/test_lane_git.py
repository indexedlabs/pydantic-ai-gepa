"""Launch checks read hostile metadata without invoking Git."""

import os
import shutil
import subprocess

import pytest

from pydantic_ai_gepa.cli import lane_repositories as repos, layout, safe_git


@pytest.fixture
def lane(tmp_path, monkeypatch, request):
    seed = tmp_path / "seed"
    repos._init(seed, getattr(request, "param", "sha1"))
    (seed / "api").mkdir()
    (seed / "api/prompt.txt").write_text("seed")
    repos._command(seed, "add", ".")
    repos._command(seed, "commit", "-m", "Seed")
    monkeypatch.setattr(layout, "_explicit_gepa_dirname", str(tmp_path / "public"))
    monkeypatch.delenv("GEPA_HELDOUT_DATASET", raising=False)
    repositories = repos.initialize(seed / "api", "run1", seed / "api")
    lane = repositories.create_lane("lane-1", repositories.seed, "gepa/lane-1")

    def no_command(*args, **kwargs):
        pytest.fail("The lane Git check must only read files")

    monkeypatch.setattr(subprocess, "Popen", no_command)
    return lane


@pytest.mark.parametrize("lane", ["sha1", "sha256"], indirect=True)
def test_library_created_lane_passes(lane):
    assert safe_git.refuse_executable_lane_git(lane / "api") == lane
    hooks = lane / ".git/hooks"
    hooks.mkdir()
    (hooks / "pre-commit.sample").write_text("sample")
    assert safe_git.refuse_executable_lane_git(lane) == lane


@pytest.mark.parametrize(
    "content",
    [
        b"[core]\nfsmonitor = /payload\n",
        b"[core]\nhooksPath = /payload\n",
        b'[filter "x"]\nclean = /payload\n',
        b"[include]\npath = /payload\n",
        b'[includeIf "gitdir:/"]\npath = /payload\n',
        b"[alias]\nstatus = !payload\n",
        b"[core]\nbare = true\n",
        b"[extensions]\nobjectformat = unknown\n",
        b'[user]\nname = "quoted"\n',
        b"[user]\nname = escaped\\n\n",
        b"[user]\nname = continued\\\nnext\n",
        b"[user]\nname = value # comment\n",
        b"[user]\nname = value ; comment\n",
        b"[user]\nname\n",
        b"name = no-section\n",
        b"[core] filemode = true\n",
        b"[core]\nfilemode = true\r\n",
        b"[user]\nname = nul\0\n",
        b"[user]\nname = non-ascii\xff\n",
    ],
)
def test_config_syntax_and_keys_fail_closed(lane, content):
    (lane / ".git/config").write_bytes(content)
    with pytest.raises(OSError):
        safe_git.refuse_executable_lane_git(lane)


@pytest.mark.parametrize(
    "kind", ["symlink", "hardlink", "directory", "fifo", "missing"]
)
def test_config_must_be_regular_unlinked_file(lane, kind):
    config = lane / ".git/config"
    saved = lane / "saved-config"
    config.rename(saved)
    if kind == "symlink":
        config.symlink_to(saved)
    elif kind == "hardlink":
        os.link(saved, config)
    elif kind == "directory":
        config.mkdir()
    elif kind == "fifo":
        os.mkfifo(config)
    with pytest.raises(OSError):
        safe_git.refuse_executable_lane_git(lane)


@pytest.mark.parametrize("name", ["hooks", "info", "objects", "objects/info"])
def test_metadata_directory_links_refused(lane, name):
    path = lane / ".git" / name
    if path.exists():
        shutil.rmtree(path)
    target = lane / "redirect"
    target.mkdir()
    path.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        safe_git.refuse_executable_lane_git(lane)


@pytest.mark.parametrize("name", safe_git._LANE_REDIRECTS)
def test_redirect_metadata_refused(lane, name):
    path = lane / ".git" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(lane / "missing")  # Dangling redirects must also be refused.
    with pytest.raises(OSError):
        safe_git.refuse_executable_lane_git(lane)


@pytest.mark.parametrize("kind", ["active", "symlink-sample", "directory-sample"])
def test_hooks_refused(lane, kind):
    hooks = lane / ".git/hooks"
    hooks.mkdir()
    if kind == "active":
        (hooks / "post-checkout").write_text("#!/bin/sh\nexit 0\n")
    elif kind == "symlink-sample":
        (hooks / "post-checkout.sample").symlink_to(lane / "missing")
    else:
        (hooks / "post-checkout.sample").mkdir()
    with pytest.raises(OSError):
        safe_git.refuse_executable_lane_git(lane)


@pytest.mark.parametrize(
    "nested,kind",
    [
        (True, "directory"),
        (True, "gitfile"),
        (True, "symlink"),
        (False, "gitfile"),
        (False, "symlink"),
    ],
)
def test_git_markers_refused(lane, nested, kind):
    if nested:
        path = lane / "api/nested/.git"
        path.parent.mkdir()
    else:
        path = lane / ".git"
        shutil.rmtree(path)
    if kind == "directory":
        path.mkdir()
    elif kind == "gitfile":
        path.write_text("gitdir: /outside\n")
    else:
        path.symlink_to(lane / "missing")
    with pytest.raises(OSError):
        safe_git.refuse_executable_lane_git(lane)

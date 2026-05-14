"""Tests for `pydantic_ai_gepa.cli.layout`."""

from __future__ import annotations

import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pydantic_ai_gepa.cli.layout import (
    GepaConfig,
    GepaConfigError,
    candidate_dir,
    components_dir,
    config_path,
    ensure_layout,
    gepa_dir,
    is_run_id,
    journal_path,
    latest_run_id,
    minibatch_path,
    new_run_id,
    pareto_log_path,
    proposal_dir,
    repo_root,
    resolve_agent,
    run_dir,
    runs_dir,
    staged_dir,
    traces_dir,
    write_default_config,
)


def test_config_parse_minimal(tmp_path: Path) -> None:
    (tmp_path / ".gepa").mkdir()
    cfg_path = tmp_path / ".gepa" / "gepa.toml"
    cfg_path.write_text(
        textwrap.dedent("""
        agent = "pkg.agents:my_agent"
    """).strip(),
        encoding="utf-8",
    )

    cfg = GepaConfig.load(cfg_path)
    assert cfg.agent == "pkg.agents:my_agent"
    assert cfg.dataset == ".gepa/dataset.jsonl"
    assert cfg.defaults == {}


def test_config_parse_with_defaults(tmp_path: Path) -> None:
    (tmp_path / ".gepa").mkdir()
    cfg_path = tmp_path / ".gepa" / "gepa.toml"
    cfg_path.write_text(
        textwrap.dedent("""
        agent = "pkg.agents:other"
        dataset = "data/cases.jsonl"

        [defaults]
        minibatch_size = 5
        max_iterations = 20
    """).strip(),
        encoding="utf-8",
    )

    cfg = GepaConfig.load(cfg_path)
    assert cfg.agent == "pkg.agents:other"
    assert cfg.dataset == "data/cases.jsonl"
    assert cfg.defaults == {"minibatch_size": 5, "max_iterations": 20}


def test_config_missing_agent(tmp_path: Path) -> None:
    cfg = tmp_path / "gepa.toml"
    cfg.write_text('dataset = "x.jsonl"', encoding="utf-8")
    with pytest.raises(GepaConfigError, match="Missing required key 'agent'"):
        GepaConfig.load(cfg)


def test_config_invalid_agent(tmp_path: Path) -> None:
    cfg = tmp_path / "gepa.toml"
    cfg.write_text('agent = "no_colon_here"', encoding="utf-8")
    with pytest.raises(GepaConfigError, match="Invalid 'agent'"):
        GepaConfig.load(cfg)


def test_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(GepaConfigError, match="No gepa.toml"):
        GepaConfig.load(tmp_path / ".gepa" / "gepa.toml")


def test_write_default_config_idempotent(tmp_path: Path) -> None:
    path = write_default_config("pkg:agent", root=tmp_path)
    assert path == config_path(tmp_path)
    assert path.read_text(encoding="utf-8").startswith('agent = "pkg:agent"')

    with pytest.raises(GepaConfigError, match="already exists"):
        write_default_config("pkg:agent", root=tmp_path)

    # Force overwrite works.
    write_default_config("pkg:other", root=tmp_path, force=True)
    cfg = GepaConfig.load(config_path(tmp_path))
    assert cfg.agent == "pkg:other"


def test_path_helpers(tmp_path: Path) -> None:
    assert gepa_dir(tmp_path) == tmp_path / ".gepa"
    assert components_dir(tmp_path) == tmp_path / ".gepa" / "components"
    assert staged_dir(tmp_path) == tmp_path / ".gepa" / "staged"
    assert journal_path(tmp_path) == tmp_path / ".gepa" / "journal.jsonl"
    assert runs_dir(tmp_path) == tmp_path / ".gepa" / "runs"

    run = "20260101T120000Z-deadbeef"
    assert run_dir(run, tmp_path) == tmp_path / ".gepa" / "runs" / run
    assert (
        minibatch_path(run, "mb-1", tmp_path)
        == tmp_path / ".gepa" / "runs" / run / "minibatches" / "mb-1.json"
    )
    assert (
        candidate_dir(run, tmp_path) == tmp_path / ".gepa" / "runs" / run / "candidates"
    )
    assert (
        proposal_dir(run, tmp_path) == tmp_path / ".gepa" / "runs" / run / "proposals"
    )
    assert (
        traces_dir(run, "case-7", tmp_path)
        == tmp_path / ".gepa" / "runs" / run / "traces" / "case-7"
    )
    assert (
        pareto_log_path(run, tmp_path)
        == tmp_path / ".gepa" / "runs" / run / "pareto.jsonl"
    )


def test_ensure_layout_idempotent(tmp_path: Path) -> None:
    base = ensure_layout(tmp_path)
    assert base.is_dir()
    for sub in (components_dir(tmp_path), staged_dir(tmp_path), runs_dir(tmp_path)):
        assert sub.is_dir()
    assert journal_path(tmp_path).exists()

    # Idempotent
    ensure_layout(tmp_path)
    assert journal_path(tmp_path).exists()


def test_new_run_id_sortable() -> None:
    a = new_run_id(now=datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc))
    b = new_run_id(now=datetime(2026, 5, 14, 12, 0, 1, tzinfo=timezone.utc))
    assert a < b
    assert is_run_id(a)
    assert is_run_id(b)
    assert not is_run_id("not-a-run-id")


def test_latest_run_id(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    assert latest_run_id(tmp_path) is None
    older = "20260101T000000Z-aaaaaaaa"
    newer = "20260102T000000Z-bbbbbbbb"
    (runs_dir(tmp_path) / older).mkdir()
    # Different filesystem tick.
    time.sleep(0.01)
    (runs_dir(tmp_path) / newer).mkdir()
    # A junk directory that doesn't match the run-id format must be ignored.
    (runs_dir(tmp_path) / "junk").mkdir()
    assert latest_run_id(tmp_path) == newer


def test_repo_root_finds_gepa_dir(tmp_path: Path) -> None:
    (tmp_path / ".gepa").mkdir()
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert repo_root(nested) == tmp_path.resolve()


def test_repo_root_falls_back_to_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").touch()
    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)
    assert repo_root(nested) == tmp_path.resolve()


def test_resolve_agent_success() -> None:
    cfg = GepaConfig(agent="pydantic_ai_gepa.cli.layout:GEPA_DIRNAME")
    assert resolve_agent(cfg) == ".gepa"


def test_resolve_agent_missing_module() -> None:
    cfg = GepaConfig(agent="nonexistent.pkg.deep:thing")
    with pytest.raises(GepaConfigError, match="Could not import"):
        resolve_agent(cfg)


def test_resolve_agent_missing_attr() -> None:
    cfg = GepaConfig(agent="pydantic_ai_gepa.cli.layout:NO_SUCH_NAME")
    with pytest.raises(GepaConfigError, match="has no attribute"):
        resolve_agent(cfg)


def test_resolve_agent_invalid_ref() -> None:
    cfg = GepaConfig(agent="no_colon")
    with pytest.raises(GepaConfigError, match="Invalid agent ref"):
        resolve_agent(cfg)

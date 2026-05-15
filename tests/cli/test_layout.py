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
    load_dotenv,
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


# ---------- --gepa-dir override ----------


@pytest.fixture
def _reset_gepa_dirname():
    """Clear the explicit override before and after each test that touches it."""
    from pydantic_ai_gepa.cli.layout import set_gepa_dirname

    set_gepa_dirname(None)
    yield
    set_gepa_dirname(None)


def test_current_gepa_dirname_default_is_dot_gepa(
    monkeypatch: pytest.MonkeyPatch, _reset_gepa_dirname: None
) -> None:
    from pydantic_ai_gepa.cli.layout import current_gepa_dirname

    monkeypatch.delenv("GEPA_DIR", raising=False)
    assert current_gepa_dirname() == ".gepa"


def test_set_gepa_dirname_overrides_default(
    monkeypatch: pytest.MonkeyPatch, _reset_gepa_dirname: None
) -> None:
    from pydantic_ai_gepa.cli.layout import (
        current_gepa_dirname,
        gepa_dir,
        set_gepa_dirname,
    )

    monkeypatch.delenv("GEPA_DIR", raising=False)
    set_gepa_dirname(".gepa.personalize")
    assert current_gepa_dirname() == ".gepa.personalize"
    assert gepa_dir(Path("/tmp/repo")) == Path("/tmp/repo/.gepa.personalize")


def test_env_var_falls_back_when_no_explicit_override(
    monkeypatch: pytest.MonkeyPatch, _reset_gepa_dirname: None
) -> None:
    from pydantic_ai_gepa.cli.layout import current_gepa_dirname

    monkeypatch.setenv("GEPA_DIR", ".gepa.from-env")
    assert current_gepa_dirname() == ".gepa.from-env"


def test_explicit_override_beats_env_var(
    monkeypatch: pytest.MonkeyPatch, _reset_gepa_dirname: None
) -> None:
    from pydantic_ai_gepa.cli.layout import current_gepa_dirname, set_gepa_dirname

    monkeypatch.setenv("GEPA_DIR", ".gepa.from-env")
    set_gepa_dirname(".gepa.from-flag")
    assert current_gepa_dirname() == ".gepa.from-flag"


def test_repo_root_finds_custom_workspace_dirname(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _reset_gepa_dirname: None
) -> None:
    from pydantic_ai_gepa.cli.layout import set_gepa_dirname

    monkeypatch.delenv("GEPA_DIR", raising=False)
    (tmp_path / ".gepa.alt").mkdir()
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    set_gepa_dirname(".gepa.alt")
    assert repo_root(nested) == tmp_path.resolve()


def test_default_dataset_path_follows_active_dirname(
    monkeypatch: pytest.MonkeyPatch, _reset_gepa_dirname: None
) -> None:
    from pydantic_ai_gepa.cli.layout import default_dataset_path, set_gepa_dirname

    monkeypatch.delenv("GEPA_DIR", raising=False)
    assert default_dataset_path() == ".gepa/dataset.jsonl"
    set_gepa_dirname(".gepa.personalize")
    assert default_dataset_path() == ".gepa.personalize/dataset.jsonl"


def test_write_default_config_uses_active_dirname(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _reset_gepa_dirname: None
) -> None:
    from pydantic_ai_gepa.cli.layout import set_gepa_dirname

    monkeypatch.delenv("GEPA_DIR", raising=False)
    set_gepa_dirname(".gepa.personalize")
    path = write_default_config("pkg:agent", root=tmp_path)
    # File lands under the custom workspace, and the dataset default
    # reflects the active dirname rather than the literal ".gepa".
    assert path == tmp_path / ".gepa.personalize" / "gepa.toml"
    body = path.read_text(encoding="utf-8")
    assert 'dataset = ".gepa.personalize/dataset.jsonl"' in body


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


# ---------- .env loader ----------


def test_load_dotenv_no_file_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_dotenv(tmp_path) == {}


def test_load_dotenv_basic_key_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("FOO_KEY=hello\nBAR_KEY=world\n", encoding="utf-8")
    monkeypatch.delenv("FOO_KEY", raising=False)
    monkeypatch.delenv("BAR_KEY", raising=False)

    applied = load_dotenv(tmp_path)
    assert applied == {"FOO_KEY": "hello", "BAR_KEY": "world"}
    import os

    assert os.environ["FOO_KEY"] == "hello"
    assert os.environ["BAR_KEY"] == "world"


def test_load_dotenv_ignores_comments_and_blanks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "\n# This is a comment\n\nA=1\n  # also a comment\nB=2\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("A", raising=False)
    monkeypatch.delenv("B", raising=False)
    applied = load_dotenv(tmp_path)
    assert applied == {"A": "1", "B": "2"}


def test_load_dotenv_does_not_override_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("X=from_dotenv\n", encoding="utf-8")
    monkeypatch.setenv("X", "from_shell")
    applied = load_dotenv(tmp_path)
    assert applied == {}
    import os

    assert os.environ["X"] == "from_shell"


def test_load_dotenv_strips_matched_quotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        'DQ="quoted value"\nSQ=\'single\'\nMIXED="ok\n', encoding="utf-8"
    )
    for key in ("DQ", "SQ", "MIXED"):
        monkeypatch.delenv(key, raising=False)
    applied = load_dotenv(tmp_path)
    assert applied["DQ"] == "quoted value"
    assert applied["SQ"] == "single"
    # Mismatched quotes (only one of the pair) are kept verbatim.
    assert applied["MIXED"] == '"ok'


def test_load_dotenv_strips_leading_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("export FOO=bar\n", encoding="utf-8")
    monkeypatch.delenv("FOO", raising=False)
    applied = load_dotenv(tmp_path)
    assert applied == {"FOO": "bar"}


def test_load_dotenv_skips_invalid_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "no_equals_here\nGOOD=ok\n=missing_key\n", encoding="utf-8"
    )
    monkeypatch.delenv("GOOD", raising=False)
    applied = load_dotenv(tmp_path)
    assert applied == {"GOOD": "ok"}


def test_load_dotenv_interpolates_var_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BASE_URL", "https://api.example")
    monkeypatch.delenv("FULL_URL_UNQUOTED", raising=False)
    monkeypatch.delenv("FULL_URL_DQ", raising=False)
    monkeypatch.delenv("BRACED", raising=False)
    (tmp_path / ".env").write_text(
        "FULL_URL_UNQUOTED=$BASE_URL/v1\n"
        'FULL_URL_DQ="$BASE_URL/v2"\n'
        "BRACED=${BASE_URL}/v3\n",
        encoding="utf-8",
    )
    applied = load_dotenv(tmp_path)
    assert applied["FULL_URL_UNQUOTED"] == "https://api.example/v1"
    assert applied["FULL_URL_DQ"] == "https://api.example/v2"
    assert applied["BRACED"] == "https://api.example/v3"


def test_load_dotenv_single_quoted_is_literal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORLD", "earth")
    monkeypatch.delenv("LITERAL", raising=False)
    (tmp_path / ".env").write_text("LITERAL='$WORLD'\n", encoding="utf-8")
    applied = load_dotenv(tmp_path)
    assert applied["LITERAL"] == "$WORLD"


def test_load_dotenv_unknown_var_expands_to_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NOT_DEFINED", raising=False)
    monkeypatch.delenv("RESULT", raising=False)
    (tmp_path / ".env").write_text(
        "RESULT=prefix-$NOT_DEFINED-suffix\n", encoding="utf-8"
    )
    applied = load_dotenv(tmp_path)
    assert applied["RESULT"] == "prefix--suffix"

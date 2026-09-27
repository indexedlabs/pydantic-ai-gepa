"""Trusted text scoring uses real isolated workers, without a Seatbelt mock."""

from dataclasses import asdict, replace
import json
import stat
import textwrap
from types import SimpleNamespace

import pytest

from pydantic_ai_gepa.cli import scoring_sandbox as sandbox
from pydantic_ai_gepa.cli.layout import AcceptanceConfig, GepaConfig, GepaConfigError
from pydantic_ai_gepa.cli.scoring_scope import (
    _masked,
    check_private_storage,
    verify_candidate,
)
from tests.cli.test_git_candidate_cli import _git, _run, _run_payload
from tests.cli import test_scoring_sandbox
from tests.cli.test_scoring_sandbox import (
    commit_evaluator,
    initialize_private_run,
    score,
    trusted_config,
)

clean_environment = test_scoring_sandbox.clean_environment
git_repo = test_scoring_sandbox.git_repo
private = test_scoring_sandbox.private


def config(**overrides):
    acceptance = replace(
        trusted_config().acceptance,
        scoring="trusted_in_process",
        text_files=("score.txt",),
    )
    return replace(trusted_config(), acceptance=replace(acceptance, **overrides))


def pin(repo, monkeypatch, source=None):
    sha = commit_evaluator(
        repo,
        source
        or textwrap.dedent("""
            import json, os
            async def evaluate(case):
                return json.loads(os.environ["GEPA_CANDIDATE_COMPONENTS_JSON"])["score.txt"].strip()
        """),
    )
    monkeypatch.setenv("GEPA_HARNESS_SCORER_REVISION", sha)
    initialize_private_run(repo, "sandbox-test")
    from pydantic_ai_gepa.cli.harness_record import for_run

    record = for_run("sandbox-test", repo)
    assert record is not None
    record.write("@trusted_text_base", sha)
    return sha


def change(repo, path, content):
    (repo / path).write_text(content)
    _git(repo, "add", path)
    _git(repo, "commit", "-m", "Text candidate")
    return _git(repo, "rev-parse", "HEAD")


def test_config_roundtrip():
    cfg = config(
        component_files=("score.txt", "prompts.py"),
        python_string_symbols={"prompts.py": ["PROMPT"]},
        scope_verifier="task_pkg.evaluation:verify",
    )
    assert GepaConfig.from_dict(json.loads(json.dumps(asdict(cfg)))) == cfg
    assert AcceptanceConfig().scoring == "seatbelt"


@pytest.mark.parametrize(
    "field,value",
    [
        ("scoring", "off"),
        ("text_files", "score.txt"),
        ("python_string_symbols", {"x.py": []}),
        ("python_string_symbols", {"x.py": ["x", "x"]}),
        ("python_string_symbols", {"x.py": ["x.y"]}),
        ("scope_verifier", "missing_colon"),
    ],
)
def test_config_rejects_malformed_fields(field, value):
    with pytest.raises(GepaConfigError):
        AcceptanceConfig.from_dict({field: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"trusted_scorer": False},
        {"pinned_scorer": False},
        {"component_files": ("prompts.py",), "text_files": ()},
        {"component_files": ("prompts.py",), "text_files": ("prompts.py",)},
        {"component_files": ("run.sh",), "text_files": ("run.sh",)},
        {"text_files": ()},
        {"mode": "vector"},
        {"component_files": ("../score.txt",), "text_files": ("../score.txt",)},
    ],
)
def test_requires_trusted_text_scope(monkeypatch, overrides):
    monkeypatch.setenv("GEPA_HARNESS_SCORER_REVISION", "a" * 40)
    with pytest.raises(sandbox.ScoringSandboxError, match="text-only"):
        sandbox.require_supported(config(**overrides), "git")


def test_requires_git_and_pinned_revision(monkeypatch):
    with pytest.raises(sandbox.ScoringSandboxError, match="full commit SHA"):
        sandbox.require_supported(config(), "git")
    monkeypatch.setenv("GEPA_HARNESS_SCORER_REVISION", "a" * 40)
    with pytest.raises(sandbox.ScoringSandboxError, match="git candidates"):
        sandbox.require_supported(config(), "components")


@pytest.mark.parametrize(
    "path", ["task_pkg/undeclared.py", "outside.txt", ".gitignore"]
)
def test_refuses_undeclared_candidate_changes(git_repo, private, monkeypatch, path):
    pin(git_repo, monkeypatch)
    sha = change(git_repo, path, "raise RuntimeError('must not import')\n")
    with pytest.raises(
        sandbox.ScoringSandboxError, match="declared text scope|untracked"
    ):
        score(git_repo, sha, config=config())


@pytest.mark.parametrize(
    "source",
    [
        "PROMPT = 3\n",
        'PROMPT = str("good")\n',
        'PROMPT = f"good"\n',
        'PROMPT = "good"\nraise RuntimeError("extra code")\n',
        'PROMPT = "good"\nCOUNT = 2\n',
    ],
)
def test_refuses_code_edits(git_repo, private, monkeypatch, source):
    (git_repo / "prompts.py").write_text('PROMPT = "bad"\nCOUNT = 1\n')
    _git(git_repo, "add", "prompts.py")
    pin(git_repo, monkeypatch)
    sha = change(git_repo, "prompts.py", source)
    cfg = config(
        component_files=("prompts.py",),
        text_files=(),
        python_string_symbols={"prompts.py": ["PROMPT"]},
    )
    with pytest.raises(sandbox.ScoringSandboxError, match="string symbols"):
        score(git_repo, sha, config=cfg)


def test_python_text_reaches_pinned_evaluator_without_import(
    git_repo, private, monkeypatch
):
    (git_repo / "prompts.py").write_text(
        'PROMPT: str = "bad"\nraise RuntimeError("never import")\n'
    )
    _git(git_repo, "add", "prompts.py")
    pin(
        git_repo,
        monkeypatch,
        textwrap.dedent("""
        import json, os
        async def evaluate(case):
            text = json.loads(os.environ["GEPA_CANDIDATE_COMPONENTS_JSON"])["prompts.py"]
            assert 'PROMPT: str = "good"' in text
            return "good"
    """),
    )
    sha = change(
        git_repo,
        "prompts.py",
        'PROMPT: str = "good"\nraise RuntimeError("never import")\n',
    )
    cfg = config(
        component_files=("prompts.py",),
        text_files=(),
        python_string_symbols={"prompts.py": ["PROMPT"]},
    )
    assert score(git_repo, sha, config=cfg)[0].score == 1


@pytest.mark.parametrize(
    "body",
    ["return False", 'raise RuntimeError("PRIVATE_HOOK_SENTINEL")', "return None"],
)
def test_scope_hook_fails_closed(git_repo, private, monkeypatch, body):
    sha = pin(
        git_repo,
        monkeypatch,
        f'def verify(*, components):\n    {body}\nasync def evaluate(case):\n    raise AssertionError("never evaluate")\n',
    )
    with pytest.raises(sandbox.ScoringSandboxError) as exc:
        score(git_repo, sha, config=config(scope_verifier="task_pkg.evaluation:verify"))
    assert "PRIVATE_HOOK_SENTINEL" not in str(exc.value)
    assert str(private.parent) not in str(exc.value)


def test_hook_uses_pinned_checkout_and_candidate_data(git_repo, private, monkeypatch):
    pin(
        git_repo,
        monkeypatch,
        textwrap.dedent("""
        import os
        from pathlib import Path
        def verify(*, components):
            assert Path("score.txt").read_text().strip() == "bad"
            return components["score.txt"].strip() == "good"
        async def evaluate(case):
            return "good"
    """),
    )
    sha = change(git_repo, "score.txt", "good")
    assert (
        score(
            git_repo, sha, config=config(scope_verifier="task_pkg.evaluation:verify")
        )[0].score
        == 1
    )


def test_local_services_and_private_scratch(git_repo, private, monkeypatch):
    sha = pin(
        git_repo,
        monkeypatch,
        textwrap.dedent("""
        import os, socket, stat
        from pathlib import Path
        async def evaluate(case):
            scratch = Path(os.environ["GEPA_HARNESS_PRIVATE_SCRATCH"])
            assert stat.S_IMODE(scratch.stat().st_mode) == 0o700
            assert scratch.stat().st_uid == os.getuid()
            (scratch / "service-data").write_text("private")
            with socket.socket() as server:
                server.bind(("127.0.0.1", 0))
                server.listen()
                with socket.create_connection(server.getsockname(), timeout=2) as client:
                    connection, _ = server.accept()
                    with connection:
                        client.sendall(b"local")
                        assert connection.recv(5) == b"local"
            assert os.environ["NO_PROXY"] == os.environ["no_proxy"] == "localhost,127.0.0.1,::1"
            return "good"
    """),
    )
    seen = []
    original = sandbox.child_environment

    def environment(scratch, port):
        seen.append(scratch)
        assert not scratch.resolve().is_relative_to(git_repo.resolve())
        assert not scratch.resolve().is_relative_to((git_repo / ".gepa").resolve())
        assert stat.S_IMODE(scratch.stat().st_mode) == 0o700
        return original(scratch, port)

    monkeypatch.setattr(sandbox, "child_environment", environment)
    assert score(git_repo, sha, config=config())[0].score == 1
    assert seen and all(not p.exists() for p in seen)


def test_validation_material_stays_private(git_repo, private, monkeypatch):
    sha = pin(
        git_repo,
        monkeypatch,
        textwrap.dedent("""
        import os
        from pathlib import Path
        async def evaluate(case):
            Path(os.environ["GEPA_TRACE_FILE"]).write_text("PRIVATE_TRACE_SENTINEL")
            print("PRIVATE_STDOUT_SENTINEL")
            return "good"
        def metric(case, output):
            from pydantic_ai_gepa.types import MetricResult
            return MetricResult(score=1.0, feedback="PRIVATE_FEEDBACK_SENTINEL")
    """),
    )
    cfg = replace(config(), metric="task_pkg.evaluation:metric")
    records = score(git_repo, sha, config=cfg)
    assert records[0].score == 1
    assert records[0].feedback is None
    assert records[0].payload == {}
    for path in git_repo.rglob("*"):
        if path.is_file() and ".git" not in path.parts and path.suffix != ".py":
            assert b"PRIVATE_" not in path.read_bytes()
    assert not list((private.parent / ".gepa-heldout/work").iterdir())


@pytest.mark.parametrize("trusted", [False, True])
def test_every_other_heldout_mode_requires_seatbelt(
    git_repo, private, monkeypatch, trusted
):
    sha = pin(git_repo, monkeypatch)

    def denied(*args):
        raise sandbox.ScoringSandboxError("Seatbelt sentinel")

    monkeypatch.setattr(sandbox, "sandbox_command", denied)
    cfg = (
        trusted_config()
        if trusted
        else GepaConfig(candidate_source="git", evaluate="task_pkg.evaluation:evaluate")
    )
    with pytest.raises(sandbox.ScoringSandboxError, match="Seatbelt sentinel"):
        score(git_repo, sha, config=cfg)
    assert score(git_repo, sha, config=config())[0].score == 0


def test_scratch_refuses_public_or_wrong_permissions(tmp_path):
    path = tmp_path / "scratch"
    path.mkdir(mode=0o700)
    with pytest.raises(sandbox.ScoringSandboxError, match="outside"):
        check_private_storage(path, [tmp_path])
    path.chmod(0o755)
    with pytest.raises(sandbox.ScoringSandboxError, match="0700"):
        check_private_storage(path, [])


def test_mask_requires_declared_symbols():
    for source in ['def f():\n X = "a"', 'Y = "a"']:
        with pytest.raises(ValueError):
            _masked(source, ["X"])


def test_mask_functions_classes_and_docstrings():
    source = '''
INSTRUCTIONS = "instructions"
def chat_receipt_instructions() -> str:
    return "receipt"
async def chat_event_creation_policy() -> str:
    return "event"
class Email:
    """email documentation"""
    unchanged = "frozen"
'''
    symbols = [
        "INSTRUCTIONS",
        "chat_receipt_instructions",
        "chat_event_creation_policy",
        "Email.__doc__",
    ]
    candidate = (
        source.replace('"receipt"', '"new receipt"')
        .replace('"event"', '"new event"')
        .replace("email documentation", "new email docs")
    )
    assert _masked(source, symbols) == _masked(candidate, symbols)
    assert _masked(source, symbols) != _masked(
        candidate.replace('"frozen"', '"changed"'), symbols
    )
    assert _masked(source, symbols) != _masked(
        candidate.replace('return "new receipt"', "return 1"), symbols
    )
    with pytest.raises(ValueError):
        _masked("class Email:\n    pass", ["Email.__doc__"])
    assert _masked('X = Y = "old"', ["Y"]) == _masked('X = Y = "new"', ["Y"])


@pytest.mark.parametrize("attack", ["import", "loaded_module"])
def test_candidate_import_guard_refuses_result(git_repo, private, monkeypatch, attack):
    (git_repo / "forbidden.py").write_text('raise RuntimeError("candidate executed")')
    _git(git_repo, "add", "forbidden.py")
    if attack == "import":
        attempt = f"""
    sys.path.insert(0, {str(git_repo)!r})
    try:
        import forbidden
    except ImportError:
        pass
    finally:
        sys.path.pop(0)
"""
    else:
        attempt = f"""
    module = types.ModuleType("forbidden")
    module.__file__ = {str(git_repo / "forbidden.py")!r}
    sys.modules["forbidden"] = module
"""
    sha = pin(
        git_repo,
        monkeypatch,
        "import sys, types\nasync def evaluate(case):\n"
        + attempt
        + '    return "good"\n',
    )
    with pytest.raises(sandbox.ScoringSandboxError):
        score(git_repo, sha, config=config())
    assert not list((private.parent / ".gepa-heldout/work").iterdir())


def test_untracked_file_refuses_score(git_repo, private, monkeypatch):
    sha = pin(git_repo, monkeypatch)
    (git_repo / "untracked.py").write_text("raise RuntimeError")
    with pytest.raises(sandbox.ScoringSandboxError, match="untracked"):
        score(git_repo, sha, config=config())


def test_export_seed_allows_omitted_scorer_files(git_repo, private, monkeypatch):
    from pydantic_evals import Case
    from pydantic_ai_gepa.cli import lane_repositories, harness_record
    from pydantic_ai_gepa.cli.spend import EvalSpendMeter

    pin(git_repo, monkeypatch)
    export = git_repo.parent / (git_repo.name + "-export")
    export.mkdir()
    (export / "score.txt").write_text("bad")
    _git(export, "init")
    _git(export, "config", "user.email", "tests@example.com")
    _git(export, "config", "user.name", "Tests")
    _git(export, "add", ".")
    _git(export, "commit", "-m", "History-free export")
    repositories = lane_repositories.initialize(git_repo, "sandbox-test", export)
    record = harness_record.for_run("sandbox-test", git_repo)
    assert record is not None
    record.write("state.json", json.dumps({"lane_repository_version": 1}))
    sha = change(export, "score.txt", "good")
    lane_repositories.import_commit(export, sha, repositories.repository, lane=False)
    candidate = repositories.candidate(sha)
    assert not (candidate / "task_pkg/evaluation.py").exists()
    assert (
        sandbox.score_cases(
            config=config(),
            project=candidate,
            scorer_project=git_repo,
            sha=sha,
            cases=[Case(name="test", inputs="test", expected_output="good")],
            validation=True,
            meter=EvalSpendMeter(
                "sandbox-test", git_repo, "eval-test", "training", None, None, 1, 1
            ),
        )[0].score
        == 1
    )
    bad = change(export, "undeclared.py", "raise RuntimeError")
    lane_repositories.import_commit(export, bad, repositories.repository, lane=False)
    with pytest.raises(sandbox.ScoringSandboxError, match="declared text scope"):
        verify_candidate(
            config(),
            repositories.candidate(bad),
            bad,
            git_repo,
            sandbox.scorer_revision(),
            repositories.seed,
        )


def test_export_seed_cannot_authorize_python_code_changes(
    git_repo, private, monkeypatch
):
    (git_repo / "prompts.py").write_text('PROMPT = "bad"\nCOUNT = 1\n')
    _git(git_repo, "add", "prompts.py")
    revision = pin(git_repo, monkeypatch)
    baseline = change(git_repo, "prompts.py", 'PROMPT = "bad"\nCOUNT = 2\n')
    cfg = config(
        component_files=("prompts.py",),
        text_files=(),
        python_string_symbols={"prompts.py": ["PROMPT"]},
    )
    with pytest.raises(sandbox.ScoringSandboxError, match="string symbols"):
        verify_candidate(cfg, git_repo, baseline, git_repo, revision, baseline)


def test_private_storage_excludes_lane_roots(git_repo, private, monkeypatch):
    from pydantic_ai_gepa.cli import lanes
    from pydantic_ai_gepa.cli.scoring_scope import private_roots

    monkeypatch.setattr(
        lanes,
        "load_all_lane_states",
        lambda *args: [
            SimpleNamespace(
                worktree_path=str(private.parent), candidate_project_path=None
            )
        ],
    )
    roots = private_roots(
        git_repo, git_repo, SimpleNamespace(root=git_repo, run_id="test")
    )
    with pytest.raises(sandbox.ScoringSandboxError, match="outside"):
        check_private_storage(private.parent, roots)


def test_run_start_records_candidate_baseline_separately(
    git_repo, private, monkeypatch
):
    from pydantic_ai_gepa.cli.harness_record import for_run

    scorer = pin(git_repo, monkeypatch)
    config_path = git_repo / ".gepa/gepa.toml"
    candidate = change(
        git_repo,
        ".gepa/gepa.toml",
        config_path.read_text()
        + """
[acceptance]
pinned_scorer = true
trusted_scorer = true
scoring = "trusted_in_process"
component_files = ["score.txt"]
text_files = ["score.txt"]
""",
    )
    assert candidate != scorer
    result = _run(
        "run",
        "start",
        "--size",
        "1",
        "--max-iterations",
        "16",
        "--acceptance-repetitions",
        "1",
        "--acceptance-paired-min-cases",
        "2",
    )
    assert result.exit_code == 0, (result.output, result.exception)
    run_id = _run_payload(result.output)["run_id"]
    record = for_run(run_id, git_repo)
    assert record is not None
    assert record.read("@trusted_text_base") == candidate
    assert not (git_repo / ".gepa/runs" / run_id / "@trusted_text_base").exists()

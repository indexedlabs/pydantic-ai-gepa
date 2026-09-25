"""Protocol tests run everywhere; adversarial tests require real OS enforcement."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import socketserver
import threading
from types import SimpleNamespace

import pytest
from pydantic_evals import Case

from pydantic_ai_gepa.cli import scoring_sandbox as sandbox
from pydantic_ai_gepa.cli.layout import AcceptanceConfig, GepaConfig
from pydantic_ai_gepa.cli.scoring_proxy import allowed_addresses, connect_proxy
from pydantic_ai_gepa.cli.spend import EvalSpendMeter
from tests.cli.test_git_candidate_cli import _git, _run, _run_payload
from tests.cli import test_git_candidate_cli

git_repo = test_git_candidate_cli.git_repo


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    from pydantic_ai_gepa.cli import layout

    monkeypatch.setattr(layout, "_explicit_gepa_dirname", None)
    for name in (
        "GEPA_DIR",
        "GEPA_HELDOUT_DATASET",
        "GEPA_HARNESS_ALLOWED_HOSTS",
        "GEPA_HARNESS_PASS_ENV",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def private(git_repo, monkeypatch):
    directory = git_repo.parent / (git_repo.name + "-private")
    directory.mkdir(mode=0o700)
    dataset = directory / "heldout.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "name": "CASE_SENTINEL_4833",
                "inputs": "INPUT_SENTINEL_4833",
                "expected_output": "good",
            }
        )
        + "\n"
    )
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(dataset))
    return dataset


@pytest.fixture
def protocol_backend(monkeypatch):
    """Exec boundary without OS enforcement, for non-adversarial fake data only."""
    monkeypatch.setattr(sandbox, "sandbox_command", lambda profile, command: command)

    @contextmanager
    def no_network(_addresses):
        yield 0

    monkeypatch.setattr(sandbox, "connect_proxy", no_network)
    monkeypatch.setattr(
        sandbox,
        "_process_cleanup",
        lambda scratch: SimpleNamespace(
            verify_worker=lambda pid: None, sweep=lambda: False
        ),
    )


@pytest.fixture
def real_backend():
    reason = sandbox.backend_unavailable_reason()
    if reason:
        if os.environ.get("GEPA_REQUIRE_SCORING_SANDBOX") == "1":
            pytest.fail(reason)
        pytest.skip(reason)


def commit_evaluator(repo, source):
    (repo / "task_pkg/evaluation.py").write_text(source)
    _git(repo, "add", "task_pkg/evaluation.py")
    _git(repo, "commit", "-m", "Candidate")
    return _git(repo, "rev-parse", "HEAD")


def score(repo, sha, *, validation=True, config=None, cases=None):
    cases = cases or [
        Case(
            name="CASE_SENTINEL_4833",
            inputs="INPUT_SENTINEL_4833",
            expected_output="good",
        )
    ]
    meter = EvalSpendMeter(
        "sandbox-test", repo, "eval-test", "training", None, None, len(cases), 1
    )
    return sandbox.score_cases(
        config=config
        or GepaConfig(candidate_source="git", evaluate="task_pkg.evaluation:evaluate"),
        project=repo,
        sha=sha,
        cases=cases,
        validation=validation,
        meter=meter,
    )


def test_no_backend_fails_closed_without_private_path(git_repo, private, monkeypatch):
    monkeypatch.setattr(sandbox.sys, "platform", "linux")
    result = _run("run", "start", "--size", "1")
    assert result.exit_code != 0
    assert "Seatbelt" in result.output, result.output
    assert str(private.parent) not in result.output


@pytest.mark.parametrize("syntax_error", [False, True])
def test_backend_probe_checks_full_profile_and_does_not_skip_syntax_errors(
    monkeypatch, syntax_error
):
    import subprocess

    monkeypatch.setattr(
        sandbox,
        "sandbox_command",
        lambda profile, command: ["sandbox-exec", "-p", profile, *command],
    )

    def run(command, **kwargs):
        profile = command[2]
        assert '(remote tcp "localhost:9")' in profile
        assert "(allow process-fork)" in profile
        assert "(require-not (subpath" in profile
        assert "(allow file-write*" in profile
        stderr = (
            b"sandbox-exec: host must be * or localhost in network address"
            if syntax_error
            else b"sandbox-exec: sandbox_apply: Operation not permitted"
        )
        return subprocess.CompletedProcess(command, 71, stdout=b"", stderr=stderr)

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    if syntax_error:
        with pytest.raises(
            sandbox.ScoringSandboxError, match="profile verification failed"
        ):
            sandbox.backend_unavailable_reason()
    else:
        assert "sandbox_apply refused" in sandbox.backend_unavailable_reason()


@pytest.mark.parametrize(
    "config",
    [
        GepaConfig(agent="evil:agent"),
        GepaConfig(
            candidate_source="git",
            evaluate="evil:evaluate",
            acceptance=AcceptanceConfig(mode="vector"),
        ),
        GepaConfig(
            candidate_source="git",
            evaluate="evil:evaluate",
            acceptance=AcceptanceConfig(pinned_scorer=True),
        ),
    ],
)
def test_unsupported_modes_fail_before_import(private, config):
    with pytest.raises(sandbox.ScoringSandboxError, match="requires git candidates"):
        sandbox.require_supported(config, config.candidate_source)


def test_parent_refuses_all_candidate_hooks(private):
    from pydantic_ai_gepa.cli.layout import resolve_module_attr

    with pytest.raises(sandbox.ScoringSandboxError, match="cannot be imported"):
        resolve_module_attr("definitely_not_importable:hook")


def test_parent_refuses_import_path_mutations(git_repo, private):
    import sys
    from pydantic_ai_gepa.cli.layout import (
        insert_repo_root_on_path,
        candidate_import_context,
    )
    from pydantic_ai_gepa.cli.run import _current_baseline_candidate_id

    original_path = list(sys.path)
    original_modules = dict(sys.modules)
    original_cwd = Path.cwd()
    assert (
        _current_baseline_candidate_id("git")
        == _git(git_repo, "rev-parse", "HEAD")[:12]
    )
    with pytest.raises(sandbox.ScoringSandboxError, match="import paths"):
        insert_repo_root_on_path(git_repo)
    with pytest.raises(sandbox.ScoringSandboxError, match="import paths"):
        with candidate_import_context(
            primary_project_root=git_repo,
            candidate_project_root=git_repo / "another-checkout",
            refs=["task_pkg.evaluation:evaluate"],
        ):
            pytest.fail("Harness entered a candidate import context")
    assert sys.path == original_path
    assert Path.cwd() == original_cwd
    assert all(
        sys.modules.get(name) is module for name, module in original_modules.items()
    )


def test_committed_shadow_module_is_not_imported_by_parent(
    git_repo, private, protocol_backend, monkeypatch
):
    import sys

    marker = private.parent / "parent-shadow-imported"
    (git_repo / "ctypes.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('parent executed candidate')\n"
        "raise RuntimeError('candidate shadow module')\n"
    )
    _git(git_repo, "add", "ctypes.py")
    _git(git_repo, "commit", "-m", "Shadow a lazy parent dependency")
    # The fixture models an earlier training invocation by adding the worktree.
    # Start this harness with only trusted import roots, as a fresh installed CLI.
    monkeypatch.setattr(
        sys,
        "path",
        [
            entry
            for entry in sys.path
            if entry and not Path(entry).resolve().is_relative_to(git_repo)
        ],
    )
    for name in tuple(sys.modules):
        if (
            name == "ctypes"
            or name.startswith("ctypes.")
            or name == "pydantic_ai_gepa.cli.scoring_processes"
        ):
            monkeypatch.delitem(sys.modules, name)
    imports = []

    def cleanup(scratch):
        # This is the production lazy import in _process_cleanup, reached only
        # after eval's former sys.path insertion. Native inspection is replaced
        # in this protocol test; importing its real dependency is not replaced.
        assert "ctypes" not in sys.modules
        from pydantic_ai_gepa.cli.scoring_processes import SandboxProcesses

        imports.append(SandboxProcesses)
        return SimpleNamespace(verify_worker=lambda pid: None, sweep=lambda: False)

    monkeypatch.setattr(sandbox, "_process_cleanup", cleanup)
    result = _run("eval", "--size", "1")
    assert result.exit_code == 0, (result.output, result.exception)
    assert imports
    assert not marker.exists()
    assert Path(sys.modules["ctypes"].__file__).resolve() != git_repo / "ctypes.py"
    assert str(git_repo) not in sys.path


def test_channel_detects_dead_worker_with_open_pipe():
    import subprocess
    import sys
    import time

    read_fd, write_fd = os.pipe()
    # Keep the writer open in this process, exactly as an inherited descendant
    # fd would. There is no EOF even after the worker exits.
    with os.fdopen(read_fd, "rb") as reader, os.fdopen(write_fd, "wb"):
        with subprocess.Popen(
            [sys.executable, "-c", "pass"], stdout=subprocess.PIPE
        ) as process:
            assert process.stdout is not None
            process.stdout.close()
            process.stdout = reader
            process.wait()
            started = time.monotonic()
            with pytest.raises(sandbox.ScoringSandboxError, match="exited"):
                sandbox.Channel(process).receive()
            assert time.monotonic() - started < 1


def test_survivor_sweep_excludes_other_processes_and_repeats(monkeypatch):
    from pydantic_ai_gepa.cli.scoring_processes import SandboxProcesses

    cleanup = SandboxProcesses.__new__(SandboxProcesses)
    cleanup.scratch, cleanup.outside = b"scratch", b"outside"
    passes = iter([[101, 102, 103], [102, 103, 104], [102, 103]])
    cleanup._pids = lambda: next(passes)
    cleanup._identity = lambda pid: (pid, 0)
    # 102 is unsandboxed (allows both); 103 is another child (denies both).
    cleanup._denied = lambda pid, path: pid != 102 and (
        path == b"outside" or pid == 103
    )
    killed = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    assert cleanup.sweep() is True
    assert killed == [(101, sandbox.signal.SIGKILL), (104, sandbox.signal.SIGKILL)]


def test_survivor_sweep_does_not_signal_reused_pid(monkeypatch):
    from pydantic_ai_gepa.cli.scoring_processes import SandboxProcesses

    cleanup = SandboxProcesses.__new__(SandboxProcesses)
    passes = iter([[101], []])
    identities = iter([(1, 0), (2, 0)])
    cleanup._pids = lambda: next(passes)
    cleanup._identity = lambda pid: next(identities)
    cleanup._matches = lambda pid: True
    monkeypatch.setattr(os, "kill", lambda *args: pytest.fail("Killed a reused PID"))
    assert cleanup.sweep() is True


@pytest.mark.parametrize("failure", ["load", "query", "enumerate", "signal"])
def test_survivor_inspection_fails_closed(monkeypatch, tmp_path, failure):
    from pydantic_ai_gepa.cli import scoring_processes as processes

    cleanup = processes.SandboxProcesses.__new__(processes.SandboxProcesses)
    if failure == "load":

        def unavailable(*args, **kwargs):
            raise OSError("sensitive diagnostic")

        monkeypatch.setattr(processes.ctypes, "CDLL", unavailable)

        def action():
            return processes.SandboxProcesses(tmp_path, tmp_path.parent)
    elif failure == "query":
        cleanup._check = lambda *args: -1

        def action():
            return cleanup._denied(101, b"scratch")
    elif failure == "enumerate":
        cleanup.uid = os.getuid()
        cleanup._list = lambda *args: -1
        action = cleanup._pids
    else:
        cleanup._pids = lambda: [101]
        cleanup._identity = lambda pid: (1, 0)
        cleanup._matches = lambda pid: True

        def refused(*args):
            raise PermissionError("sensitive diagnostic")

        monkeypatch.setattr(os, "kill", refused)
        action = cleanup.sweep
    with pytest.raises(sandbox.ScoringSandboxError) as error:
        action()
    assert "sensitive diagnostic" not in str(error.value)


def test_survivor_inspection_resizes_enumeration(monkeypatch):
    from pydantic_ai_gepa.cli import scoring_processes as processes

    cleanup = processes.SandboxProcesses.__new__(processes.SandboxProcesses)
    cleanup.uid = os.getuid()
    sizes = []

    def list_pids(kind, uid, buffer, size):
        assert kind == 4 and uid == os.getuid()
        sizes.append(size)
        if len(sizes) == 1:
            return size
        buffer[0] = os.getpid()
        buffer[1] = 101
        return 2 * processes.ctypes.sizeof(processes.ctypes.c_int)

    cleanup._list = list_pids
    assert cleanup._pids() == [os.getpid(), 101]
    assert sizes[1] == sizes[0] * 2


@pytest.mark.parametrize("status,same_uid", [(2, True), (5, True), (2, False)])
def test_survivor_identity_uses_start_time_and_excludes_zombies(status, same_uid):
    from pydantic_ai_gepa.cli import scoring_processes as processes

    cleanup = processes.SandboxProcesses.__new__(processes.SandboxProcesses)
    cleanup.uid = os.getuid()
    info = processes._BsdInfo()
    info.pid, info.uid, info.status = 101, cleanup.uid + (not same_uid), status
    info.start_sec, info.start_usec = 123, 456
    assert processes.ctypes.sizeof(info) == 136  # Darwin's 64-bit proc_bsdinfo ABI

    def pid_info(pid, flavor, arg, destination, size):
        assert pid == 101 and flavor == 3 and size == 136
        processes.ctypes.memmove(destination, processes.ctypes.byref(info), size)
        return size

    cleanup._info = pid_info
    assert cleanup._identity(101) == ((123, 456) if same_uid and status != 5 else None)


def test_worker_cleanup_failure_does_not_expose_command(
    git_repo, private, protocol_backend, monkeypatch
):
    sha = commit_evaluator(git_repo, "async def evaluate(case): return 'good'\n")
    original_popen = sandbox.subprocess.Popen

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        original_wait = process.wait

        def wait(*args, **kwargs):
            original_wait(*args, **kwargs)
            raise sandbox.subprocess.TimeoutExpired("PRIVATE_COMMAND_SENTINEL", 5)

        # Only replace the scoring worker's wait, not Git archive invocations.
        if kwargs.get("start_new_session"):
            process.wait = wait
        return process

    monkeypatch.setattr(sandbox.subprocess, "Popen", popen)
    with pytest.raises(
        sandbox.ScoringSandboxError, match="Cannot terminate the scoring worker"
    ) as error:
        score(git_repo, sha)
    assert "PRIVATE_COMMAND_SENTINEL" not in str(error.value)


def test_found_survivor_refuses_quality_score(
    git_repo, private, protocol_backend, monkeypatch
):
    monkeypatch.setattr(
        sandbox,
        "_process_cleanup",
        lambda scratch: SimpleNamespace(
            verify_worker=lambda pid: None,
            sweep=lambda: True,
        ),
    )
    sha = commit_evaluator(git_repo, "async def evaluate(case): return 'good'\n")
    with pytest.raises(sandbox.ScoringSandboxError, match="survivors were terminated"):
        score(git_repo, sha)
    assert not list((private.parent / ".gepa-heldout/work").iterdir())


def test_harness_never_loads_candidate_dotenv(git_repo, private, monkeypatch):
    monkeypatch.delenv("GEPA_HARNESS_ALLOWED_HOSTS", raising=False)
    (git_repo / ".env").write_text(
        "GEPA_HARNESS_ALLOWED_HOSTS=evil.example:443\n"
        "GEPA_HARNESS_PASS_ENV=TYPESAFE_API_KEY\n"
    )
    _run("run", "status")
    assert "GEPA_HARNESS_ALLOWED_HOSTS" not in os.environ
    assert "GEPA_HARNESS_PASS_ENV" not in os.environ


def test_environment_is_built_from_allowlisted_names(tmp_path, monkeypatch):
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", "/private/secret/dataset")
    monkeypatch.setenv("PYTHONPATH", "/evil")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "do-not-inherit")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    env = sandbox.child_environment(tmp_path, 1234)
    assert env["OPENAI_API_KEY"] == "provider-key"
    assert env["HOME"] == env["TMPDIR"] == str(tmp_path)
    assert (
        not {"GEPA_HELDOUT_DATASET", "PYTHONPATH", "AWS_SECRET_ACCESS_KEY"} & env.keys()
    )
    assert "/private/secret" not in json.dumps(env)


def test_harness_can_pass_named_extra_environment(private, tmp_path, monkeypatch):
    monkeypatch.setenv("GEPA_HARNESS_PASS_ENV", "TYPESAFE_API_KEY, CUSTOM_SETTING")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-typesafe-key")
    monkeypatch.setenv("CUSTOM_SETTING", "fake-setting")
    monkeypatch.setenv("UNNAMED_KEY", "do-not-pass")
    env = sandbox.child_environment(tmp_path, 1234)
    assert env["TYPESAFE_API_KEY"] == "fake-typesafe-key"
    assert env["CUSTOM_SETTING"] == "fake-setting"
    assert "GEPA_HARNESS_PASS_ENV" not in env
    assert "UNNAMED_KEY" not in env


@pytest.mark.parametrize(
    "name",
    [
        "GEPA_HELDOUT_DATASET",
        "GEPA_HARNESS_ALLOWED_HOSTS",
        "GEPA_HARNESS_PASS_ENV",
        "PYTHONPATH",
        "DYLD_INSERT_LIBRARIES",
        "HOME",
        "HTTP_PROXY",
        "invalid-name",
    ],
)
def test_harness_pass_env_refuses_reserved_names(private, tmp_path, monkeypatch, name):
    monkeypatch.setenv("GEPA_HARNESS_PASS_ENV", name)
    with pytest.raises(sandbox.ScoringSandboxError, match="reserved or invalid"):
        sandbox.child_environment(tmp_path, 1234)


@pytest.mark.parametrize("prefix, suffix", [("", ""), ("see ", "?query=yes")])
@pytest.mark.parametrize("name", ["TYPESAFE_API_KEY", "OPENAI_API_KEY"])
def test_harness_pass_env_refuses_private_path_values(
    private, tmp_path, monkeypatch, name, prefix, suffix
):
    monkeypatch.setenv("GEPA_HARNESS_PASS_ENV", name)
    monkeypatch.setenv(name, prefix + str(private) + suffix)
    with pytest.raises(
        sandbox.ScoringSandboxError, match="containing the held-out path"
    ) as error:
        sandbox.child_environment(tmp_path, 1234)
    assert str(private) not in str(error.value)


def test_harness_pass_env_refuses_resolved_private_path(tmp_path, private, monkeypatch):
    alias = tmp_path / "dataset-link"
    alias.symlink_to(private)
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(alias))
    monkeypatch.setenv("GEPA_HARNESS_PASS_ENV", "TYPESAFE_API_KEY")
    monkeypatch.setenv("TYPESAFE_API_KEY", "prefix=" + str(private.resolve()))
    with pytest.raises(
        sandbox.ScoringSandboxError, match="containing the held-out path"
    ):
        sandbox.child_environment(tmp_path, 1234)


def test_harness_pass_env_requires_a_present_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("GEPA_HARNESS_PASS_ENV", "MISSING_EXTRA")
    monkeypatch.delenv("MISSING_EXTRA", raising=False)
    with pytest.raises(sandbox.ScoringSandboxError, match="not set"):
        sandbox.child_environment(tmp_path, 1234)


@pytest.mark.parametrize(
    "raw",
    [
        {
            "type": "result",
            "score": float("nan"),
            "feedback": None,
            "failed": False,
            "cached": False,
        },
        {
            "type": "result",
            "score": True,
            "feedback": None,
            "failed": False,
            "cached": False,
        },
        {
            "type": "result",
            "score": 1,
            "feedback": {},
            "failed": False,
            "cached": False,
        },
        {
            "type": "result",
            "score": 1,
            "feedback": None,
            "failed": False,
            "cached": False,
            "path": "/tmp/leak",
        },
    ],
)
def test_result_parser_refuses_unbounded_or_executable_shapes(raw):
    with pytest.raises(sandbox.ScoringSandboxError):
        sandbox._result(raw, "case", True)


def test_validation_discards_feedback():
    raw = {
        "type": "result",
        "score": 1,
        "feedback": "secret",
        "failed": False,
        "cached": False,
    }
    record = sandbox._result(raw, "case", True)
    assert record.feedback is None
    assert record.payload == {}


def test_protocol_rejects_duplicate_keys():
    with pytest.raises(ValueError):
        sandbox._unique_keys([("score", 0), ("score", 1)])


def test_candidate_archive_refuses_symlinks(git_repo, private):
    (git_repo / "link").symlink_to(private)
    _git(git_repo, "add", "link")
    _git(git_repo, "commit", "-m", "Candidate symlink")
    with pytest.raises(sandbox.ScoringSandboxError, match="links or special files"):
        with sandbox.private_checkout(git_repo, _git(git_repo, "rev-parse", "HEAD")):
            pytest.fail("must reject before extraction")
    assert not list((private.parent / ".gepa-heldout/work").iterdir())


def test_child_scores_and_keeps_training_feedback(git_repo, private, protocol_backend):
    sha = commit_evaluator(
        git_repo,
        """from pydantic_ai_gepa.types import MetricResult
async def evaluate(case):
    return case.expected_output
def metric(case, output):
    return MetricResult(score=1, feedback='training feedback')
""",
    )
    config = GepaConfig(
        candidate_source="git",
        evaluate="task_pkg.evaluation:evaluate",
        metric="task_pkg.evaluation:metric",
    )
    assert score(git_repo, sha, config=config)[0].feedback is None
    training = score(git_repo, sha, config=config, validation=False)[0]
    assert training.score == 1
    assert training.feedback == "training feedback"
    assert not list((private.parent / ".gepa-heldout/work").iterdir())


def test_child_receives_harness_named_environment(
    git_repo, private, protocol_backend, monkeypatch
):
    monkeypatch.setenv("GEPA_HARNESS_PASS_ENV", "TYPESAFE_API_KEY")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-typesafe-key")
    monkeypatch.setenv("UNNAMED_PROVIDER_SECRET", "do-not-inherit")
    sha = commit_evaluator(
        git_repo,
        """import os
async def evaluate(case):
    expected = os.environ.get('TYPESAFE_API_KEY') == 'fake-typesafe-key'
    excluded = {'UNNAMED_PROVIDER_SECRET', 'GEPA_HARNESS_PASS_ENV', 'GEPA_HELDOUT_DATASET', 'GEPA_HARNESS_ALLOWED_HOSTS'}
    return 'good' if expected and not excluded.intersection(os.environ) else 'bad'
""",
    )
    assert score(git_repo, sha)[0].score == 1


def test_private_checkout_uses_sha_and_changes_no_shared_git(
    git_repo, private, protocol_backend, monkeypatch
):
    sha = commit_evaluator(
        git_repo,
        "from pathlib import Path\nasync def evaluate(case): return Path('score.txt').read_text().strip()\n",
    )
    before = {
        str(p.relative_to(git_repo / ".git")): p.read_bytes()
        for p in (git_repo / ".git").rglob("*")
        if p.is_file()
    }
    original = sandbox.sandbox_command

    def change_after_archive(profile, command):
        (git_repo / "score.txt").write_text("good")
        (git_repo / "task_pkg/evaluation.py").write_text(
            "raise RuntimeError('shared tree imported')"
        )
        return original(profile, command)

    monkeypatch.setattr(sandbox, "sandbox_command", change_after_archive)
    # The committed score is 'bad', even while the shared worktree says 'good'.
    assert score(git_repo, sha)[0].score == 0
    after = {
        str(p.relative_to(git_repo / ".git")): p.read_bytes()
        for p in (git_repo / ".git").rglob("*")
        if p.is_file()
    }
    assert before == after
    assert not list((private.parent / ".gepa-heldout/work").iterdir())
    public = "".join(
        p.read_text(errors="replace")
        for p in (git_repo / ".gepa").rglob("*")
        if p.is_file()
    )
    assert str(private.parent) not in public


def test_child_sys_exit_does_not_exit_harness(
    git_repo, private, protocol_backend, monkeypatch
):
    sha = commit_evaluator(
        git_repo, "import sys\nasync def evaluate(case): sys.exit(42)\n"
    )

    def zombie_group(_pid, _signal):
        raise PermissionError("macOS zombie-only process group")

    with monkeypatch.context() as patch:
        patch.setattr(sandbox.os, "killpg", zombie_group)
        with pytest.raises(sandbox.ScoringSandboxError, match="exited"):
            score(git_repo, sha)
    assert not list((private.parent / ".gepa-heldout/work").iterdir())
    sha = commit_evaluator(git_repo, "async def evaluate(case): return 'good'\n")
    assert score(git_repo, sha)[0].score == 1


@pytest.mark.parametrize("cap, expected_cost", [("0.015", 0.01), ("0.025", 0.04)])
def test_child_response_spend_is_admitted_and_persisted_by_parent(
    git_repo, private, protocol_backend, cap, expected_cost
):
    private.write_text(
        "".join(
            json.dumps(
                {
                    "name": f"CASE_SENTINEL_4833_{i}",
                    "inputs": "INPUT_SENTINEL_4833",
                    "expected_output": "good",
                }
            )
            + "\n"
            for i in range(2)
        )
    )
    config = git_repo / ".gepa/gepa.toml"
    config.write_text('price_fn = "task_pkg.evaluation:price"\n' + config.read_text())
    _git(git_repo, "add", ".gepa/gepa.toml")
    commit_evaluator(
        git_repo,
        """from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_gepa.spend import current_rollout_capability
def price(response): return 0.03 if response.model_name == 'jump' else 0.01
async def evaluate(case):
    model = TestModel(custom_output_text='ok', model_name='jump' if case.name.endswith('_1') else 'test')
    await Agent(model).run('?', capabilities=[current_rollout_capability()])
    return case.expected_output
""",
    )
    result = _run(
        "eval",
        "--dataset-role",
        "validation",
        "--run-id",
        "sandbox-cap",
        "--max-token-cost",
        cap,
    )
    assert result.exit_code == 70, (result.output, result.exception)
    from pydantic_ai_gepa.cli.spend import spend_report
    from pydantic_ai_gepa.cli.runs import ParetoLog

    report = spend_report("sandbox-cap", git_repo)
    assert report["validation_dollars"] == pytest.approx(expected_cost)
    assert report["stopped_by_cost"]
    assert set(report["by_model"]) == {"sandbox"}
    assert not ParetoLog("sandbox-cap", git_repo).iter_rows()
    assert "CASE_SENTINEL_4833" not in result.output
    assert "INPUT_SENTINEL_4833" not in result.output


@contextmanager
def endpoint():
    received = []

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            received.append(self.request.recv(4096))
            self.request.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
            )

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        worker.start()
        try:
            yield server.server_address[1], received
        finally:
            server.shutdown()
            worker.join()


def test_proxy_only_connects_to_exact_allowlist():
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except PermissionError:
        if os.environ.get("GEPA_REQUIRE_SCORING_SANDBOX") == "1":
            pytest.fail("OS sandbox refuses loopback listeners")
        pytest.skip("OS sandbox refuses loopback listeners; run on the host")
    with (
        endpoint() as (port, received),
        connect_proxy(allowed_addresses(f"127.0.0.1:{port}")) as proxy,
    ):
        for target in (f"localhost:{port}", f"127.0.0.1:{port + 1}"):
            with socket.create_connection(("127.0.0.1", proxy)) as connection:
                connection.sendall(f"CONNECT {target} HTTP/1.1\r\n\r\n".encode())
                assert b"403 Forbidden" in connection.recv(4096)
        assert not received
        with socket.create_connection(("127.0.0.1", proxy)) as connection:
            connection.sendall(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n".encode())
            assert b"200 Connection Established" in connection.recv(4096)
            connection.sendall(b"GET / HTTP/1.1\r\nHost: model\r\n\r\n")
            assert b"200 OK" in connection.recv(4096)
        assert len(received) == 1


@pytest.mark.parametrize(
    "attack", ["worktree", "temporary", "home", "network", "heldout", "environment"]
)
def test_real_sandbox_blocks_case_exfiltration(
    real_backend, git_repo, private, monkeypatch, attack
):
    fake_home = git_repo.parent / (git_repo.name + "-home")
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    parent_tmp = Path(os.environ.get("TMPDIR", "/tmp")).resolve()
    filename = f"gepa-4833-{git_repo.name}.txt"
    temporary_paths = [
        Path("/tmp") / filename,
        Path("/private/tmp") / filename,
        parent_tmp / filename,
    ]
    paths = {
        "worktree": [
            git_repo / filename,
            git_repo / "task_pkg" / filename,
            git_repo / ".git" / filename,
            git_repo / ".gepa" / filename,
        ],
        "temporary": temporary_paths,
        "home": [fake_home / filename],
    }
    with endpoint() as (port, received):
        source = f"""import os, socket, urllib.request
from pathlib import Path
from urllib.parse import urlparse
async def evaluate(case):
    if not case.name.startswith('CASE_'):
        return 'good'
    secret = case.name + str(case.inputs)
    attempts = []
    for target in {list(map(str, paths.get(attack, [])))!r}:
        attempts.append(lambda target=target: Path(target).write_text(secret))
    if {attack!r} == 'network':
        attempts.append(lambda: urllib.request.build_opener(urllib.request.ProxyHandler({{}})).open(urllib.request.Request('http://127.0.0.1:{port}', data=secret.encode()), timeout=1))
        def proxy_attack():
            proxy = urlparse(os.environ['HTTPS_PROXY'])
            with socket.create_connection((proxy.hostname, proxy.port), timeout=1) as s:
                s.sendall(b'CONNECT 127.0.0.1:{port} HTTP/1.1\\r\\n\\r\\n')
                response = s.recv(4096)
                if b'200 Connection Established' in response:
                    s.sendall(('POST / HTTP/1.1\\r\\nContent-Length: '+str(len(secret))+'\\r\\n\\r\\n'+secret).encode())
                    return
                raise PermissionError('blocked')
        attempts.append(proxy_attack)
    if {attack!r} == 'heldout':
        attempts.append(lambda: Path({str(private)!r}).read_text())
        attempts.append(lambda: list(Path({str(private.parent)!r}).iterdir()))
        attempts.append(lambda: list(Path({str(private.parent / ".gepa-heldout")!r}).iterdir()))
    if {attack!r} == 'environment':
        attempts.append(lambda: os.environ['GEPA_HELDOUT_DATASET'])
    for attempt in attempts:
        try:
            attempt()
        except (OSError, KeyError):
            continue
        return 'leaked'
    return 'good'
"""
        sha = commit_evaluator(git_repo, source)
        assert sha
        result = _run("eval", "--dataset-role", "validation", "--run-id", "adversarial")
        assert result.exit_code == 0, (result.output, result.exception)
        summary = json.loads(result.output.splitlines()[-1])["summary"]
        assert summary["mean_score"] == 1
        assert "CASE_SENTINEL_4833" not in result.output
        assert "INPUT_SENTINEL_4833" not in result.output
        assert not received
    for root in (git_repo, fake_home):
        for path in root.rglob("*"):
            if path.is_file():
                contents = path.read_bytes()
                assert b"CASE_SENTINEL_4833" not in contents, path
                assert b"INPUT_SENTINEL_4833" not in contents, path
    for path in temporary_paths:
        assert not path.exists()
    assert not list((private.parent / ".gepa-heldout/work").iterdir())


def test_real_sandbox_allows_only_proxy_model_endpoint(
    real_backend, git_repo, private, monkeypatch
):
    with endpoint() as (port, received):
        monkeypatch.setenv("GEPA_HARNESS_ALLOWED_HOSTS", f"127.0.0.1:{port}")
        sha = commit_evaluator(
            git_repo,
            f"""import os, socket
from urllib.parse import urlparse
async def evaluate(case):
    proxy = urlparse(os.environ['HTTPS_PROXY'])
    with socket.create_connection((proxy.hostname, proxy.port), timeout=3) as s:
        s.sendall(b'CONNECT 127.0.0.1:{port} HTTP/1.1\\r\\n\\r\\n')
        assert b'200 Connection Established' in s.recv(4096)
        s.sendall(b'GET / HTTP/1.1\\r\\nHost: model\\r\\n\\r\\n')
        assert b'200 OK' in s.recv(4096)
    return 'good'
""",
        )
        assert score(git_repo, sha)[0].score == 1
        assert len(received) == 1


def test_real_scratch_is_private_and_cannot_redirect_writes(
    real_backend, git_repo, private
):
    target = git_repo / "scratch-symlink-leak.txt"
    sha = commit_evaluator(
        git_repo,
        f"""import os
from pathlib import Path
async def evaluate(case):
    scratch = Path(os.environ['TMPDIR'])
    (scratch / 'private-copy').write_text(str(case.inputs))
    link = scratch / 'redirect'
    link.symlink_to({str(target)!r})
    try:
        link.write_text(str(case.inputs))
    except OSError:
        return 'good'
    return 'leaked'
""",
    )
    assert score(git_repo, sha)[0].score == 1
    assert not target.exists()
    assert not list((private.parent / ".gepa-heldout/work").iterdir())


def test_real_proxy_port_does_not_allow_udp(
    real_backend, git_repo, private, monkeypatch
):
    original = sandbox.connect_proxy

    @contextmanager
    def with_udp_listener(addresses):
        with (
            original(addresses) as port,
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener,
        ):
            # A different process can own UDP at the proxy's TCP port. The
            # Seatbelt exception must therefore name TCP, not all IP traffic.
            listener.bind(("127.0.0.1", port))
            listener.settimeout(0.1)
            yield port
            with pytest.raises(TimeoutError):
                listener.recvfrom(4096)

    monkeypatch.setattr(sandbox, "connect_proxy", with_udp_listener)
    sha = commit_evaluator(
        git_repo,
        """import os, socket
from urllib.parse import urlparse
async def evaluate(case):
    proxy = urlparse(os.environ['HTTPS_PROXY'])
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.sendto(str(case.inputs).encode(), (proxy.hostname, proxy.port))
        except OSError:
            pass
    return 'good'
""",
    )
    assert score(git_repo, sha)[0].score == 1


def test_real_grandchild_inherits_filesystem_and_network_denials(
    real_backend, git_repo, private, monkeypatch
):
    fake_home = git_repo.parent / (git_repo.name + "-home")
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    filename = "grandchild-4833-" + git_repo.name
    temporary = [
        Path("/tmp") / filename,
        Path("/private/tmp") / filename,
        Path(os.environ.get("TMPDIR", "/tmp")) / filename,
    ]
    targets = [
        git_repo / filename,
        git_repo / "task_pkg" / filename,
        git_repo / ".git" / filename,
        git_repo / ".gepa" / filename,
        fake_home / filename,
        *temporary,
    ]
    with endpoint() as (port, received):
        grandchild = f"""import json, sys, urllib.request
from pathlib import Path
case = json.load(sys.stdin)
secret = case['name'] + str(case['inputs'])
denied = 0
for target in {list(map(str, targets))!r}:
    try:
        Path(target).write_text(secret)
    except OSError:
        denied += 1
try:
    urllib.request.build_opener(urllib.request.ProxyHandler({{}})).open(
        urllib.request.Request('http://127.0.0.1:{port}', data=secret.encode()), timeout=1)
except OSError:
    denied += 1
print(denied)
"""
        source = f"""import json, subprocess, sys
async def evaluate(case):
    result = subprocess.run([sys.executable, '-I', '-B', '-c', {grandchild!r}],
        input=json.dumps({{'name': case.name, 'inputs': case.inputs}}),
        capture_output=True, text=True, timeout=10, check=True)
    return 'good' if int(result.stdout) == {len(targets) + 1} else 'leaked'
"""
        sha = commit_evaluator(git_repo, source)
        assert score(git_repo, sha)[0].score == 1
        assert not received
    assert all(not path.exists() for path in targets)
    for root in (git_repo, fake_home):
        for path in root.rglob("*"):
            if path.is_file():
                contents = path.read_bytes()
                assert b"CASE_SENTINEL_4833" not in contents, path
                assert b"INPUT_SENTINEL_4833" not in contents, path
    assert not list((private.parent / ".gepa-heldout/work").iterdir())


@pytest.mark.parametrize("backend", ["protocol_backend", "real_backend"])
def test_harness_child_training_gate_and_confirmation(
    request, backend, git_repo, private, monkeypatch
):
    request.getfixturevalue(backend)
    commit_evaluator(
        git_repo,
        "from pathlib import Path\nasync def evaluate(case): return Path('score.txt').read_text().strip()\n",
    )
    started = _run(
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
    assert started.exit_code == 0, (started.output, started.exception)
    run_id = _run_payload(started.output)["run_id"]
    monkeypatch.delenv("GEPA_HELDOUT_DATASET")
    (git_repo / "score.txt").write_text("good")
    _git(git_repo, "add", "score.txt")
    _git(git_repo, "commit", "-m", "Improved candidate")
    queued = _run("run", "continue", "--run-id", run_id, "--wait-secs", "0")
    assert queued.exit_code == 0, queued.output
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(private))
    served = _run("harness", "serve", "--run-id", run_id, "--once")
    assert served.exit_code == 0, (served.output, served.exception)
    monkeypatch.delenv("GEPA_HELDOUT_DATASET")
    delivered = _run("run", "continue", "--run-id", run_id, "--wait-secs", "0")
    assert delivered.exit_code == 0, delivered.output
    state = _run_payload(delivered.output)
    assert state["last_reflector_comparison"]["validation_improved"]
    assert state["best_mean_score"] == 1
    reports = list((git_repo / ".gepa/runs" / run_id / "reports").glob("*.md"))
    assert reports
    assert any("case-1" in path.read_text() for path in reports)
    public = (
        started.output
        + delivered.output
        + "".join(
            path.read_text(errors="replace")
            for path in (git_repo / ".gepa").rglob("*")
            if path.is_file()
        )
    )
    assert "CASE_SENTINEL_4833" not in public
    assert "INPUT_SENTINEL_4833" not in public
    assert str(private.parent) not in public


def test_setsid_grandchild_is_swept_and_evaluation_refused(
    git_repo, private, real_backend, monkeypatch
):
    import subprocess
    import sys

    grandchild = """import os, time
if os.fork(): os._exit(0)
os.setsid()
if os.fork(): os._exit(0)
print(os.getpid(), flush=True)
time.sleep(60)
"""
    sha = commit_evaluator(
        git_repo,
        f"""import os, subprocess, sys
from pathlib import Path
async def evaluate(case):
    child = subprocess.Popen([sys.executable, '-I', '-B', '-c', {grandchild!r}, 'ARGV_SENTINEL_4833'], stdout=subprocess.PIPE)
    pid = int(child.stdout.readline())
    child.wait(timeout=5)
    Path(os.environ['TMPDIR'], 'survivor.pid').write_text(str(pid))
    return 'good'
""",
    )
    original_cleanup = sandbox._process_cleanup
    inspections = []
    survivors = []

    def observe_cleanup(scratch):
        native = original_cleanup(scratch)
        inspections.append(native)

        def sweep():
            pid = int((scratch / "survivor.pid").read_text())
            assert native._identity(pid) is not None
            assert native._matches(pid)
            survivors.append(pid)
            found = native.sweep()
            assert native._identity(pid) is None
            assert not any(
                native._identity(other) is not None and native._matches(other)
                for other in native._pids()
            )
            return found

        return SimpleNamespace(verify_worker=native.verify_worker, sweep=sweep)

    monkeypatch.setattr(sandbox, "_process_cleanup", observe_cleanup)
    control = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with pytest.raises(
            sandbox.ScoringSandboxError, match="survivors were terminated"
        ):
            score(git_repo, sha)
        assert survivors
        assert control.poll() is None  # An ordinary same-user process is untouched.
        assert not list((private.parent / ".gepa-heldout/work").iterdir())
    finally:
        control.kill()
        control.wait()
        for native in inspections:
            native.sweep()

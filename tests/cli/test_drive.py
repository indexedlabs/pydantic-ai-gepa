"""Managed loops use fake reflectors and the existing fake evaluator fixtures."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from pydantic_ai_gepa.cli import drive, events, layout
from pydantic_ai_gepa.cli.process_guard import Process
from tests.cli import test_harness_scoring, test_lanes_cli, test_process_guard
from tests.cli.test_git_candidate_cli import _run, _run_payload

git_repo = test_lanes_cli.git_repo
heldout = test_harness_scoring.heldout
host_processes = test_process_guard.host_processes


class OwnProcesses:
    """Portable controller-test seam; host libproc is tested separately."""

    def __init__(self):
        self.pids = set()

    def identity(self, pid, *, include_zombie=False):
        self.pids.add(pid)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        return Process(pid, pid + 1000000, 1, (1, 0), "test reflector")

    def snapshot(self):
        return [Process(os.getpid(), 1, 0, (1, 0), "test driver")] + [
            item for pid in list(self.pids) if (item := self.identity(pid)) is not None
        ]

    def kill(self, process):
        assert process.pid in self.pids
        os.kill(process.pid, signal.SIGKILL)


@pytest.fixture(autouse=True)
def driver_environment(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("driver-private")))
    monkeypatch.setattr(drive, "DarwinProcesses", OwnProcesses)
    monkeypatch.setattr(layout, "_explicit_gepa_dirname", None)
    monkeypatch.delenv("GEPA_DIR", raising=False)
    monkeypatch.delenv("GEPA_HELDOUT_DATASET", raising=False)


def config(repo: Path, source: str | None = None, entries=None) -> Path:
    private = repo / ".gepa/runs/fake-reflector"
    private.mkdir(parents=True, exist_ok=True)
    script = private / "reflect.py"
    script.write_text(
        source
        or """import json, os, subprocess, sys
from pathlib import Path
packet = json.loads(Path(sys.argv[1]).read_text())
assert "GEPA_HELDOUT_DATASET" not in os.environ
assert not any(k.startswith("GEPA_HARNESS_") for k in os.environ)
Path("score.txt").write_text("good\\n")
subprocess.run(["git", "add", "score.txt"], check=True)
subprocess.run(["git", "commit", "--allow-empty", "-m", "Fake reflection"], check=True)
if "continue_argv" in packet:
    argv = packet["continue_argv"] + ["--foreground"]
else:
    argv = [sys.executable, "-I", "-c", "from pydantic_ai_gepa.cli import app; app()"] + packet["next_command"]["argv"][1:] + ["--wait-secs", "0"]
subprocess.run(argv, check=True)
"""
    )
    path = private / "reflectors.json"
    path.write_text(
        json.dumps(
            entries
            or [{"label": "fake", "argv": [sys.executable, str(script), "{packet}"]}]
        )
    )
    return path


def start_lanes():
    started = _run(
        "run",
        "start",
        "--lanes",
        "1",
        "--size",
        "1",
        "--max-iterations",
        "8",
        "--acceptance-repetitions",
        "1",
        "--acceptance-max-repetitions",
        "1",
    )
    assert started.exit_code == 0, started.output
    return str(_run_payload(started.output)["run_id"])


def invoke(run_id, path, *extra):
    return _run(
        "run",
        "drive",
        "--run-id",
        run_id,
        "--reflectors",
        str(path),
        "--step-timeout",
        "20",
        "--max-steps",
        "4",
        *extra,
    )


def driver_state(repo, run_id):
    return json.loads(drive.private_state_path(run_id).read_text())


def test_lane_run_reaches_done_and_records_before_ack(git_repo, monkeypatch):
    run_id = start_lanes()
    path = config(git_repo)
    original = events.ack
    acknowledgements = []

    def checked_ack(run_id, event_id, root=None):
        saved = driver_state(git_repo, run_id)
        assert saved["phase"] == "recorded"
        assert saved["event"]["id"] == event_id
        acknowledgements.append(event_id)
        return original(run_id, event_id, root)

    monkeypatch.setattr(events, "ack", checked_ack)
    result = invoke(run_id, path)
    assert result.exit_code == 0, (result.output, result.exception)
    run = json.loads((git_repo / ".gepa/runs" / run_id / "state.json").read_text())
    assert run["status"] == "done"
    assert "final_report.md" in result.output
    assert acknowledgements
    assert driver_state(git_repo, run_id)["sequence"] >= 1
    report = git_repo / ".gepa/runs" / run_id / "final_report.md"
    report.unlink()  # a kill can precede the original controller's report write
    restarted = invoke(run_id, path)
    assert restarted.exit_code == 0, restarted.output
    assert report.exists()


def test_single_heldout_run_scores_after_reflector_exit(git_repo, heldout, monkeypatch):
    dataset, run_id, _ = heldout
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(dataset))
    monkeypatch.setenv("GEPA_HARNESS_PASS_ENV", "SECRET")
    monkeypatch.setenv("SECRET", "private")
    path = config(git_repo)
    original = drive.harness.serve
    calls = []

    def checked_serve(**kwargs):
        pin = drive._pin_path(str(dataset), git_repo, run_id)
        saved = json.loads(pin.with_name(pin.stem + ".drive.json").read_text())
        pid = saved["step"]["pid"]
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        calls.append(pid)
        return original(**kwargs)

    monkeypatch.setattr(drive.harness, "serve", checked_serve)
    result = invoke(run_id, path)
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls
    assert len(calls) == len(set(calls))
    pin = drive._pin_path(str(dataset), git_repo, run_id)
    state_path = pin.with_name(pin.stem + ".drive.json")
    assert state_path.stat().st_mode & 0o777 == 0o600
    assert state_path.parent.stat().st_mode & 0o777 == 0o700
    assert not (git_repo / ".gepa/runs" / run_id / "drive.json").exists()
    assert "final_report.md" in result.output


def test_timeout_retries_then_pauses(git_repo):
    run_id = start_lanes()
    path = config(git_repo, "import time\ntime.sleep(30)\n")
    result = invoke(run_id, path, "--step-timeout", "0.1")
    assert result.exit_code == drive.EXIT_PAUSED, (result.output, result.exception)
    saved = driver_state(git_repo, run_id)
    assert saved["pause_reason"] == "attempt_limit"
    assert saved["attempts"]["lane-1"] == 2
    assert saved["sequence"] == 2
    with pytest.raises(ProcessLookupError):
        os.kill(saved["step"]["pid"], 0)


def test_usage_limit_fallback_and_exhaustion_restart(git_repo):
    run_id = start_lanes()
    path = config(git_repo)
    good = json.loads(path.read_text())[0]
    limit_script = path.parent / "limited.py"
    limit_script.write_text('print("usage limit reached")\nraise SystemExit(9)\n')
    limited = {
        "label": "limited",
        "argv": [sys.executable, str(limit_script)],
        "usage_limit": {"exit_codes": [9]},
    }
    path.write_text(json.dumps([limited]))
    first = invoke(run_id, path)
    assert first.exit_code == drive.EXIT_USAGE_LIMIT, (first.output, first.exception)
    assert driver_state(git_repo, run_id)["reflector_index"] == 1
    second = invoke(run_id, path)
    assert second.exit_code == drive.EXIT_USAGE_LIMIT
    assert driver_state(git_repo, run_id)["sequence"] == 2
    assert [
        entry["label"]
        for entry in driver_state(git_repo, run_id)["usage_limits"].values()
    ] == ["limited", "limited"]
    path.write_text(json.dumps([limited, good]))
    refused = invoke(run_id, path)
    assert refused.exit_code == 2
    assert "Reflector list changed" in refused.output
    accepted = invoke(run_id, path, "--accept-reflector-change")
    assert accepted.exit_code == 0, (accepted.output, accepted.exception)
    assert driver_state(git_repo, run_id)["changes"]


def test_crash_between_record_and_ack_does_not_repeat_step(git_repo, monkeypatch):
    run_id = start_lanes()
    path = config(git_repo)
    original = events.ack
    monkeypatch.setattr(
        events,
        "ack",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("crash before ack")),
    )
    first = invoke(run_id, path)
    assert first.exit_code != 0
    saved = driver_state(git_repo, run_id)
    assert saved["phase"] == "recorded"
    assert saved["sequence"] == 1
    monkeypatch.setattr(events, "ack", original)
    second = invoke(run_id, path)
    assert second.exit_code == 0, (second.output, second.exception)
    assert driver_state(git_repo, run_id)["sequence"] == 1


def test_scrub_harness_environment(monkeypatch):
    for key in (
        "GEPA_HELDOUT_DATASET",
        "GEPA_HARNESS_EXTRA",
        "OPENAI_API_KEY",
        "SCORER_TOKEN",
    ):
        monkeypatch.setenv(key, "secret")
    monkeypatch.setenv("GEPA_HARNESS_PASS_ENV", "SCORER_TOKEN")
    monkeypatch.setenv("CODEX_HOME", "reflector-auth")
    environment = drive.reflector_environment()
    assert environment["CODEX_HOME"] == "reflector-auth"
    assert (
        not {
            "GEPA_HELDOUT_DATASET",
            "GEPA_HARNESS_EXTRA",
            "OPENAI_API_KEY",
            "SCORER_TOKEN",
        }
        & environment.keys()
    )


def test_unattributable_survivor_pauses_without_scoring_or_killing(
    git_repo, monkeypatch
):
    class UnknownSurvivor(OwnProcesses):
        def snapshot(self):
            snapshot = super().snapshot()
            if self.pids:
                snapshot.append(
                    Process(99999999, 99999999, 99999998, (9, 0), "unknown")
                )
            return snapshot

        def kill(self, process):
            pytest.fail("An unattributable process must never be signaled")

    run_id = start_lanes()
    path = config(git_repo)
    monkeypatch.setattr(drive, "DarwinProcesses", UnknownSurvivor)
    from pydantic_ai_gepa.cli import select

    monkeypatch.setattr(
        select, "run_select", lambda *args: pytest.fail("scoring started")
    )
    result = invoke(run_id, path, "--survivor-grace", "0.01")
    assert result.exit_code == drive.EXIT_SURVIVORS, (result.output, result.exception)
    assert "99999999" in result.output
    assert driver_state(git_repo, run_id)["pause_reason"] == "process_survivors"


def test_usage_limit_single_reissues_epoch(git_repo, heldout, monkeypatch):
    dataset, run_id, _ = heldout
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", str(dataset))
    path = config(git_repo)
    good = json.loads(path.read_text())[0]
    script = path.parent / "limit.py"
    script.write_text('print("subscription exhausted")\nraise SystemExit(8)\n')
    path.write_text(
        json.dumps(
            [
                {
                    "label": "limited",
                    "argv": [sys.executable, str(script)],
                    "usage_limit": {"output_regexes": ["subscription exhausted"]},
                },
                good,
            ]
        )
    )
    result = invoke(run_id, path)
    assert result.exit_code == 0, (result.output, result.exception)
    run = json.loads((git_repo / ".gepa/runs" / run_id / "state.json").read_text())
    assert run["reflector"]["epoch"] == 2
    assert run["reflector"]["label"] == "fake"
    assert run["reflector"]["history"][-1]["lost_reason"] == "usage_limit"


def test_provider_stop_is_not_retried_on_restart(git_repo, monkeypatch):
    from pydantic_ai_gepa.cli import select
    from pydantic_ai_gepa.provider_errors import ProviderStopError

    run_id = start_lanes()
    path = config(git_repo)
    calls = []

    def unavailable(*args):
        calls.append(args)
        raise ProviderStopError("provider unavailable")

    monkeypatch.setattr(select, "run_select", unavailable)
    for _ in range(2):
        result = invoke(run_id, path)
        assert result.exit_code == drive.EXIT_PAUSED, (result.output, result.exception)
        assert driver_state(git_repo, run_id)["pause_reason"] == "provider_stop"
    assert len(calls) == 1


def wait_for(path, process):
    deadline = time.monotonic() + 20
    while not path.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert path.exists(), f"Driver exited {process.poll()} before {path.name}"


def driver_subprocess(repo, run_id, config_path, extra_source="", env=None):
    launcher = config_path.parent / "launch-driver.py"
    if env and env.get("GEPA_HELDOUT_DATASET"):
        pin = drive._pin_path(env["GEPA_HELDOUT_DATASET"], repo, run_id)
        state_path = pin.with_name(pin.stem + ".drive.json")
    else:
        state_path = drive.private_state_path(run_id)
    # Preserve real kernel identities and signaling, but provide only the test's
    # process tree to the guard. Uncontrolled launchd/XPC activity correctly
    # pauses production; it must not make this replay test nondeterministic.
    controlled_processes = f"""import json, os
from pathlib import Path
from pydantic_ai_gepa.cli import drive
from pydantic_ai_gepa.cli.process_guard import DarwinProcesses
class ControlledProcesses(DarwinProcesses):
    def snapshot(self):
        processes = super().snapshot()
        state_path = Path({str(state_path)!r})
        state = json.loads(state_path.read_text()) if state_path.exists() else {{}}
        guard = (state.get("step") or {{}}).get("guard", {{}})
        owner = super().identity(os.getpid()).unique_id
        roots = {{owner, guard.get("root"), guard.get("owner")}} - {{None}}
        parents = {{int(k): v["parent_id"] for k, v in guard.get("seen", {{}}).items()}}
        parents.update({{p.unique_id: p.parent_id for p in processes}})
        def controlled(identity):
            seen = set()
            while identity not in seen:
                if identity in roots: return True
                seen.add(identity)
                if identity not in parents: return False
                identity = parents[identity]
            return False
        return [p for p in processes if controlled(p.unique_id)]
drive.DarwinProcesses = ControlledProcesses
"""
    launcher.write_text(
        "from pydantic_ai_gepa.cli import app, scoring_sandbox, safe_git\n"
        "scoring_sandbox.required = lambda: False\n"
        "safe_git.refuse_heldout_git_mutations = lambda: None\n"
        + controlled_processes
        + extra_source
        + "\napp()\n"
    )
    log = (config_path.parent / "driver.log").open("ab")
    try:
        return subprocess.Popen(
            [
                sys.executable,
                "-I",
                str(launcher),
                "--gepa-dir",
                str(repo / ".gepa"),
                "run",
                "drive",
                "--run-id",
                run_id,
                "--reflectors",
                str(config_path),
                "--step-timeout",
                "30",
                "--max-steps",
                "3",
            ],
            cwd=repo,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )
    finally:
        log.close()


def test_host_restart_killed_driver_guards_orphan_before_next_step(
    git_repo, host_processes
):
    run_id = start_lanes()
    path = config(git_repo)
    script = path.parent / "reflect.py"
    marker = path.parent / "reflector-entered"
    original = script.read_text()
    script.write_text(
        "import os, time\nfrom pathlib import Path\n"
        f"marker = Path({str(marker)!r})\n"
        "if not marker.exists():\n"
        "    marker.write_text(str(os.getpid()))\n"
        "    time.sleep(30)\n" + original
    )
    first = driver_subprocess(git_repo, run_id, path)
    second = None
    orphan = None
    try:
        wait_for(marker, first)
        orphan = host_processes.identity(int(marker.read_text()))
        assert orphan is not None
        first.kill()  # only the driver; deliberately leave its reflector
        first.wait(timeout=5)
        assert host_processes.identity(orphan.pid) is not None
        second = driver_subprocess(git_repo, run_id, path)
        assert second.wait(timeout=30) == 0, (path.parent / "driver.log").read_text()
        assert host_processes.identity(orphan.pid) is None
        assert driver_state(git_repo, run_id)["sequence"] == 2
        assert (git_repo / ".gepa/runs" / run_id / "final_report.md").exists()
    finally:
        for process in (first, second):
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
        if orphan is not None:
            host_processes.kill(orphan)


def test_host_restart_during_scoring_replays_paid_nomination(
    git_repo, heldout, host_processes
):
    dataset, run_id, _ = heldout
    path = config(git_repo)
    marker = path.parent / "scoring-entered"
    calls = path.parent / "validation-calls"
    intercept = f"""from pathlib import Path
import time
from pydantic_ai_gepa.cli import run
original = run.run_eval_once
def interrupted(**kwargs):
    result = original(**kwargs)
    if kwargs.get("dataset_role") == "validation":
        with Path({str(calls)!r}).open("a") as output:
            output.write("validation\\n")
        marker = Path({str(marker)!r})
        if not marker.exists():
            marker.touch()
            time.sleep(30)
    return result
run.run_eval_once = interrupted
"""
    environment = dict(os.environ, GEPA_HELDOUT_DATASET=str(dataset))
    first = driver_subprocess(git_repo, run_id, path, intercept, environment)
    second = None
    try:
        wait_for(marker, first)
        first.kill()
        first.wait(timeout=5)
        second = driver_subprocess(git_repo, run_id, path, intercept, environment)
        assert second.wait(timeout=30) == 0, (path.parent / "driver.log").read_text()
        assert calls.read_text().splitlines() == ["validation"]
        root = git_repo / ".gepa/runs" / run_id
        assert len(list((root / "nominations").glob("*.json"))) == 1
        assert len(list((root / "results").glob("*.json"))) == 1
        assert (root / "final_report.md").exists()
    finally:
        for process in (first, second):
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)


def test_nomination_takes_priority_over_usage_limit_output(git_repo):
    run_id = start_lanes()
    path = config(git_repo)
    script = path.parent / "reflect.py"
    script.write_text(script.read_text() + '\nprint("usage limit")\n')
    entries = json.loads(path.read_text())
    entries[0]["usage_limit"] = {"output_regexes": ["usage limit"]}
    path.write_text(json.dumps(entries))
    result = invoke(run_id, path)
    assert result.exit_code == 0, (result.output, result.exception)
    assert driver_state(git_repo, run_id)["usage_limits"] == {}


def test_timeout_without_published_root_is_bounded(git_repo, monkeypatch):
    class Unpublished(OwnProcesses):
        def identity(self, pid, *, include_zombie=False):
            if include_zombie:
                self.pids.add(pid)
                return None
            return super().identity(pid)

    monkeypatch.setattr(drive, "DarwinProcesses", Unpublished)
    run_id = start_lanes()
    path = config(git_repo, "import time\ntime.sleep(60)\n")
    started = time.monotonic()
    result = invoke(run_id, path, "--step-timeout", "0.1", "--max-attempts", "1")
    assert result.exit_code == drive.EXIT_PAUSED, (result.output, result.exception)
    assert time.monotonic() - started < 10
    state = driver_state(git_repo, run_id)
    assert state["step"]["guard"]["root"] is None
    with pytest.raises(ProcessLookupError):
        os.kill(state["step"]["pid"], 0)


def test_training_credentials_explicit_passthrough(git_repo, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "training-only")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.name")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "Test")
    path = config(git_repo)
    path.write_text(
        json.dumps(
            {"reflectors": json.loads(path.read_text()), "pass_env": ["OPENAI_API_KEY"]}
        )
    )
    _, _, names = drive.load_reflectors(path)
    env = drive.reflector_environment(names)
    assert env["OPENAI_API_KEY"] == "training-only"
    assert env["GIT_CONFIG_COUNT"] == "3"
    assert env["GIT_CONFIG_KEY_0"] == "user.name"
    assert env["GIT_CONFIG_KEY_1"] == "gc.autoDetach"
    assert env["GIT_CONFIG_KEY_2"] == "maintenance.autoDetach"
    assert env["GIT_CONFIG_VALUE_1"] == env["GIT_CONFIG_VALUE_2"] == "false"
    assert "OPENAI_API_KEY" not in drive.reflector_environment()
    script = path.parent / "reflect.py"
    script.write_text(
        'import os\nassert os.environ["OPENAI_API_KEY"] == "training-only"\n'
        + script.read_text()
    )
    run_id = start_lanes()
    result = invoke(run_id, path)
    assert result.exit_code == 0, (result.output, result.exception)


@pytest.mark.parametrize(
    "name",
    [
        "SCORER_TOKEN",
        "OPENAI_API_KEY",
        "GEPA_HELDOUT_OTHER",
        "GEPA_HARNESS_EXTRA",
        "GEPA_CANDIDATE_COMPONENTS_JSON",
        "GEPA_TRACE_FILE",
        "*_KEY",
    ],
)
def test_forbidden_credentials_cannot_be_allowlisted(git_repo, monkeypatch, name):
    monkeypatch.setenv("GEPA_HARNESS_PASS_ENV", "SCORER_TOKEN,OPENAI_API_KEY")
    monkeypatch.setenv(name, "harness-only")
    path = config(git_repo)
    path.write_text(
        json.dumps({"reflectors": json.loads(path.read_text()), "pass_env": [name]})
    )
    with pytest.raises(drive.typer.BadParameter, match="explicit training-only"):
        drive.load_reflectors(path)


def test_public_forged_guard_is_never_loaded(git_repo):
    run_id = start_lanes()
    path = config(git_repo)
    public = git_repo / ".gepa/runs" / run_id / "drive.json"
    public.write_text('{"step":{"guard":{"root":1,"boot_id":"forged"}}}')
    result = invoke(run_id, path)
    assert result.exit_code == 2, result.output
    assert "Public drive.json is untrusted" in result.output
    assert json.loads(public.read_text())["step"]["guard"]["boot_id"] == "forged"


@pytest.mark.parametrize("location", [".gepa/private", "project-private"])
def test_private_state_inside_reflector_roots_refused(git_repo, monkeypatch, location):
    run_id = start_lanes()
    path = config(git_repo)
    monkeypatch.setenv("XDG_STATE_HOME", str(git_repo / location))
    result = invoke(run_id, path)
    assert result.exit_code == 2, result.output
    assert "outside GEPA_DIR" in result.output


def test_private_state_permissions(git_repo):
    run_id = start_lanes()
    path = drive.private_state_path(run_id)
    assert not path.is_relative_to(git_repo)
    assert path.parent.stat().st_mode & 0o777 == 0o700
    config_path = config(git_repo)
    assert invoke(run_id, config_path).exit_code == 0
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.with_suffix(".lock").stat().st_mode & 0o777 == 0o600
    path.parent.chmod(0o755)
    result = invoke(run_id, config_path)
    assert result.exit_code == 2
    assert "0700" in result.output


def test_guard_failure_saves_memory_without_reloading_disk(git_repo, monkeypatch):
    run_id = start_lanes()
    path = config(git_repo)
    saved = {}

    def failing_run(self):
        saved.update(self.state)
        self.path.write_text('{"step":{"guard":{"boot_id":"forged"}}}')
        raise drive.GuardError("inspection failed")

    monkeypatch.setattr(drive.Driver, "run", failing_run)
    result = invoke(run_id, path)
    assert result.exit_code == drive.EXIT_SURVIVORS
    expected = dict(saved, pause_reason="process_inspection_failed")
    assert driver_state(git_repo, run_id) == expected


def test_host_unfiltered_minute_step_kills_readers_before_scoring(
    git_repo, host_processes, monkeypatch
):
    from pydantic_ai_gepa.cli import select

    run_id = start_lanes()
    path = config(git_repo)
    script = path.parent / "reflect.py"
    original = script.read_text()
    markers = path.parent
    script.write_text(
        f"""import os, time
from pathlib import Path
base = Path({str(markers)!r})
started = time.monotonic()
for kind in ("double", "session"):
    if os.fork() == 0:
        os.setsid()
        if kind == "double" and os.fork() != 0:
            (base / "intermediate.pid").write_text(str(os.getpid()))
            while not (base / "release").exists(): time.sleep(0.01)
            os._exit(0)
        (base / (kind + ".pid")).write_text(str(os.getpid()))
        time.sleep(180)
        os._exit(0)
while not (base / "release").exists(): time.sleep(0.01)
while time.monotonic() - started < 60: time.sleep(0.05)
"""
        + original
    )
    owned = {}
    ordering = []
    original_observe = drive.ProcessGuard.observe
    original_phase = drive.Driver.phase
    original_kill, original_killpg = os.kill, os.killpg
    original_select = select.run_select
    reader_names = ("double", "session", "intermediate")

    def phase(self, name, **kwargs):
        original_phase(self, name, **kwargs)
        if name == "reflector_started":
            process = host_processes.identity(
                self.state["step"]["pid"], include_zombie=True
            )
            assert process is not None
            owned[process.pid] = process
            ordering.append(("started", time.monotonic()))

    def observe(self):
        snapshot = original_observe(self)  # Full same-UID snapshot, no filtering.
        for name in reader_names:
            marker = markers / (name + ".pid")
            if marker.exists() and marker.read_text():
                pid = int(marker.read_text())
                process = next((p for p in snapshot if p.pid == pid), None)
                if process is not None:
                    owned[pid] = process
        if all((markers / (name + ".pid")).exists() for name in reader_names):
            ids = [
                int((markers / (name + ".pid")).read_text()) for name in reader_names
            ]
            if all(
                pid in owned and str(owned[pid].unique_id) in self.record["seen"]
                for pid in ids
            ):
                (markers / "release").touch()
        return snapshot

    def check_target(pid):
        current = host_processes.identity(pid, include_zombie=True)
        assert current is not None and pid in owned, f"Unowned signal target {pid}"
        assert current.unique_id == owned[pid].unique_id
        return current.unique_id

    def kill(pid, sig):
        if sig:
            identity = check_target(pid)
            ordering.append(("kill", pid, identity, time.monotonic()))
        return original_kill(pid, sig)

    def killpg(pgid, sig):
        # The leader can be reaped; check every live member before the group call.
        members = []
        for process in host_processes.snapshot():
            try:
                if os.getpgid(process.pid) == pgid:
                    members.append(process)
            except ProcessLookupError:
                continue
        assert pgid in owned
        for member in members:
            check_target(member.pid)
        ordering.append(
            ("killpg", pgid, [p.unique_id for p in members], time.monotonic())
        )
        return original_killpg(pgid, sig)

    def score(run_id):
        assert time.monotonic() - ordering[0][1] >= 60
        for name in ("double", "session"):
            pid = int((markers / (name + ".pid")).read_text())
            assert host_processes.identity(pid) is None
            assert any(row[0] == "kill" and row[1] == pid for row in ordering)
        ordering.append(("scoring", time.monotonic()))
        return original_select(run_id)

    monkeypatch.setattr(drive, "DarwinProcesses", lambda: host_processes)
    monkeypatch.setattr(drive.Driver, "phase", phase)
    monkeypatch.setattr(drive.ProcessGuard, "observe", observe)
    monkeypatch.setattr(os, "kill", kill)
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(select, "run_select", score)
    try:
        result = invoke(run_id, path, "--step-timeout", "90", "--survivor-grace", "5")
        assert result.exit_code == 0, (result.output, result.exception)
        assert any(row[0] == "scoring" for row in ordering)
        assert drive._load_state(run_id).status == "done"
    finally:
        # Persist timing and signal targets for review, even on assertion failure.
        (markers / "signal-order.json").write_text(json.dumps(ordering))
        for process in owned.values():
            host_processes.kill(process)

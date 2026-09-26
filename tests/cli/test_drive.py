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

    def identity(self, pid):
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
def driver_environment(monkeypatch):
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
    return json.loads((repo / ".gepa/runs" / run_id / "drive.json").read_text())


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
    state_path = repo / ".gepa/runs" / run_id / "drive.json"
    if env and env.get("GEPA_HELDOUT_DATASET"):
        pin = drive._pin_path(env["GEPA_HELDOUT_DATASET"], repo, run_id)
        state_path = pin.with_name(pin.stem + ".drive.json")
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

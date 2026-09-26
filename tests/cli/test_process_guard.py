"""Guard classification and host-only detached descendant cleanup."""

from dataclasses import asdict
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from pydantic_ai_gepa.cli.process_guard import (
    DarwinProcesses,
    GuardError,
    Process,
    ProcessGuard,
)


class SnapshotProcesses:
    def __init__(self):
        self.processes = [Process(os.getpid(), 1, 0, (1, 0), "driver")]
        self.killed = []

    def snapshot(self):
        return list(self.processes)

    def kill(self, process):
        self.killed.append(process.pid)
        self.processes.remove(process)


def test_creation_chain_survives_parent_death_and_pid_reuse():
    backend = SnapshotProcesses()
    shell = Process(110, 2, 0, (1, 0), "unrelated shell")
    backend.processes.append(shell)
    record = {}
    guard = ProcessGuard(backend, record, lambda: None)
    guard.begin()
    root = Process(111, 3, 1, (2, 0), "reflector")
    middle = Process(112, 4, 3, (3, 0), "fork")
    backend.processes += [root, middle]
    record["root"] = root.unique_id
    guard.observe()
    # Intermediate parent is gone; its immutable identity remains in snapshots.
    reader = Process(113, 5, 4, (4, 0), "reader")
    unrelated = Process(112, 6, 2, (5, 0), "reused pid, unrelated")
    unknown = Process(114, 8, 7, (6, 0), "unseen parent")
    backend.processes = [backend.processes[0], shell, reader, unrelated, unknown]
    assert guard.finish(0.06) == [unknown]
    assert backend.killed == [reader.pid]
    assert guard.classify(unrelated) == "unrelated"
    assert guard.classify(unknown) == "unknown"
    assert unknown in backend.processes
    # Restart reads exactly the serialized ancestry, not current parent PIDs.
    resumed = ProcessGuard(backend, record, lambda: None)
    backend.processes.remove(unknown)
    assert resumed.finish(0.06) == []


def test_spawn_before_root_publication_is_unknown_not_unrelated():
    backend = SnapshotProcesses()
    guard = ProcessGuard(backend, {}, lambda: None)
    guard.begin()
    child = Process(123, 2, 1, (2, 0), "unrecorded reflector")
    backend.processes.append(child)
    assert guard.finish(0.01) == [child]
    assert backend.killed == []


def test_guard_persists_edges_before_signaling():
    backend = SnapshotProcesses()
    record = {}
    saved = []
    guard = ProcessGuard(
        backend, record, lambda: saved.append(dict(record.get("seen", {})))
    )
    guard.begin()
    root = Process(222, 2, 1, (2, 0), "reflector")
    record["root"] = 2
    backend.processes.append(root)
    assert guard.finish(0.1) == []
    assert saved[-1]["2"] == asdict(root)
    assert backend.killed == [222]


def test_reboot_does_not_signal_reused_kernel_identity():
    backend = SnapshotProcesses()
    backend.boot_id = "first-boot"
    guard = ProcessGuard(backend, {}, lambda: None)
    guard.begin()
    guard.record["root"] = 2
    backend.boot_id = "second-boot"
    backend.processes.append(Process(222, 2, 1, (2, 0), "different process"))
    assert guard.finish(0.01) == []
    assert backend.killed == []


@pytest.fixture
def host_processes():
    try:
        backend = DarwinProcesses()
        backend.snapshot()
    except (GuardError, OSError) as exc:
        pytest.skip(f"Host libproc process access required: {exc}")
    return backend


@pytest.mark.parametrize("detachment", ["double_fork", "setsid"])
def test_host_detached_reader_cannot_survive_guard(
    tmp_path: Path, host_processes, detachment
):
    script = tmp_path / "reader.py"
    script.write_text("""import os, sys, time
from pathlib import Path
base = Path(sys.argv[1])
if os.fork() == 0:
    os.setsid()
    if sys.argv[2] == "double_fork":
        if os.fork() != 0:
            (base / "intermediate").write_text(str(os.getpid()))
            while not (base / "release").exists(): time.sleep(0.01)
            os._exit(0)
    (base / "reader").write_text(str(os.getpid()))
    time.sleep(30)
    os._exit(0)
while not (base / "release").exists(): time.sleep(0.01)
""")
    record = {}
    guard = ProcessGuard(host_processes, record, lambda: None)
    guard.begin()
    process = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path), detachment], start_new_session=True
    )
    reader_pid = None
    try:
        root = host_processes.identity(process.pid)
        assert root is not None
        record["root"] = root.unique_id
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            guard.observe()
            if (tmp_path / "reader").exists() and (
                detachment != "double_fork" or (tmp_path / "intermediate").exists()
            ):
                reader_pid = int((tmp_path / "reader").read_text())
                guard.observe()
                break
            time.sleep(0.01)
        assert reader_pid is not None
        (tmp_path / "release").touch()
        process.wait(timeout=5)
        assert guard.finish(3) == []
        # This assertion is the scoring boundary: no reader can execute here.
        assert host_processes.identity(reader_pid) is None
    finally:
        (tmp_path / "release").touch()
        # Only PIDs from this test's child-written files are eligible for cleanup.
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for name in ("reader", "intermediate"):
            path = tmp_path / name
            if path.exists():
                pid = int(path.read_text())
                known = next(
                    (p for p in record.get("seen", {}).values() if p["pid"] == pid),
                    None,
                )
                current = host_processes.identity(pid)
                if current and known and current.unique_id == known["unique_id"]:
                    os.kill(pid, signal.SIGKILL)


def test_host_terminate_whole_reflector_group(tmp_path: Path, host_processes):
    from pydantic_ai_gepa.cli.drive import Driver

    script = tmp_path / "group.py"
    script.write_text("""import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
if os.fork() == 0:
    Path(sys.argv[1]).write_text(str(os.getpid()))
    time.sleep(30)
    os._exit(0)
time.sleep(30)
""")
    guard = ProcessGuard(host_processes, {}, lambda: None)
    guard.begin()
    path = tmp_path / "child"
    process = subprocess.Popen(
        [sys.executable, str(script), str(path)], start_new_session=True
    )
    child = None
    root = None
    try:
        root = host_processes.identity(process.pid)
        assert root is not None
        guard.record["root"] = root.unique_id
        deadline = time.monotonic() + 5
        while not path.exists() and time.monotonic() < deadline:
            guard.observe()
            time.sleep(0.01)
        assert path.exists()
        child = host_processes.identity(int(path.read_text()))
        assert child is not None
        owner = SimpleNamespace(backend=host_processes)
        try:
            Driver.signal_group(owner, process.pid, guard, signal.SIGTERM)
            time.sleep(0.05)
            Driver.signal_group(owner, process.pid, guard, signal.SIGKILL)
        except PermissionError:
            pytest.skip(
                "Host process-group signaling required; Codex sandbox denies killpg"
            )
        process.wait(timeout=5)
        assert host_processes.identity(root.pid) is None
        deadline = time.monotonic() + 3
        while host_processes.identity(child.pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert host_processes.identity(child.pid) is None
    finally:
        # This test owns both captured creation identities; never signal other PIDs.
        for created in (root, child):
            if created is not None:
                host_processes.kill(created)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)

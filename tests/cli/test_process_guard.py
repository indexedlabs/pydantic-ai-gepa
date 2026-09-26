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


def test_finite_window_survives_restart_and_tracks_later_descendants():
    backend = SnapshotProcesses()
    record = {}
    guard = ProcessGuard(backend, record, lambda: None)
    guard.begin()
    unknown = Process(10, 3, 2, (1, 0), "unknown during step")
    backend.processes.append(unknown)
    guard.close_window()
    assert record["ceiling"] == 3
    unrelated_birth = Process(11, 5, 4, (1, 0), "after step")
    child = Process(12, 6, 3, (1, 0), "descendant after step")
    backend.processes += [unrelated_birth, child]
    guard.observe()
    backend.processes.remove(unknown)
    resumed = ProcessGuard(backend, record, lambda: None)
    assert resumed.finish(0.01) == [child]
    assert record["ceiling"] == 3
    backend.processes.remove(child)
    assert resumed.finish(0.01) == []
    assert backend.killed == []


def test_interrupted_window_closes_on_first_restart_snapshot():
    backend = SnapshotProcesses()
    record = {}
    ProcessGuard(backend, record, lambda: None).begin()
    unknown = Process(10, 3, 2, (1, 0), "before restart")
    backend.processes.append(unknown)
    resumed = ProcessGuard(backend, record, lambda: None)
    assert resumed.finish(0.01) == [unknown]
    assert record["ceiling"] == 3
    backend.processes = [
        backend.processes[0],
        Process(11, 5, 4, (1, 0), "after restart"),
    ]
    assert resumed.finish(0.01) == []


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "parent",
        "path",
        "unsigned",
        "csops",
        "path_unavailable",
        "responsibility_unavailable",
        "driver",
        "root",
        "missing_driver",
        "missing_root",
        "identity_changed",
        "launchd_unavailable",
    ],
)
def test_platform_job_requires_every_proof(failure):
    backend = DarwinProcesses.__new__(DarwinProcesses)
    process = Process(20, 20, 7 if failure != "parent" else 8, (1, 0), "platform")

    def read(pid, flavor, info):
        assert pid == 1 and flavor == 17
        info.unique_id = 7
        if failure == "launchd_unavailable":
            raise GuardError("denied")
        return True

    def path(pid, buffer, size):
        buffer.value = b"/bin/sleep" if failure == "path" else b"/System/Library/job"
        return 0 if failure == "path_unavailable" else len(buffer.value)

    def csops(pid, operation, pointer, size):
        pointer._obj.value = 0 if failure == "unsigned" else 0x04000000
        return -1 if failure == "csops" else 0

    backend._read = read
    backend.lib = SimpleNamespace(proc_pidpath=path)
    backend.libc = SimpleNamespace(csops=csops)
    backend.responsible_pid = lambda pid: {
        "responsibility_unavailable": None,
        "driver": 100,
        "root": 200,
    }.get(failure, 300)
    backend.identity = lambda pid: None if failure == "identity_changed" else process
    driver = None if failure == "missing_driver" else 100
    root = None if failure == "missing_root" else 200
    assert backend.platform_job(process, driver, root) == (failure is None)


def test_platform_proof_does_not_override_attributed_ancestry():
    backend = SnapshotProcesses()
    backend.platform_job = lambda *args: True
    record = {}
    guard = ProcessGuard(backend, record, lambda: None)
    guard.begin()
    record["root"] = 2
    root = Process(20, 2, 1, (1, 0), "platform reflector")
    backend.processes.append(root)
    assert guard.classify(root) == "attributed"
    assert guard.finish(0.1) == []
    assert backend.killed == [20]


def test_host_zombie_root_identity_is_available(tmp_path, host_processes):
    marker = tmp_path / "exiting"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ]
    )
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        # Do not poll/wait: keep the exited leader unreaped.
        time.sleep(0.2)
        assert host_processes.identity(process.pid) is None
        assert host_processes.identity(process.pid, include_zombie=True) is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_host_fast_double_fork_platform_binary_is_unknown(tmp_path, host_processes):
    marker = tmp_path / "reader"
    script = tmp_path / "fork.py"
    script.write_text("""import os, sys, time
from pathlib import Path
if os.fork() == 0:
    os.setsid()
    if os.fork() != 0: os._exit(0)
    Path(sys.argv[1]).write_text(str(os.getpid()))
    os.execl("/bin/sleep", "sleep", "60")
time.sleep(0.3)
""")
    record = {}
    guard = ProcessGuard(host_processes, record, lambda: None)
    guard.begin()
    process = subprocess.Popen(
        [sys.executable, str(script), str(marker)], start_new_session=True
    )
    reader = None
    try:
        root = host_processes.identity(process.pid, include_zombie=True)
        assert root is not None
        record.update(
            root=root.unique_id,
            root_responsible=host_processes.responsible_pid(root.pid),
        )
        process.wait(timeout=5)  # Deliberately miss the intermediate's lifetime.
        reader = host_processes.identity(int(marker.read_text()))
        assert reader is not None
        guard.observe()
        assert guard.classify(reader) == "unknown"
        assert reader in guard.finish(0.01)
        assert host_processes.identity(reader.pid) == reader
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if reader is not None:
            host_processes.kill(reader)

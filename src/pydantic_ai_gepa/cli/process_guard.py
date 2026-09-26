"""Separate reflector lifetimes from scoring using Darwin kernel identities."""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import errno
import os
import signal
import sys
import time
from typing import Any, Callable

from .scoring_processes import _BsdInfo


class GuardError(RuntimeError):
    """Process inspection failed; scoring must remain stopped."""


class _UniqueInfo(ctypes.Structure):
    # XNU bsd/sys/proc_info_private.h, PROC_PIDUNIQIDENTIFIERINFO (17).
    _fields_ = [
        ("uuid", ctypes.c_uint8 * 16),
        ("unique_id", ctypes.c_uint64),
        ("parent_id", ctypes.c_uint64),
        ("idversion", ctypes.c_int32),
        ("parent_version", ctypes.c_int32),
        ("reserved2", ctypes.c_uint64),
        ("reserved3", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class Process:
    pid: int
    unique_id: int
    parent_id: int
    start: tuple[int, int]
    command: str


class DarwinProcesses:
    """Read creation identities, including the original parent's identity."""

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise GuardError("Driving requires the macOS libproc process guard.")
        self.lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.lib.proc_listpids.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.lib.proc_listpids.restype = ctypes.c_int
        self.lib.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.lib.proc_pidinfo.restype = ctypes.c_int
        libc = ctypes.CDLL(None, use_errno=True)
        libc.sysctlbyname.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        buffer = ctypes.create_string_buffer(128)
        size = ctypes.c_size_t(len(buffer))
        if libc.sysctlbyname(
            b"kern.bootsessionuuid", buffer, ctypes.byref(size), None, 0
        ):
            raise GuardError("Cannot read the host boot identity.")
        self.boot_id = buffer.value.decode("ascii")

    def _read(self, pid: int, flavor: int, info: ctypes.Structure) -> bool:
        ctypes.set_errno(0)
        size = self.lib.proc_pidinfo(
            pid, flavor, 0, ctypes.byref(info), ctypes.sizeof(info)
        )
        if size == ctypes.sizeof(info):
            return True
        if size == 0 and ctypes.get_errno() in (errno.ESRCH, errno.ENOENT):
            return False
        raise GuardError(f"Cannot inspect process {pid}; scoring refused.")

    def identity(self, pid: int) -> Process | None:
        before, after = _UniqueInfo(), _UniqueInfo()
        info = _BsdInfo()
        if not self._read(pid, 17, before) or not self._read(pid, 3, info):
            return None
        if not self._read(pid, 17, after):
            return None
        if before.unique_id != after.unique_id:
            raise GuardError(f"Process {pid} changed identity during inspection.")
        if info.uid != os.getuid() or info.status == 5:  # zombie: no executing code
            return None
        return Process(
            pid,
            before.unique_id,
            before.parent_id,
            (info.start_sec, info.start_usec),
            (info.name or info.comm).decode(errors="replace"),
        )

    def snapshot(self) -> list[Process]:
        capacity = 256
        while capacity <= 1024 * 1024:
            pids = (ctypes.c_int * capacity)()
            size = self.lib.proc_listpids(4, os.getuid(), pids, ctypes.sizeof(pids))
            if size <= 0 or size % ctypes.sizeof(ctypes.c_int):
                raise GuardError("Cannot enumerate same-uid processes.")
            if size >= ctypes.sizeof(pids):
                capacity *= 2
                continue
            result = [
                p
                for pid in pids[: size // ctypes.sizeof(ctypes.c_int)]
                if (p := self.identity(pid)) is not None
            ]
            if not any(p.pid == os.getpid() for p in result):
                raise GuardError("Process enumeration is incomplete.")
            return result
        raise GuardError("Process enumeration exceeded its limit.")

    def kill(self, process: Process) -> None:
        current = self.identity(process.pid)
        if current is not None and current.unique_id == process.unique_id:
            try:
                os.kill(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                raise GuardError(
                    f"Cannot terminate attributed process {process.pid}."
                ) from exc


class ProcessGuard:
    """Persist each observed ancestry edge before allowing a scoring phase."""

    def __init__(
        self, backend: DarwinProcesses, record: dict[str, Any], save: Callable[[], None]
    ) -> None:
        self.backend, self.record, self.save = backend, record, save

    def begin(self) -> None:
        processes = self.backend.snapshot()
        self.record.update(
            boot_id=getattr(self.backend, "boot_id", None),
            before=[p.unique_id for p in processes],
            watermark=max(p.unique_id for p in processes),
            owner=next(p.unique_id for p in processes if p.pid == os.getpid()),
            root=None,
            seen={str(p.unique_id): asdict(p) for p in processes},
        )
        self.save()

    def observe(self) -> list[Process]:
        processes = self.backend.snapshot()
        seen = self.record["seen"]
        changed = False
        for process in processes:
            key = str(process.unique_id)
            if key not in seen:
                seen[key] = asdict(process)
                changed = True
        if changed:
            self.save()
        return processes

    def classify(self, process: Process) -> str:
        identity = process.unique_id
        visited: set[int] = set()
        while identity not in visited:
            if identity == self.record.get("root"):
                return "attributed"
            # A kill between spawn and root publication must not make the
            # new child look unrelated merely because the driver predates it.
            if identity == self.record["owner"]:
                return "unknown"
            if identity in self.record["before"]:
                return "unrelated"
            visited.add(identity)
            parent = self.record["seen"].get(str(identity))
            if parent is None:
                return "unknown"
            identity = parent["parent_id"]
        return "unknown"

    def finish(self, grace: float) -> list[Process]:
        if self.record.get("boot_id") != getattr(self.backend, "boot_id", None):
            # Unique IDs are scoped to a boot. No old reflector can survive a
            # reboot; never signal a new process using the old boot's IDs.
            return []
        deadline = time.monotonic() + grace
        while True:
            blocked = []
            for process in self.observe():
                if (
                    process.pid == os.getpid()
                    or process.unique_id <= self.record["watermark"]
                ):
                    continue
                kind = self.classify(process)
                if kind == "unrelated":
                    continue
                blocked.append(process)
                if kind == "attributed":
                    self.backend.kill(process)
            if not blocked or time.monotonic() >= deadline:
                return blocked
            time.sleep(0.05)

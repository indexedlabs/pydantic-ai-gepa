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
        self.libc = libc
        self.lib.proc_pidpath.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.lib.proc_pidpath.restype = ctypes.c_int
        libc.csops.argtypes = [
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        libc.csops.restype = ctypes.c_int
        self.responsibility = getattr(
            libc, "responsibility_get_pid_responsible_for_pid", None
        )
        if self.responsibility is not None:
            self.responsibility.argtypes = [ctypes.c_int]
            self.responsibility.restype = ctypes.c_int
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

    def _read(
        self,
        pid: int,
        flavor: int,
        info: ctypes.Structure,
        *,
        include_zombie: bool = False,
    ) -> bool:
        ctypes.set_errno(0)
        # XNU proc_pidinfo opts into zombie lookup via arg=1 for BSD/unique info.
        size = self.lib.proc_pidinfo(
            pid, flavor, int(include_zombie), ctypes.byref(info), ctypes.sizeof(info)
        )
        if size == ctypes.sizeof(info):
            return True
        if size == 0 and ctypes.get_errno() in (errno.ESRCH, errno.ENOENT):
            return False
        raise GuardError(f"Cannot inspect process {pid}; scoring refused.")

    def identity(self, pid: int, *, include_zombie: bool = False) -> Process | None:
        before, after = _UniqueInfo(), _UniqueInfo()
        info = _BsdInfo()
        if not self._read(
            pid, 17, before, include_zombie=include_zombie
        ) or not self._read(pid, 3, info, include_zombie=include_zombie):
            return None
        if not self._read(pid, 17, after, include_zombie=include_zombie):
            return None
        if before.unique_id != after.unique_id:
            raise GuardError(f"Process {pid} changed identity during inspection.")
        if info.uid != os.getuid() or (info.status == 5 and not include_zombie):
            return None
        return Process(
            pid,
            before.unique_id,
            before.parent_id,
            (info.start_sec, info.start_usec),
            (info.name or info.comm).decode(errors="replace"),
        )

    def responsible_pid(self, pid: int) -> int | None:
        responsible = self.responsibility(pid) if self.responsibility else -1
        return responsible if responsible > 0 else None

    def platform_job(
        self, process: Process, driver: int | None, root: int | None
    ) -> bool:
        """Require all three independent proofs; orphan PPID is not evidence."""
        if driver is None or root is None:
            return False
        try:
            launchd = _UniqueInfo()
            if not self._read(1, 17, launchd) or process.parent_id != launchd.unique_id:
                return False
            path = ctypes.create_string_buffer(4096)
            if self.lib.proc_pidpath(process.pid, path, len(path)) <= 0:
                return False
            if not path.value.startswith((b"/System/", b"/usr/libexec/")):
                return False
            flags = ctypes.c_uint32()
            # XNU cs_blobs.h: CS_OPS_STATUS=0, CS_PLATFORM_BINARY=0x04000000.
            if self.libc.csops(
                process.pid, 0, ctypes.byref(flags), ctypes.sizeof(flags)
            ):
                return False
            if not flags.value & 0x04000000:
                return False
            responsible = self.responsible_pid(process.pid)
            current = self.identity(process.pid)
            return (
                responsible is not None
                and responsible not in (driver, root)
                and current is not None
                and current.unique_id == process.unique_id
            )
        except (GuardError, OSError):
            return False

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
            driver_responsible=self.responsible_pid(os.getpid()),
            root_responsible=None,
            seen={str(p.unique_id): asdict(p) for p in processes},
        )
        self.save()

    def responsible_pid(self, pid: int) -> int | None:
        getter = getattr(self.backend, "responsible_pid", None)
        return getter(pid) if getter else None

    def close_window(self) -> None:
        self.observe()
        if "ceiling" not in self.record:
            self.record["ceiling"] = max(map(int, self.record["seen"]))
            self.save()

    def in_window(self, process: Process) -> bool:
        identity = process.unique_id
        visited = set()
        while identity not in visited:
            if self.record["watermark"] < identity <= self.record["ceiling"]:
                return True
            visited.add(identity)
            parent = self.record["seen"].get(str(identity))
            if parent is None:
                break
            identity = parent["parent_id"]
        return False

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
                break
            identity = parent["parent_id"]
        check_job = getattr(self.backend, "platform_job", None)
        if check_job and check_job(
            process,
            self.record.get("driver_responsible"),
            self.record.get("root_responsible"),
        ):
            return "unrelated"
        return "unknown"

    def finish(self, grace: float) -> list[Process]:
        if self.record.get("boot_id") != getattr(self.backend, "boot_id", None):
            # Unique IDs are scoped to a boot. No old reflector can survive a
            # reboot; never signal a new process using the old boot's IDs.
            return []
        # An interrupted step gets a finite ceiling on its first restart snapshot.
        if "ceiling" not in self.record:
            self.close_window()
        deadline = time.monotonic() + grace
        while True:
            blocked = []
            for process in self.observe():
                if process.pid == os.getpid() or not self.in_window(process):
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

"""Find live descendants by their inherited Seatbelt write permissions.

This is a repeated process-table sweep, not an atomic kernel process container.
It does not inspect arguments or SysV IPC objects.
"""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import signal
import time

from .scoring_sandbox import ScoringSandboxError


class _BsdInfo(ctypes.Structure):
    # Darwin sys/proc_info.h: PROC_PIDTBSDINFO, including process start time so
    # a PID reused between enumeration and signaling is not treated as a member.
    _fields_ = [
        (name, ctypes.c_uint32)
        for name in (
            "flags",
            "status",
            "xstatus",
            "pid",
            "ppid",
            "uid",
            "gid",
            "ruid",
            "rgid",
            "svuid",
            "svgid",
            "reserved",
        )
    ] + [
        ("comm", ctypes.c_char * 16),
        ("name", ctypes.c_char * 32),
        ("nfiles", ctypes.c_uint32),
        ("pgid", ctypes.c_uint32),
        ("jobc", ctypes.c_uint32),
        ("tdev", ctypes.c_uint32),
        ("tpgid", ctypes.c_uint32),
        ("nice", ctypes.c_int32),
        ("start_sec", ctypes.c_uint64),
        ("start_usec", ctypes.c_uint64),
    ]


class SandboxProcesses:
    def __init__(self, scratch: Path, outside: Path) -> None:
        self.scratch = os.fsencode(scratch.resolve())
        self.outside = os.fsencode(outside.resolve())
        self.uid = os.getuid()
        try:
            self._sandbox = ctypes.CDLL("/usr/lib/libsandbox.dylib", use_errno=True)
            self._proc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            self._check = self._sandbox.sandbox_check
            # sandbox_check is variadic: declare the three fixed arguments.
            self._check.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
            self._check.restype = ctypes.c_int
            self._list = self._proc.proc_listpids
            self._list.argtypes = [
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            self._list.restype = ctypes.c_int
            self._info = self._proc.proc_pidinfo
            self._info.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            self._info.restype = ctypes.c_int
        except (OSError, AttributeError):
            raise ScoringSandboxError(
                "Cannot load Seatbelt process inspection; held-out scoring is refused."
            ) from None
        if self._denied(os.getpid(), self.outside) or self._denied(
            os.getpid(), self.scratch
        ):
            raise ScoringSandboxError(
                "Harness cannot verify scoring cleanup permissions."
            )
        # Check enumeration before starting an untrusted worker.
        self._pids()

    def _pids(self) -> list[int]:
        capacity = 256
        while capacity <= 1024 * 1024:
            buffer = (ctypes.c_int * capacity)()
            count = self._list(
                4, self.uid, buffer, ctypes.sizeof(buffer)
            )  # PROC_UID_ONLY
            if count <= 0 or count % ctypes.sizeof(ctypes.c_int):
                raise ScoringSandboxError("Cannot enumerate scoring cleanup processes.")
            if count >= ctypes.sizeof(buffer):
                capacity *= 2
                continue
            pids = list(buffer[: count // ctypes.sizeof(ctypes.c_int)])
            if os.getpid() not in pids:
                raise ScoringSandboxError("Scoring process enumeration is incomplete.")
            return pids
        raise ScoringSandboxError("Scoring process enumeration exceeded its limit.")

    def _identity(self, pid: int) -> tuple[int, int] | None:
        info = _BsdInfo()
        ctypes.set_errno(0)
        size = self._info(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
        if size != ctypes.sizeof(info):
            if size == 0 and ctypes.get_errno() in (errno.ESRCH, errno.ENOENT):
                return None
            raise ScoringSandboxError("Cannot inspect a scoring cleanup process.")
        if info.pid != pid:
            raise ScoringSandboxError("Invalid scoring process identity.")
        if (
            info.uid != self.uid or info.status == 5
        ):  # SZOMB: no executing code or open fds
            return None
        return info.start_sec, info.start_usec

    def _denied(self, pid: int, path: bytes) -> bool:
        ctypes.set_errno(0)
        result = self._check(
            pid, b"file-write-data", 1, ctypes.c_char_p(path)
        )  # SANDBOX_FILTER_PATH
        if result not in (0, 1):
            if ctypes.get_errno() in (errno.ESRCH, errno.ENOENT):
                raise ProcessLookupError
            raise ScoringSandboxError(
                "Cannot query scoring process Seatbelt permissions."
            )
        return result == 1

    def _matches(self, pid: int) -> bool:
        # An unsandboxed process allows both. A different scoring child denies
        # both. Only this child's inherited profile allows its unique scratch
        # while denying the adjacent parent-owned file.
        return self._denied(pid, self.outside) and not self._denied(pid, self.scratch)

    def verify_worker(self, pid: int) -> None:
        try:
            matches = self._identity(pid) is not None and self._matches(pid)
        except ProcessLookupError:
            matches = False
        if not matches:
            raise ScoringSandboxError(
                "Scoring worker permissions could not be verified."
            )

    def sweep(self) -> bool:
        """Kill matching live processes until one full enumeration finds none.

        Recheck permissions and start time immediately before signaling. PID
        signaling is not atomic with those checks; see the documented limits.
        """
        found = False
        deadline = time.monotonic() + 5
        while True:
            matched = False
            for pid in self._pids():
                if pid == os.getpid():
                    continue
                try:
                    identity = self._identity(pid)
                    if identity is None or not self._matches(pid):
                        continue
                    found = matched = True
                    if self._identity(pid) != identity or not self._matches(pid):
                        continue
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    continue
                except OSError:
                    raise ScoringSandboxError(
                        "Cannot terminate a scoring sandbox survivor."
                    ) from None
            if not matched:
                return found
            if time.monotonic() >= deadline:
                raise ScoringSandboxError(
                    "Scoring sandbox survivors did not terminate."
                )
            time.sleep(0.01)

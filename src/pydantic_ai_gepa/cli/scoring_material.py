"""Bounded, untrusted training material; never used as scoring evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

MAX_FILE_BYTES = 1024 * 1024
MAX_CASE_BYTES = 4 * 1024 * 1024
MAX_EVAL_BYTES = 32 * 1024 * 1024
MAX_PAYLOAD_BYTES = 256 * 1024
MAX_ENTRIES = 256
# Charge names/index bookkeeping too, so empty files cannot bypass byte budgets.
FILE_OVERHEAD_BYTES = 1024
MAX_PATH_DEPTH = 8
MAX_PATH_BYTES = 256
MAX_JSON_DEPTH = 16
MAX_JSON_KEYS = 2048
MAX_JSON_NODES = 16384
TEXT_EXTENSIONS = frozenset({".txt", ".md", ".csv", ".log"})
REFUSAL_NOTE = (
    "Sandbox training material refused: unsupported or exceeds channel limits."
)
PAYLOAD_FIELDS = frozenset({"output", "side_info", "metric_side_info"})


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def checked_json(value: Any) -> str:
    """Validate JSON types and structural complexity before serialization."""
    nodes = keys = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes, keys
        nodes += 1
        if depth > MAX_JSON_DEPTH or nodes > MAX_JSON_NODES:
            raise ValueError
        if type(item) is dict:
            keys += len(item)
            if keys > MAX_JSON_KEYS or any(type(k) is not str for k in item):
                raise ValueError
            for child in item.values():
                visit(child, depth + 1)
        elif type(item) is list:
            for child in item:
                visit(child, depth + 1)
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError
        elif item is not None and type(item) not in (str, int, bool):
            raise ValueError

    visit(value, 0)
    return json.dumps(value, allow_nan=False, sort_keys=True, ensure_ascii=True)


def structured_payload(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != PAYLOAD_FIELDS:
        raise ValueError
    if any(
        value[k] is not None and type(value[k]) is not dict
        for k in ("side_info", "metric_side_info")
    ):
        raise ValueError
    if len(checked_json(value).encode()) > MAX_PAYLOAD_BYTES:
        raise ValueError
    return value


def valid_relative(name: str) -> bool:
    parts = name.split("/")
    return (
        len(name) <= MAX_PATH_BYTES
        and len(parts) <= MAX_PATH_DEPTH
        and all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", p) for p in parts)
    )


@dataclass
class MaterialBudget:
    used: int = 0

    def charge(self, size: int, case_used: int) -> bool:
        if size + case_used > MAX_CASE_BYTES or size + self.used > MAX_EVAL_BYTES:
            return False
        self.used += size
        return True


def collect_files(
    fd: int, budget: MaterialBudget, case_used: int, *, case_id: str | None = None
) -> tuple[dict[str, str], bool]:
    """Walk only an already-open per-case inode; never resolve child paths.

    O_NONBLOCK avoids a FIFO race; fstat checks the inode actually opened.
    Bounded reads and repeated inode checks also cover concurrent child writers.
    """
    files: dict[str, str] = {}
    refused = False
    entries = 0

    def walk(directory: int, prefix: str = "") -> None:
        nonlocal entries, refused, case_used
        # scandir streams entries so even an enormous directory is bounded.
        with os.scandir(directory) as iterator:
            for entry in iterator:
                entries += 1
                if entries > MAX_ENTRIES:
                    refused = True
                    return
                relative = prefix + entry.name
                if not valid_relative(relative):
                    refused = True
                    continue
                opened = None
                try:
                    info = os.stat(entry.name, dir_fd=directory, follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        if relative.count("/") + 1 >= MAX_PATH_DEPTH:
                            raise ValueError
                        opened = os.open(
                            entry.name,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=directory,
                        )
                        walk(opened, relative + "/")
                        continue
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise ValueError
                    suffix = Path(relative).suffix
                    if suffix not in TEXT_EXTENSIONS | {".json", ".jsonl"}:
                        raise ValueError
                    opened = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=directory,
                    )
                    before = os.fstat(opened)
                    if (
                        not stat.S_ISREG(before.st_mode)
                        or before.st_nlink != 1
                        or before.st_size > MAX_FILE_BYTES
                    ):
                        raise ValueError
                    with os.fdopen(os.dup(opened), "rb") as stream:
                        data = stream.read(MAX_FILE_BYTES + 1)
                    after = os.fstat(opened)
                    if len(data) > MAX_FILE_BYTES or (
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                        before.st_nlink,
                    ) != (
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                        after.st_nlink,
                    ):
                        raise ValueError
                    content = data.decode("utf-8")
                    if suffix == ".jsonl":
                        rows = []
                        normalized_bytes = 0
                        # JSONL uses LF, not Unicode's other line separators,
                        # which may be legitimate characters inside strings.
                        lines = (
                            content.removesuffix("\n").split("\n") if content else []
                        )
                        for line in lines:
                            row = json.loads(line, object_pairs_hook=unique_keys)
                            if type(row) is not dict:
                                raise ValueError
                            if relative == "trace.jsonl" and case_id is not None:
                                row["case_id"] = case_id
                            serialized = checked_json(row) + "\n"
                            normalized_bytes += len(serialized.encode("utf-8"))
                            if normalized_bytes > MAX_FILE_BYTES:
                                raise ValueError
                            rows.append(serialized)
                        content = "".join(rows)
                    elif suffix == ".json":
                        content = (
                            checked_json(
                                json.loads(content, object_pairs_hook=unique_keys)
                            )
                            + "\n"
                        )
                    elif any(
                        ord(c) < 32 and c not in "\t\r\n" or ord(c) == 127
                        for c in content
                    ):
                        raise ValueError
                    size = max(len(data), len(content.encode("utf-8")))
                    if size > MAX_FILE_BYTES:
                        raise ValueError
                    size += FILE_OVERHEAD_BYTES + len(checked_json(case_id).encode())
                    if not budget.charge(size, case_used):
                        raise ValueError
                    case_used += size
                    files[relative] = content
                except (OSError, ValueError, UnicodeError, RecursionError):
                    refused = True
                finally:
                    if opened is not None:
                        os.close(opened)

    try:
        walk(fd)
    except OSError:
        refused = True
    return files, refused


def publish_files(*, root: Path, path: Path, records: list[Any]) -> None:
    """Parent-selected namespaces attach sibling files to the sent case only."""
    from .harness_record import SafeDir

    rows = []
    for index, record in enumerate(records):
        files = record.payload.get("sandbox_files", {})
        artifacts = []
        for relative, content in files.items():
            if relative == "trace.jsonl":
                for line in content.splitlines():
                    row = json.loads(line)
                    row["case_id"] = record.case_id
                    rows.append(json.dumps(row, sort_keys=True) + "\n")
                continue
            # Never share a sibling name between cases, evals or candidates.
            destination = (
                path.parent / (path.name + ".cases") / f"case-{index:06d}" / relative
            )
            with SafeDir.open(root, destination.parent, create=True) as directory:
                directory.write_text(destination.name, content)
            artifacts.append(destination.relative_to(path.parent).as_posix())
        if artifacts:
            rows.append(
                json.dumps(
                    {"case_id": record.case_id, "artifacts": sorted(artifacts)},
                    sort_keys=True,
                )
                + "\n"
            )
    if rows:
        with SafeDir.open(root, path.parent, create=True) as directory:
            directory.write_text(path.name, "".join(rows))

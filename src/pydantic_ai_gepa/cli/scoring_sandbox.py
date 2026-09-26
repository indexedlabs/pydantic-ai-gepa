"""Harness-owned candidate checkouts and a fail-closed scoring subprocess.

The parent treats every child byte as untrusted. No candidate callable, pickle,
exception text, validation feedback, or candidate-selected path crosses back.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator

import typer
from pydantic_ai.messages import ModelResponse
from pydantic_ai.usage import RequestUsage

from ..evaluation import EvaluationRecord
from ..types import RolloutOutput
from .layout import GepaConfig, git_root
from .scoring_proxy import allowed_addresses, connect_proxy
from .validation import heldout_dataset

from .safe_git import SafeGit, safe_repository, unsafe_checkout_component

MAX_MESSAGE = 1024 * 1024
RESPONSE_TIMEOUT = 300
PROVIDER_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)
nominated_commit: ContextVar[str | None] = ContextVar("scoring_commit", default=None)


class ScoringSandboxError(typer.BadParameter):
    """Static messages only: errors are visible to the reflector."""


def required() -> bool:
    return heldout_dataset(required=False) is not None


def require_supported(config: GepaConfig, source: str) -> None:
    acceptance = config.acceptance
    if (
        source != "git"
        or acceptance.mode != "scalar"
        or (acceptance.pinned_scorer and not acceptance.trusted_scorer)
    ):
        raise ScoringSandboxError(
            "Held-out sandbox scoring requires git candidates with scalar acceptance, "
            "either unpinned or pinned with trusted_scorer. "
            "Unsupported modes are refused before importing candidate code."
        )
    if acceptance.trusted_scorer:
        if not acceptance.pinned_scorer or not acceptance.component_files:
            raise ScoringSandboxError(
                "acceptance.trusted_scorer requires pinned_scorer = true and non-empty component_files."
            )
        scorer_revision()


def scorer_revision() -> str:
    revision = os.environ.get("GEPA_HARNESS_SCORER_REVISION", "")
    if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision):
        raise ScoringSandboxError(
            "Trusted scoring requires GEPA_HARNESS_SCORER_REVISION to be a full commit SHA."
        )
    return revision.lower()


def _relative_file(value: Any) -> bool:
    return (
        isinstance(value, str)
        and not Path(value).is_absolute()
        and "\0" not in value
        and all(
            part not in ("", ".", "..")
            and not unsafe_checkout_component(os.fsencode(part))
            for part in value.split("/")
        )
    )


def frozen_files() -> dict[str, str] | None:
    """Harness-owned SHA-256 hashes, with paths relative to the Git root."""
    if "GEPA_HARNESS_FROZEN_FILES" not in os.environ:
        return None
    try:
        files = json.loads(os.environ["GEPA_HARNESS_FROZEN_FILES"])
    except ValueError:
        files = None
    if not isinstance(files, dict) or any(
        not _relative_file(path)
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
        for path, digest in files.items()
    ):
        raise ScoringSandboxError(
            "GEPA_HARNESS_FROZEN_FILES must map repository-relative file paths to SHA-256 hashes."
        )
    return files


def _verify_frozen_files(
    checkout: Path, entries: list[tuple[Path, bytes, bytes]], files: dict[str, str]
) -> None:
    exact = {
        path.relative_to(checkout).as_posix(): (path, mode) for path, mode, _ in entries
    }
    for relative, digest in files.items():
        try:
            path, mode = exact[relative]
            if mode not in (b"100644", b"100755") or not stat.S_ISREG(
                path.lstat().st_mode
            ):
                raise ValueError
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest.lower():
                raise ValueError
        except (OSError, ValueError, KeyError):
            raise ScoringSandboxError(
                "Frozen scorer file verification failed."
            ) from None


def candidate_components(
    project: Path, sha: str, files: tuple[str, ...]
) -> dict[str, str]:
    """Read only nominated raw blobs; candidate trees never become scorer code."""
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha):
        raise ScoringSandboxError("Scoring requires a committed candidate.")
    if not all(_relative_file(path) for path in files):
        raise ScoringSandboxError(
            "Candidate components require project-relative file paths."
        )
    try:
        with safe_repository(project) as git:
            prefix = project.resolve().relative_to(git.repository.root)
            if not all(_relative_file((prefix / path).as_posix()) for path in files):
                raise ValueError
            entries = {
                path.relative_to(git.repository.root).as_posix(): (mode, oid)
                for path, mode, oid in _verified_checkout_entries(
                    git, git.repository.root, sha
                )
            }
            components = {}
            with _object_reader(git) as reader:
                for relative in files:
                    mode, oid = entries[(prefix / relative).as_posix()]
                    if mode not in (b"100644", b"100755"):
                        raise ValueError
                    components[relative] = reader.read(oid, b"blob").decode("utf-8")
            return components
    except (OSError, ValueError, KeyError):
        raise ScoringSandboxError(
            "Cannot read candidate component UTF-8 blobs."
        ) from None


def seatbelt_profile(private: Path, checkout: Path, scratch: Path, port: int) -> str:
    def quoted(path: Path) -> str:
        return json.dumps(str(path.resolve()))

    return f"""(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow sysctl-read)
(allow file-read* (require-not (subpath {quoted(private)})))
(allow file-read* (subpath {quoted(checkout)}) (subpath {quoted(scratch)}))
(allow file-write* (subpath {quoted(scratch)}) (literal "/dev/null"))
(allow network-outbound (remote tcp "localhost:{port}"))
"""


def sandbox_command(profile: str, command: list[str]) -> list[str]:
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise ScoringSandboxError(
            "Held-out scoring requires a usable macOS Seatbelt backend; no sandbox backend is available."
        )
    return ["/usr/bin/sandbox-exec", "-p", profile, *command]


def backend_unavailable_reason() -> str | None:
    """Compile the real profile shape and probe nested-Seatbelt availability.

    Use non-sensitive dummy paths. A profile error is a bug, not an unavailable
    backend, and must fail tests even when unavailable backends may be skipped.
    """
    try:
        private = Path("/gepa-scoring-backend-probe")
        command = sandbox_command(
            seatbelt_profile(private, private / "checkout", private / "scratch", 9),
            ["/usr/bin/true"],
        )
    except ScoringSandboxError:
        return "No macOS Seatbelt backend is available"
    try:
        result = subprocess.run(command, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "Seatbelt backend probe could not run"
    if result.returncode:
        if b"sandbox_apply:" in result.stderr:
            return "Seatbelt backend is unusable (sandbox_apply refused; nested OS sandbox may prohibit it)"
        raise ScoringSandboxError(
            "Seatbelt scoring profile verification failed; check profile syntax and permissions."
        )
    return None


def child_environment(scratch: Path, port: int) -> dict[str, str]:
    env = dict(
        PATH="/usr/bin:/bin",
        HOME=str(scratch),
        TMPDIR=str(scratch),
        XDG_CACHE_HOME=str(scratch),
        XDG_CONFIG_HOME=str(scratch),
        XDG_DATA_HOME=str(scratch),
        PYTHONPYCACHEPREFIX=str(scratch / "pycache"),
        PYTHONDONTWRITEBYTECODE="1",
        HTTP_PROXY=f"http://127.0.0.1:{port}",
        HTTPS_PROXY=f"http://127.0.0.1:{port}",
        NO_PROXY="",
    )
    names = set(PROVIDER_KEYS).intersection(os.environ)
    reserved = set(env) | {
        "GEPA_HELDOUT_DATASET",
        "GEPA_HARNESS_ALLOWED_HOSTS",
        "GEPA_HARNESS_PASS_ENV",
        "GEPA_HARNESS_SCORER_REVISION",
        "GEPA_HARNESS_FROZEN_FILES",
        "GEPA_CANDIDATE_COMPONENTS_JSON",
    }
    for item in os.environ.get("GEPA_HARNESS_PASS_ENV", "").split(","):
        name = item.strip()
        if not name:
            continue
        if (
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            or name in reserved
            or name.startswith(("PYTHON", "DYLD_", "LD_"))
        ):
            raise ScoringSandboxError(
                "GEPA_HARNESS_PASS_ENV contains a reserved or invalid variable name."
            )
        if name not in os.environ:
            raise ScoringSandboxError(
                "Requested harness environment variable is not set."
            )
        names.add(name)
    dataset = heldout_dataset(required=False)
    private_paths = {dataset, str(Path(dataset).resolve())} if dataset else set()
    for name in names:
        value = os.environ[name]
        if any(path in value for path in private_paths):
            raise ScoringSandboxError(
                "Refusing an environment value containing the held-out path."
            )
        env[name] = value
    return env


@contextmanager
def private_checkout(
    project: Path, sha: str, *, frozen: dict[str, str] | None = None
) -> Iterator[tuple[Path, Path, Path]]:
    """Read Git objects only. Never register a worktree or touch shared refs/index."""
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha):
        raise ScoringSandboxError("Scoring requires a committed candidate.")
    private = Path(str(heldout_dataset())).resolve().parent
    storage = private / ".gepa-heldout" / "work"
    storage.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(storage.parent, 0o700)
    os.chmod(storage, 0o700)
    repository = git_root(project)
    prefix = project.resolve().relative_to(repository)
    with tempfile.TemporaryDirectory(dir=storage) as temporary:
        base = Path(temporary)
        checkout, scratch = base / "checkout", base / "scratch"
        checkout.mkdir(mode=0o700)
        scratch.mkdir(mode=0o700)
        # Read raw objects, never worktree conversions. In particular, do not
        # use archive, checkout, or cat-file's --filters/--textconv options.
        try:
            with safe_repository(repository) as git:
                entries = _verified_checkout_entries(git, checkout, sha)
                if frozen is not None:
                    _refuse_frozen_shadows(checkout, entries, frozen)
                _write_checkout_blobs(git, entries)
            _verify_frozen_files(checkout, entries, frozen or {})
        except OSError:
            raise ScoringSandboxError(
                "Cannot create private candidate checkout."
            ) from None
        yield private, checkout / prefix, scratch


def _checkout_entries(
    checkout: Path, listing: bytes
) -> list[tuple[Path, bytes, bytes]]:
    entries = []
    if listing and not listing.endswith(b"\0"):
        raise ScoringSandboxError("Invalid candidate tree listing.")
    for record in listing.split(b"\0")[:-1]:
        metadata, separator, name = record.partition(b"\t")
        fields = metadata.split(b" ")
        if not separator or len(fields) != 3:
            raise ScoringSandboxError("Invalid candidate tree listing.")
        mode, kind, oid = fields
        if (mode, kind) not in {
            (b"040000", b"tree"),
            (b"100644", b"blob"),
            (b"100755", b"blob"),
        }:
            raise ScoringSandboxError(
                "Private candidate checkouts cannot contain links or special files."
            )
        target = checkout / os.fsdecode(name)
        if (
            not re.fullmatch(rb"[0-9a-f]{40}|[0-9a-f]{64}", oid)
            or any(
                part in (b"", b".", b"..") or unsafe_checkout_component(part)
                for part in name.split(b"/")
            )
            or Path(os.fsdecode(name)).is_absolute()
            or not target.resolve().is_relative_to(checkout)
        ):
            raise ScoringSandboxError("Invalid candidate checkout path or object ID.")
        entries.append((target, mode, oid))
    return entries


class _VerifiedObjects:
    def __init__(self, process: subprocess.Popen[bytes], object_format: str):
        if object_format not in {"sha1", "sha256"}:
            raise ScoringSandboxError("Unsupported Git object format.")
        self.process = process
        self.object_format = object_format
        self.oid_size = 20 if object_format == "sha1" else 32

    def blocks(self, oid: bytes, kind: bytes) -> Iterator[bytes]:
        if not re.fullmatch(rb"[0-9a-f]{%d}" % (self.oid_size * 2), oid):
            raise ScoringSandboxError("Invalid Git object ID.")
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(oid + b"\n")
        self.process.stdin.flush()
        header = self.process.stdout.readline(256)
        match = re.fullmatch(oid + b" " + kind + rb" ([0-9]+)\n", header)
        if match is None:
            # kind is selected by library code, never by the candidate.
            raise ScoringSandboxError(f"Invalid candidate {kind.decode()} response.")
        remaining = size = int(match[1])
        digest = hashlib.new(self.object_format)
        digest.update(kind + b" " + str(size).encode("ascii") + b"\0")
        while remaining:
            block = self.process.stdout.read(min(remaining, 1024 * 1024))
            if not block:
                raise ScoringSandboxError("Truncated Git object.")
            digest.update(block)
            remaining -= len(block)
            yield block
        if self.process.stdout.read(1) != b"\n":
            raise ScoringSandboxError("Invalid Git object terminator.")
        if digest.hexdigest().encode("ascii") != oid:
            raise ScoringSandboxError("Git object integrity verification failed.")

    def read(self, oid: bytes, kind: bytes) -> bytes:
        return b"".join(self.blocks(oid, kind))


@contextmanager
def _object_reader(git: SafeGit) -> Iterator[_VerifiedObjects]:
    with git.popen(
        "cat-file",
        "--batch",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ) as process:
        assert process.stdin is not None and process.stdout is not None
        try:
            yield _VerifiedObjects(process, git.object_format)
            process.stdin.close()
            if process.stdout.read(1) or process.wait():
                raise ScoringSandboxError("Cannot read Git objects.")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def _verified_checkout_entries(
    git: SafeGit, checkout: Path, sha: str
) -> list[tuple[Path, bytes, bytes]]:
    """Follow only hash-verified commit/tree bytes, never Git's ls-tree traversal."""
    entries = []
    with _object_reader(git) as reader:
        commit = reader.read(sha.encode("ascii"), b"commit")
        tree_line = commit.partition(b"\n")[0]
        match = re.fullmatch(rb"tree ([0-9a-f]{%d})" % (reader.oid_size * 2), tree_line)
        if match is None:
            raise ScoringSandboxError("Invalid scoring commit tree.")
        pending = [(b"", match[1])]
        while pending:
            prefix, oid = pending.pop()
            tree = reader.read(oid, b"tree")
            offset = 0
            names = set()
            while offset < len(tree):
                end = tree.find(b"\0", offset)
                if end < 0 or end + 1 + reader.oid_size > len(tree):
                    raise ScoringSandboxError("Invalid raw Git tree.")
                mode, separator, name = tree[offset:end].partition(b" ")
                if not separator or not name or b"/" in name or name in names:
                    raise ScoringSandboxError("Invalid raw Git tree.")
                names.add(name)
                oid = tree[end + 1 : end + 1 + reader.oid_size].hex().encode("ascii")
                offset = end + 1 + reader.oid_size
                mode = b"040000" if mode == b"40000" else mode
                kind = b"tree" if mode == b"040000" else b"blob"
                relative = prefix + name
                # Preserve the checkout's mode/path refusals for both consumers.
                entries.extend(
                    _checkout_entries(
                        checkout,
                        mode + b" " + kind + b" " + oid + b"\t" + relative + b"\0",
                    )
                )
                if kind == b"tree":
                    pending.append((relative + b"/", oid))
    return entries


def _refuse_frozen_shadows(
    checkout: Path, entries: list[tuple[Path, bytes, bytes]], files: dict[str, str]
) -> None:
    packages = {path[:-3].casefold() for path in files if path.endswith(".py")}
    for path, mode, _ in entries:
        relative = path.relative_to(checkout)
        if (
            "__pycache__" in (part.casefold() for part in relative.parts)
            or (
                mode != b"040000"
                and relative.suffix.casefold() in {".pyc", ".so", ".pyd"}
            )
            or (mode == b"040000" and relative.as_posix().casefold() in packages)
        ):
            raise ScoringSandboxError(
                "Frozen scorer checkouts cannot contain bytecode, extensions or module shadows."
            )


def _write_checkout_blobs(
    git: SafeGit, entries: list[tuple[Path, bytes, bytes]]
) -> None:
    with _object_reader(git) as reader:
        for target, mode, oid in entries:
            if mode == b"040000":
                target.mkdir(mode=0o700)
                continue
            with target.open("xb") as output:
                for block in reader.blocks(oid, b"blob"):
                    output.write(block)
            target.chmod(0o755 if mode == b"100755" else 0o644)


class Channel:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self.process = process
        self.pending = bytearray()

    def send(self, value: Any) -> None:
        data = json.dumps(value, allow_nan=False).encode() + b"\n"
        if len(data) > MAX_MESSAGE:
            raise ScoringSandboxError("Scoring protocol input exceeds its size limit.")
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(data)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            raise ScoringSandboxError(
                "Sandboxed scorer exited before returning a result."
            ) from None

    def receive(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        deadline = time.monotonic() + RESPONSE_TIMEOUT
        while b"\n" not in self.pending:
            if self.process.poll() is not None:
                raise ScoringSandboxError(
                    "Sandboxed scorer exited before returning a result."
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ScoringSandboxError("Sandboxed scorer response timed out.")
            # Descendants can retain stdout after the main worker dies. Bound
            # each wait so that detecting death does not depend on pipe EOF.
            if not select.select([self.process.stdout], [], [], min(remaining, 0.1))[0]:
                continue
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                raise ScoringSandboxError(
                    "Sandboxed scorer exited or its OS sandbox was refused."
                )
            self.pending.extend(block)
            if len(self.pending) > MAX_MESSAGE:
                raise ScoringSandboxError(
                    "Scoring protocol response exceeds its size limit."
                )
        line, _, self.pending = self.pending.partition(b"\n")
        try:
            value = json.loads(line, object_pairs_hook=_unique_keys)
            if not isinstance(value, dict):
                raise ValueError
        except (ValueError, UnicodeError, RecursionError):
            raise ScoringSandboxError("Invalid scoring protocol response.") from None
        return value


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _number(value: Any, *, integer: bool = False) -> float:
    if type(value) not in (int, float) or abs(value) > 1e12 or not math.isfinite(value):
        raise ScoringSandboxError("Invalid numeric scoring response.")
    if integer and (type(value) is not int or value < 0):
        raise ScoringSandboxError("Invalid scoring usage response.")
    return value


def _result(raw: dict[str, Any], case_id: str, validation: bool) -> EvaluationRecord:
    if (
        set(raw) != {"type", "score", "feedback", "failed", "cached"}
        or raw["type"] != "result"
    ):
        raise ScoringSandboxError("Invalid scoring result shape.")
    score = _number(raw["score"])
    if type(raw["failed"]) is not bool or type(raw["cached"]) is not bool:
        raise ScoringSandboxError("Invalid scoring result flags.")
    feedback = raw["feedback"]
    if feedback is not None and (
        not isinstance(feedback, str) or len(feedback) > 65536
    ):
        raise ScoringSandboxError("Invalid scoring feedback.")
    payload: dict[str, Any] = {}
    if raw["failed"]:
        payload["output"] = RolloutOutput.from_error(
            RuntimeError("Sandboxed evaluation failed"), kind="system"
        )
    return EvaluationRecord(case_id, score, None if validation else feedback, payload)


def _process_cleanup(scratch: Path) -> Any:
    # Load the macOS inspection dependency only on the sandbox scoring path.
    from .scoring_processes import SandboxProcesses

    outside = scratch.parent / "parent-cleanup-probe"
    outside.touch(mode=0o600, exist_ok=False)
    return SandboxProcesses(scratch, outside)


def score_cases(
    *,
    config: GepaConfig,
    project: Path,
    sha: str,
    cases: list[Any],
    validation: bool,
    meter: Any,
    scorer_project: Path | None = None,
) -> list[EvaluationRecord]:
    """Serial rollouts keep admission and durable spend in the trusted parent.

    A new child is used for each eval/phase, so validation memory or scratch
    cannot be replayed in a later training report.
    """
    require_supported(config, "git")
    nomination = nominated_commit.get()
    if nomination is not None and nomination != sha:
        raise ScoringSandboxError("Candidate commit changed during scoring.")
    frozen = frozen_files()
    revision = scorer_revision() if config.acceptance.trusted_scorer else sha
    # Fail before opening the proxy or creating candidate storage on hosts
    # without a backend. There is deliberately no environment override.
    sandbox_command("", [])
    components = (
        candidate_components(project, sha, config.acceptance.component_files)
        if config.acceptance.trusted_scorer
        else None
    )
    try:
        addresses = allowed_addresses(os.environ.get("GEPA_HARNESS_ALLOWED_HOSTS", ""))
    except ValueError:
        raise ScoringSandboxError(
            "Invalid GEPA_HARNESS_ALLOWED_HOSTS; use host:port pairs."
        ) from None
    with (
        private_checkout(
            (scorer_project or project)
            if config.acceptance.trusted_scorer
            else project,
            revision,
            frozen=frozen,
        ) as (
            private,
            checkout,
            scratch,
        ),
        connect_proxy(addresses) as port,
    ):
        # -I excludes cwd, PYTHONPATH and the user site from interpreter startup.
        library = str(Path(__file__).resolve().parents[2])
        bootstrap = f"import sys; sys.path.insert(0, {library!r}); from pydantic_ai_gepa.cli.scoring_child import main; main()"
        profile = seatbelt_profile(private, checkout, scratch, port)
        command = sandbox_command(
            profile, [sys.executable, "-I", "-B", "-c", bootstrap]
        )
        cleanup = _process_cleanup(scratch)
        process = subprocess.Popen(
            command,
            cwd=checkout,
            env=child_environment(scratch, port),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        try:
            channel = Channel(process)
            channel.send(
                {
                    "config": asdict(config),
                    "validation": validation,
                    "components": components,
                }
            )
            if channel.receive() != {"type": "ready"}:
                raise ScoringSandboxError("Sandboxed scorer could not initialize.")
            cleanup.verify_worker(process.pid)

            async def evaluate() -> list[EvaluationRecord]:
                records = []
                for case in cases:
                    async with meter.rollout():
                        channel.send(
                            {
                                "name": case.name,
                                "inputs": case.inputs,
                                "expected_output": case.expected_output,
                                "metadata": case.metadata,
                            }
                        )
                        while True:
                            raw = channel.receive()
                            if raw.get("type") != "usage":
                                record = _result(raw, case.name, validation)
                                if raw["cached"]:
                                    meter.declare_cached_rollout()
                                if not raw["failed"]:
                                    meter.callable_completed()
                                records.append(record)
                                break
                            if set(raw) != {
                                "type",
                                "dollars",
                                "input_tokens",
                                "output_tokens",
                            }:
                                raise ScoringSandboxError(
                                    "Invalid scoring usage shape."
                                )
                            dollars = raw["dollars"]
                            if dollars is not None and _number(dollars) < 0:
                                raise ScoringSandboxError("Invalid scoring cost.")
                            input_tokens = _number(raw["input_tokens"], integer=True)
                            output_tokens = _number(raw["output_tokens"], integer=True)

                            def price(_response: Any) -> float:
                                if dollars is None:
                                    raise ValueError(
                                        "Sandboxed response price unavailable"
                                    )
                                return dollars

                            meter.price_fn = price
                            # Model names are candidate-controlled strings. Publish
                            # a fixed bucket, never a validation input hidden in one.
                            meter.record(
                                "rollout",
                                ModelResponse(
                                    parts=[],
                                    model_name="sandbox",
                                    usage=RequestUsage(
                                        input_tokens=int(input_tokens),
                                        output_tokens=int(output_tokens),
                                    ),
                                ),
                            )
                            channel.send({"type": "ack"})
                return records

            return asyncio.run(evaluate())
        finally:
            try:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    # macOS can report EPERM for a zombie-only process group.
                    if process.poll() is None:
                        process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                raise ScoringSandboxError(
                    "Cannot terminate the scoring worker; evaluation refused."
                ) from None
            finally:
                # setsid descendants escape killpg but inherit the profile.
                # Sweep before retiring private paths and the model proxy.
                try:
                    survivors = cleanup.sweep()
                finally:
                    for stream in (process.stdin, process.stdout):
                        if stream is not None:
                            stream.close()
                if survivors:
                    raise ScoringSandboxError(
                        "Scoring sandbox survivors were terminated; evaluation refused."
                    )

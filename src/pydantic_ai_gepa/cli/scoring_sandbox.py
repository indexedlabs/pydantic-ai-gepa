"""Harness-owned candidate checkouts and a fail-closed scoring subprocess.

The parent treats every child byte as untrusted. No candidate callable, pickle,
exception text, validation feedback, or candidate-selected path crosses back.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
import io
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import tarfile
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
    if (
        source != "git"
        or config.acceptance.mode != "scalar"
        or config.acceptance.pinned_scorer
    ):
        raise ScoringSandboxError(
            "Held-out sandbox scoring currently requires git candidates with scalar, "
            "unpinned acceptance. Unsupported modes are refused before importing candidate code."
        )


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
def private_checkout(project: Path, sha: str) -> Iterator[tuple[Path, Path, Path]]:
    """Read Git objects only. Never register a worktree or touch shared refs/index."""
    if not re.fullmatch(r"[0-9a-f]{40,64}", sha):
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
        # Disable replace refs and user/system config. archive's builtin tar
        # format uses no hooks, smudge filters, external archivers or checkout.
        result = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(repository),
                "archive",
                "--format=tar",
                sha,
            ],
            env={
                "PATH": "/usr/bin:/bin",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
            },
            capture_output=True,
        )
        if result.returncode:
            raise ScoringSandboxError("Cannot create private candidate checkout.")
        try:
            with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
                # Reject symlinks entirely: no candidate-controlled link can
                # redirect extraction, imports, or the scratch write allowance.
                for member in archive.getmembers():
                    target = checkout / member.name
                    if not target.resolve().is_relative_to(checkout) or not (
                        member.isfile() or member.isdir()
                    ):
                        raise ScoringSandboxError(
                            "Private candidate archives cannot contain links or special files."
                        )
                archive.extractall(checkout, filter="data")
        except (tarfile.TarError, OSError):
            raise ScoringSandboxError(
                "Cannot extract private candidate checkout."
            ) from None
        yield private, checkout / prefix, scratch


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
            remaining = deadline - time.monotonic()
            if (
                remaining <= 0
                or not select.select([self.process.stdout], [], [], remaining)[0]
            ):
                raise ScoringSandboxError("Sandboxed scorer response timed out.")
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


def score_cases(
    *,
    config: GepaConfig,
    project: Path,
    sha: str,
    cases: list[Any],
    validation: bool,
    meter: Any,
) -> list[EvaluationRecord]:
    """Serial rollouts keep admission and durable spend in the trusted parent.

    A new child is used for each eval/phase, so validation memory or scratch
    cannot be replayed in a later training report.
    """
    require_supported(config, "git")
    nomination = nominated_commit.get()
    if nomination is not None and nomination != sha:
        raise ScoringSandboxError("Candidate commit changed during scoring.")
    # Fail before opening the proxy or creating candidate storage on hosts
    # without a backend. There is deliberately no environment override.
    sandbox_command("", [])
    try:
        addresses = allowed_addresses(os.environ.get("GEPA_HARNESS_ALLOWED_HOSTS", ""))
    except ValueError:
        raise ScoringSandboxError(
            "Invalid GEPA_HARNESS_ALLOWED_HOSTS; use host:port pairs."
        ) from None
    with (
        private_checkout(project, sha) as (private, checkout, scratch),
        connect_proxy(addresses) as port,
    ):
        # -I excludes cwd, PYTHONPATH and the user site from interpreter startup.
        library = str(Path(__file__).resolve().parents[2])
        bootstrap = f"import sys; sys.path.insert(0, {library!r}); from pydantic_ai_gepa.cli.scoring_child import main; main()"
        profile = seatbelt_profile(private, checkout, scratch, port)
        command = sandbox_command(
            profile, [sys.executable, "-I", "-B", "-c", bootstrap]
        )
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
            channel.send({"config": asdict(config), "validation": validation})
            if channel.receive() != {"type": "ready"}:
                raise ScoringSandboxError("Sandboxed scorer could not initialize.")

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
            # Terminate the worker's process group, including normal evaluator
            # subprocesses, before retiring the proxy and private directories.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                # macOS may return EPERM for a process group containing only
                # zombies. Reap below without masking the scorer's real error.
                pass
            process.wait()
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()

"""Drive managed reflection steps without keeping a coding agent in the loop."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any, Iterator

import typer

from . import events, harness, harness_record, lanes, reflector
from .layout import gepa_dir, repo_root, run_dir, set_gepa_dirname
from .process_guard import DarwinProcesses, GuardError, ProcessGuard
from .run import _load_state
from .safe_git import SafeGitError, refuse_executable_lane_git
from .scoring_sandbox import PROVIDER_KEYS
from .validation import _pin_path, check_heldout_pin, harness_environment

EXIT_PAUSED = 76
EXIT_USAGE_LIMIT = 77
EXIT_SURVIVORS = 78

DEFAULT_REFLECTORS = [
    {
        "label": "codex",
        "argv": [
            "codex",
            "exec",
            "--profile",
            "gepa-reflector",
            "-C",
            "{checkout}",
            "Read {prompt} and follow its instructions. The packet is {packet}.",
        ],
        "usage_limit": {"output_regexes": [r"(?i)usage limit|usage_limit"]},
    }
]


def load_reflectors(path: Path | None) -> tuple[list[dict[str, Any]], str, list[str]]:
    try:
        entries = json.loads(path.read_text()) if path else DEFAULT_REFLECTORS
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(
            "Cannot read reflector configuration as JSON."
        ) from exc
    pass_env = []
    if isinstance(entries, dict):
        if set(entries) - {"reflectors", "pass_env"}:
            raise typer.BadParameter("Invalid reflector configuration.")
        pass_env = entries.get("pass_env", [])
        entries = entries.get("reflectors")
    validate_pass_env(pass_env)
    if not isinstance(entries, list) or not entries:
        raise typer.BadParameter("Reflectors must be a nonempty JSON array.")
    labels = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"label", "argv", "usage_limit"}:
            raise typer.BadParameter("Invalid reflector entry.")
        label, argv = entry.get("label"), entry.get("argv")
        if not isinstance(label, str) or not label or label in labels:
            raise typer.BadParameter("Reflector labels must be nonempty and unique.")
        labels.add(label)
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(a, str) and a for a in argv)
        ):
            raise typer.BadParameter(
                "Reflector argv must be a nonempty array of strings."
            )
        try:
            for arg in argv:
                arg.format(checkout="checkout", packet="packet", prompt="prompt")
            detection = entry.get("usage_limit", {})
            if not isinstance(detection, dict) or set(detection) - {
                "exit_codes",
                "output_regexes",
            }:
                raise ValueError
            codes = detection.get("exit_codes", [])
            patterns = detection.get("output_regexes", [])
            if not isinstance(codes, list) or not all(
                type(code) is int for code in codes
            ):
                raise ValueError
            if not isinstance(patterns, list) or not all(
                isinstance(p, str) for p in patterns
            ):
                raise ValueError
            for pattern in patterns:
                re.compile(pattern)
        except (KeyError, ValueError, IndexError, re.error) as exc:
            raise typer.BadParameter(
                "Invalid reflector template or usage-limit detection."
            ) from exc
    digest = hashlib.sha256(
        json.dumps(
            {"reflectors": entries, "pass_env": pass_env} if pass_env else entries,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return entries, digest, pass_env


def reserved_environment() -> set[str]:
    return (
        {"GEPA_CANDIDATE_COMPONENTS_JSON", "GEPA_TRACE_FILE"}
        | {
            name.strip()
            for name in os.environ.get("GEPA_HARNESS_PASS_ENV", "").split(",")
            if name.strip()
        }
        | {
            name
            for name in os.environ
            if name.startswith(("GEPA_HELDOUT_", "GEPA_HARNESS_"))
        }
    )


def validate_pass_env(names: Any) -> None:
    if not isinstance(names, list) or any(
        not isinstance(name, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
        or name.startswith(("GEPA_HELDOUT_", "GEPA_HARNESS_"))
        or name in reserved_environment()
        for name in names
    ):
        raise typer.BadParameter(
            "pass_env must contain explicit training-only variable names; harness variables are forbidden."
        )


def reflector_environment(pass_env: list[str] | None = None) -> dict[str, str]:
    """Do not pass harness inputs or evaluator credentials to a reflector."""
    validate_pass_env(pass_env or [])
    private = (set(PROVIDER_KEYS) - set(pass_env or [])) | reserved_environment()
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in private and not key.startswith(("GEPA_HELDOUT_", "GEPA_HARNESS_"))
    }
    try:
        count = int(environment.get("GIT_CONFIG_COUNT", "0"))
        if count < 0:
            raise ValueError
    except ValueError as exc:
        raise typer.BadParameter(
            "GIT_CONFIG_COUNT must be a nonnegative integer."
        ) from exc
    for key, value in (
        ("gc.autoDetach", "false"),
        ("maintenance.autoDetach", "false"),
        ("safe.bareRepository", "explicit"),
    ):
        environment[f"GIT_CONFIG_KEY_{count}"] = key
        environment[f"GIT_CONFIG_VALUE_{count}"] = value
        count += 1
    environment["GIT_CONFIG_COUNT"] = str(count)
    return environment


def private_state_path(run_id: str) -> Path:
    run = _load_state(run_id)
    public = run_dir(run_id) / "drive.json"
    if public.exists() or public.is_symlink():
        raise typer.BadParameter(
            "Public drive.json is untrusted; remove it explicitly before driving. It will not be loaded or migrated."
        )
    if run.heldout_required:
        dataset, _ = check_heldout_pin(repo_root(), run_id)
        pin = _pin_path(dataset, repo_root(), run_id)
        path = pin.with_name(pin.stem + ".drive.json")
    else:
        key = hashlib.sha256(
            f"{gepa_dir(repo_root()).resolve()}\0{run_id}".encode()
        ).hexdigest()
        base = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
        path = base / "pydantic-ai-gepa" / "drive" / key / "drive.json"
    roots = [gepa_dir(repo_root()), repo_root()]
    if run.project_root:
        roots.append(Path(run.project_root))
    if run.lanes:
        for lane in lanes.load_all_lane_states(repo_root(), run_id):
            roots.extend(
                Path(p) for p in (lane.worktree_path, lane.candidate_project_path) if p
            )
    if any(path.resolve().is_relative_to(root.resolve()) for root in roots):
        raise typer.BadParameter(
            "Driver-private state must be outside GEPA_DIR, project and lane worktrees."
        )
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.parent.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o777 != 0o700:
        raise typer.BadParameter(
            "Driver-private directory must be owned by the current user with mode 0700."
        )
    check_private_file(path)
    return path


def check_private_file(path: Path) -> None:
    if path.is_symlink():
        raise typer.BadParameter("Driver-private files must not be symlinks.")
    if path.exists():
        info = path.stat()
        if (
            not path.is_file()
            or info.st_uid != os.getuid()
            or info.st_mode & 0o777 != 0o600
        ):
            raise typer.BadParameter(
                "Driver-private files must be owned by the current user with mode 0600."
            )


@contextmanager
def driver_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    check_private_file(path)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise typer.BadParameter("A driver already holds this run.") from None
        yield
    finally:
        os.close(fd)


class Driver:
    def __init__(
        self,
        run_id: str,
        path: Path,
        entries: list[dict[str, Any]],
        digest: str,
        accept_change: bool,
        timeout: float,
        max_steps: int | None,
        attempts: int,
        grace: float,
        pass_env: list[str] | None = None,
    ) -> None:
        self.run_id, self.path, self.entries = run_id, path, entries
        self.timeout, self.max_steps, self.attempt_cap, self.grace = (
            timeout,
            max_steps,
            attempts,
            grace,
        )
        self.steps = 0
        self.pass_env = pass_env or []
        self.backend = DarwinProcesses()
        self.state = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "reflectors_hash": digest,
                "reflector_index": 0,
                "attempts": {},
                "usage_limits": {},
                "changes": [],
                "step": None,
                "pause_reason": None,
                "sequence": 0,
                "event": None,
                "phase": "recorded",
            }
        )
        self.guard_pending()
        reason = self.state.get("pause_reason") or ""
        if (
            reason.startswith(("scoring_", "harness_exit_", "provider_stop"))
            and self.state.get("pause_run_updated_at")
            == _load_state(self.run_id).updated_at
        ):
            self.pause(reason)
        if self.state["reflectors_hash"] != digest:
            if not accept_change:
                raise typer.BadParameter(
                    "Reflector list changed; pass --accept-reflector-change to record and accept it."
                )
            self.state["changes"].append(
                {
                    "old": self.state["reflectors_hash"],
                    "new": digest,
                    "time": time.time(),
                }
            )
            self.state.update(reflectors_hash=digest, reflector_index=0)
            if reason != "usage_limit" and self.state["phase"] in {
                "leased",
                "reflector_started",
                "reflector_exited",
                "guard_passed",
                "retry_ready",
            }:
                # The old step has already been guarded. Its nomination remains
                # authoritative, but its fallback index belongs to the old list.
                self.state.update(step=None, phase="retry_ready")
        if self.state["pause_reason"] == "usage_limit":
            run = _load_state(self.run_id)
            previous = self.state.get("step") or {}
            if run.lanes == 0 and run.reflector["epoch"] == previous.get(
                "resume_epoch"
            ):
                reflector.resume(
                    self.run_id,
                    reason="usage_limit",
                    reflector=self.entries[0]["label"],
                )
            self.state["reflector_index"] = 0
            self.state.update(step=None, phase="received")
        if self.state["pause_reason"] == "attempt_limit":
            self.state["attempts"] = {}
            self.state.update(step=None, phase="received")
        self.state["pause_reason"] = None
        self.save()

    def save(self) -> None:
        harness._atomic_json(self.path, self.state)

    def phase(self, phase: str, **kwargs: Any) -> None:
        self.state.update(phase=phase, **kwargs)
        self.save()

    def pause(self, reason: str, code: int = EXIT_PAUSED) -> None:
        self.state["pause_reason"] = reason
        self.state["pause_run_updated_at"] = _load_state(self.run_id).updated_at
        self.save()
        typer.echo(f"Drive paused: {reason}", err=True)
        raise typer.Exit(code)

    def guard_pending(self) -> None:
        step = self.state.get("step")
        if not step or not step.get("guard"):
            return
        guard = ProcessGuard(self.backend, step["guard"], self.save)
        try:
            survivors = guard.finish(self.grace)
        except GuardError:
            self.state["pause_reason"] = "process_inspection_failed"
            self.save()
            raise
        if survivors:
            for process in survivors:
                typer.echo(
                    f"Survivor pid={process.pid} command={process.command!r} ({guard.classify(process)})",
                    err=True,
                )
            self.pause("process_survivors", EXIT_SURVIVORS)

    def recorded(self) -> None:
        self.phase("recorded", step=None)
        event = self.state.get("event")
        if event:
            events.ack(self.run_id, event["id"])
        self.state["event"] = None
        self.save()

    def reset_lane(self, lane: str) -> None:
        current = lanes.load_lane_state(repo_root(), self.run_id, lane)
        if current.status in {"awaiting_selection", "paused_for_reflection"}:
            return
        # lane_reset's legacy PID termination is not an attribution mechanism.
        # Guarded descendants must be gone before that command may run.
        if current.eval_pid and lanes._pid_alive(current.eval_pid):
            self.pause("live_lane_evaluator")
        lanes.lane_reset(lane=lane, run_id=self.run_id)

    def fail_step(self, lane: str | None, reason: str) -> None:
        key = lane or "single"
        step = self.state["step"]
        count = step.setdefault(
            "failed_attempt", self.state["attempts"].get(key, 0) + 1
        )
        self.state["attempts"][key] = count
        self.state["last_failure"] = reason
        self.save()
        if lane:
            self.reset_lane(lane)
        if count >= self.attempt_cap:
            self.pause("attempt_limit")

    def signal_group(self, pgid: int, guard: ProcessGuard, sig: int) -> None:
        # A reaped leader's PID can be reused. Verify that the group still has
        # an attributed member before signaling it, using kernel creation IDs.
        for member in guard.observe():
            if guard.classify(member) != "attributed":
                continue
            try:
                if os.getpgid(member.pid) != pgid:
                    continue
                current = self.backend.identity(member.pid)
                if current is not None and current.unique_id == member.unique_id:
                    os.killpg(pgid, sig)
                    return
            except ProcessLookupError:
                continue

    def check_checkout(self, lane: str | None, checkout: Path) -> None:
        from . import lane_repositories

        def check() -> None:
            root = refuse_executable_lane_git(checkout)
            if lane and root != lane_repositories.lane_path(
                repo_root(), self.run_id, lane
            ):
                raise SafeGitError("Reflector project is outside its lane repository.")

        try:
            check()
            return
        except OSError as error:
            entry = {
                "lane": lane,
                "sequence": self.state["sequence"],
                "outcome": "refused",
                "reason": str(error),
            }
            self.state.setdefault("lane_git", []).append(entry)
            self.save()
        if lane is None:
            self.pause("unsafe_project_git")
        try:
            run = _load_state(self.run_id)
            base = run.reflection_baseline_commit_sha
            if not base:
                raise SafeGitError("Lane base commit is unavailable.")
            current = lanes.load_lane_state(repo_root(), self.run_id, lane)
            repositories = lane_repositories.load(repo_root(), self.run_id)
            repositories.create_lane(lane, base, str(current.branch), replace=True)
            check()
        except (
            OSError,
            ValueError,
            typer.BadParameter,
            subprocess.SubprocessError,
        ) as error:
            entry.update(outcome="rebuild_failed", rebuild_error=str(error))
            self.save()
            self.pause("unsafe_lane_git")
        entry["outcome"] = "rebuilt"
        self.save()

    def step(self, lane: str | None) -> None:
        if self.max_steps is not None and self.steps >= self.max_steps:
            self.pause("max_steps")
        run = _load_state(self.run_id)
        if lane:
            current = lanes.load_lane_state(repo_root(), self.run_id, lane)
            if current.status == "awaiting_selection":
                self.recorded()
                return
            if current.status != "paused_for_reflection":
                self.reset_lane(lane)
            lanes.lane_lease(lane=lane, run_id=self.run_id)
            current = lanes.load_lane_state(repo_root(), self.run_id, lane)
            packet = Path(str(current.packet_path))
            checkout = Path(
                str(current.candidate_project_path or current.worktree_path)
            )
        else:
            packet = reflector.write_packet(self.run_id)
            checkout = Path(run.project_root) if run.project_root else repo_root()
        sequence = self.state["sequence"] + 1
        logs = run_dir(self.run_id) / "drive" / f"step-{sequence:06d}"
        logs.mkdir(parents=True, exist_ok=True)
        prompt = logs / "prompt.txt"
        prompt.write_text(
            "Perform one reflection step. Read the packet at "
            + str(packet.resolve())
            + ".\n"
            "Reflect on its training reports and journal, edit only within the declared scope, "
            "and commit the candidate. Follow the packet's recovery instructions. "
            "Run its nominate/continue command exactly once, "
            + (
                "adding --foreground and waiting for training evaluation to finish. "
                if lane
                else "adding --wait-secs 0 so it only enqueues a nomination. "
            )
            + "Then exit. Do not start a harness or run selection. Do not leave background processes.\n"
        )
        entry = self.entries[self.state["reflector_index"]]
        step = {
            "lane": lane,
            "logs": str(logs),
            "reflector": entry["label"],
            "guard": {},
        }
        self.phase("leased", step=step, sequence=sequence)
        guard = ProcessGuard(self.backend, step["guard"], self.save)
        argv = [
            arg.format(
                checkout=str(checkout.resolve()),
                packet=str(packet.resolve()),
                prompt=str(prompt.resolve()),
            )
            for arg in entry["argv"]
        ]
        timed_out = False
        with (
            (logs / "stdout.log").open("wb") as stdout,
            (logs / "stderr.log").open("wb") as stderr,
        ):
            self.check_checkout(lane, checkout)
            guard.begin()
            process = subprocess.Popen(
                argv,
                cwd=checkout,
                env=reflector_environment(self.pass_env),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            self.steps += 1
            step["pid"] = process.pid
            self.phase("reflector_started")
            identity = self.backend.identity(process.pid, include_zombie=True)
            if identity:
                step["guard"]["root"] = identity.unique_id
                step["guard"]["seen"][str(identity.unique_id)] = asdict(identity)
                step["guard"]["root_responsible"] = guard.responsible_pid(process.pid)
                step["unique_id"] = identity.unique_id
                step["start"] = identity.start
                self.save()
            deadline = time.monotonic() + self.timeout
            try:
                while process.poll() is None:
                    guard.observe()
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                    time.sleep(0.05)
            finally:
                # The child's session is separate from the driver. Do not wait
                # for output pipes: every descendant inherits regular log files.
                for sig, delay in ((signal.SIGTERM, 0.2), (signal.SIGKILL, 0)):
                    process.poll()
                    try:
                        if process.returncode is None:
                            # An unreaped leader reserves this PID even if it
                            # exited before publishing its kernel identity.
                            os.killpg(process.pid, sig)
                        else:
                            self.signal_group(process.pid, guard, sig)
                    except ProcessLookupError:
                        pass
                    except PermissionError as exc:
                        raise GuardError(
                            "Cannot signal the reflector process group; scoring refused."
                        ) from exc
                    if delay:
                        time.sleep(delay)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired as exc:
                    raise GuardError(
                        "Reflector did not exit after group cleanup."
                    ) from exc
                guard.close_window()
        output = "\n".join(
            (logs / name).read_text(errors="replace")
            for name in ("stdout.log", "stderr.log")
        )
        detection = entry.get("usage_limit", {})
        limited = process.returncode in detection.get("exit_codes", []) or any(
            re.search(pattern, output)
            for pattern in detection.get("output_regexes", [])
        )
        step.update(
            exit_code=process.returncode, usage_limit=limited, timeout=timed_out
        )
        self.phase("reflector_exited")
        self.resolve_step()

    def resolve_step(self) -> None:
        step = self.state["step"]
        lane = step["lane"]
        self.guard_pending()
        self.phase("guard_passed")
        if lane:
            nominated = (
                lanes.load_lane_state(repo_root(), self.run_id, lane).status
                == "awaiting_selection"
            )
        else:
            nominated = bool(self.pending())
        if not nominated and step["usage_limit"]:
            self.state["usage_limits"][str(self.state["sequence"])] = {
                "label": step["reflector"],
                "exit_code": step["exit_code"],
            }
            if "fallback_index" not in step:
                step["fallback_index"] = self.state["reflector_index"] + 1
                step["resume_epoch"] = _load_state(self.run_id).reflector["epoch"]
            self.state["reflector_index"] = step["fallback_index"]
            self.save()
            if lane:
                self.reset_lane(lane)
            if self.state["reflector_index"] >= len(self.entries):
                self.pause("usage_limit", EXIT_USAGE_LIMIT)
            if (
                not lane
                and _load_state(self.run_id).reflector["epoch"] == step["resume_epoch"]
            ):
                reflector.resume(
                    self.run_id,
                    reason="usage_limit",
                    reflector=self.entries[self.state["reflector_index"]]["label"],
                )
            self.phase("retry_ready", step=None)
            return
        if not nominated:
            self.fail_step(
                lane,
                "timeout"
                if step["timeout"]
                else f"reflector_exit_{step['exit_code']}_without_nomination",
            )
            self.phase("retry_ready", step=None)
            return
        self.state["attempts"][lane or "single"] = 0
        self.save()
        if lane:
            self.recorded()

    def pending(self) -> list[tuple[Path, dict[str, Any]]]:
        return [
            (path, request)
            for path, request in harness._requests(self.run_id)
            if not harness_record.exists(harness._result_path(path))
        ]

    def score(self, lane_run: bool) -> None:
        self.guard_pending()
        self.phase("scoring")
        try:
            if lane_run:
                from .select import run_select

                with harness_environment():
                    run_select(self.run_id)
            else:
                harness.serve(run_id=self.run_id, once=True)
        except typer.Exit as exc:
            if _load_state(self.run_id).status != "done":
                self.pause(f"scoring_exit_{exc.exit_code}")
        except Exception as exc:
            from ..provider_errors import is_provider_stop_error

            self.pause(
                "provider_stop"
                if is_provider_stop_error(exc)
                else f"scoring_error_{type(exc).__name__}"
            )
        run = _load_state(self.run_id)
        if run.status == "paused_after_infrastructure_error":
            self.pause("paused_after_infrastructure_error")
        if not lane_run:
            for path, request in harness._requests(self.run_id):
                if request.get("reflector_epoch") != run.reflector["epoch"]:
                    continue
                result = harness._result_path(path)
                if harness_record.exists(result):
                    payload = json.loads(harness_record.read_text(result))
                    if payload.get("state_updated_at") == run.updated_at and payload[
                        "exit_code"
                    ] not in (0, 70):
                        self.pause(f"harness_exit_{payload['exit_code']}")
        self.phase("scoring_command_run")
        self.recorded()

    def run(self) -> None:
        if self.state["phase"] == "recorded":
            self.recorded()  # a crash after recording but before ack
        elif self.state["phase"] in {"leased", "reflector_started"}:
            lane = (self.state.get("step") or {}).get("lane")
            if lane:
                self.reset_lane(lane)
        elif self.state["phase"] in {"reflector_exited", "guard_passed"}:
            self.resolve_step()
        while True:
            run = _load_state(self.run_id)
            if run.status == "done":
                from .run import _write_final_report

                report, _ = _write_final_report(
                    run,
                    overshoot=max(0, run.iterations - run.max_iterations)
                    if run.lanes
                    else None,
                )
                self.recorded()
                typer.echo(str(report))
                return
            if run.status == "paused_after_infrastructure_error":
                self.pause("paused_after_infrastructure_error")
            if run.lanes == 0:
                if self.pending():
                    self.score(False)
                else:
                    self.step(None)
                continue
            event = self.state.get("event")
            if event is None:
                events.run_reaper_pass(self.run_id)
                next_event = events.next_event(self.run_id)
                if next_event is None:
                    time.sleep(0.05)
                    continue
                event = next_event.to_dict()
                self.phase("received", event=event)
            if event["type"] in {"lane_ready", "lane_stalled"}:
                self.step(event["lane"])
            elif event["type"] == "run_done":
                self.recorded()
                typer.echo(str(event["payload"]["final_report_path"]))
                return
            elif event["type"] == "selection_due":
                # A completed select advances the lane iteration. Its event can
                # remain unacked if the driver dies after the final checkpoint.
                if run.select_phase is None and all(
                    lane.status == "paused_for_reflection"
                    for lane in lanes.load_all_lane_states(repo_root(), self.run_id)
                ):
                    self.recorded()
                else:
                    self.score(True)
            else:
                self.recorded()


def drive(
    run_id: str = typer.Option(..., "--run-id"),
    reflectors: Path | None = typer.Option(
        None, "--reflectors", exists=True, dir_okay=False
    ),
    step_timeout: float = typer.Option(900, "--step-timeout", min=0.01),
    max_steps: int | None = typer.Option(None, "--max-steps", min=1),
    max_attempts: int = typer.Option(2, "--max-attempts", min=1),
    survivor_grace: float = typer.Option(5, "--survivor-grace", min=0.01),
    accept_reflector_change: bool = typer.Option(False, "--accept-reflector-change"),
) -> None:
    """Drive an already-started run; this process owns all scoring timing."""
    if not all(math.isfinite(value) for value in (step_timeout, survivor_grace)):
        raise typer.BadParameter("Step timeout and survivor grace must be finite.")
    run = _load_state(run_id)
    set_gepa_dirname(str(gepa_dir(repo_root()).resolve()))
    if run.lanes == 0 and not run.heldout_required:
        raise typer.BadParameter(
            "Drive supports lane runs and single-checkout held-out runs."
        )
    path = private_state_path(run_id)
    entries, digest, pass_env = load_reflectors(reflectors)
    with driver_lock(path.with_suffix(".lock")):
        driver = None
        try:
            driver = Driver(
                run_id,
                path,
                entries,
                digest,
                accept_reflector_change,
                step_timeout,
                max_steps,
                max_attempts,
                survivor_grace,
                pass_env,
            )
            driver.run()
        except GuardError as exc:
            if driver is not None:
                driver.state["pause_reason"] = "process_inspection_failed"
                driver.save()
            typer.echo(f"Drive process guard refused: {exc}", err=True)
            raise typer.Exit(EXIT_SURVIVORS) from exc

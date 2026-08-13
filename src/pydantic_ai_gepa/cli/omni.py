"""Durable outer Omni meta-run controller.

The controller deliberately records work instead of starting processes.  A
Codex (or another) orchestrator consumes the packets, creates isolated child
workspaces, and submits immutable receipts.  Consequently all decisions,
budgets, and retries survive a process restart without asking an orchestrator
to reconstruct phase state from reflection text or traces.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, Literal

import fcntl
import typer
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from ..acceptance import compare_candidate_samples
from .layout import gepa_dir, repo_root

app = typer.Typer(no_args_is_help=True, add_completion=False)
VERSION = 2
_SAFE_IDENTIFIER = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
_OMNI_ID_PATTERN = re.compile(r"omni-[a-z0-9]{12}")
_transition_context: ContextVar[tuple[Path, dict[str, Any]] | None] = ContextVar(
    "omni_transition_context", default=None
)


def _root() -> Path:
    return gepa_dir(repo_root()) / "omni" / "runs"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    """Return the standard raw-byte SHA-256 advertised for an artifact file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _safe_path(value: str) -> str:
    path = Path(value).resolve()
    base = repo_root().resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError("Artifact paths must stay within the repository.") from exc
    return str(path)


def _verify_artifact(value: str, expected_sha256: str) -> str:
    if not _is_digest(expected_sha256):
        raise ValueError("sha256 must be a lower-case 64-character digest.")
    path = Path(_safe_path(value))
    if not path.is_file():
        raise ValueError(f"Frozen artifact does not exist: {path}.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"Frozen artifact digest mismatch: {path}.")
    return str(path)


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _load(omni_id: str) -> tuple[Path, dict[str, Any]]:
    if not _OMNI_ID_PATTERN.fullmatch(omni_id):
        raise typer.BadParameter("Invalid Omni run ID.")
    root = _root().resolve()
    directory = (root / omni_id).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise typer.BadParameter("Invalid Omni run ID.") from exc
    path = directory / "state.json"
    if not path.exists():
        raise typer.BadParameter(f"Unknown Omni run {omni_id!r}.")
    return directory, json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _locked(omni_id: str) -> Any:
    """Serialize every state transition across independently launched CLIs."""
    directory, _ = _load(omni_id)
    fd = os.open(directory / ".run.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield directory, json.loads((directory / "state.json").read_text())
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _serialized_transition(function: Any) -> Any:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        omni_id = kwargs.get("omni_id")
        if omni_id is None and args:
            omni_id = args[0]
        if not isinstance(omni_id, str):
            raise typer.BadParameter("omni_id is required.")
        with _locked(omni_id) as (directory, state):
            token = _transition_context.set((directory, state))
            try:
                return function(*args, **kwargs)
            finally:
                _transition_context.reset(token)

    return wrapped


def _transition() -> tuple[Path, dict[str, Any]]:
    value = _transition_context.get()
    if value is None:
        raise RuntimeError("An Omni mutation requires the run transition lock.")
    return value


def _save(directory: Path, state: dict[str, Any]) -> None:
    state["revision"] = int(state.get("revision", 0)) + 1
    _atomic(directory / "state.json", state)


def _event_key(type_: str, payload: dict[str, Any]) -> str:
    return _digest({"type": type_, "payload": payload})


_EVENT_REQUIRED_KEYS: dict[str, frozenset[str]] = {
    "child_ready": frozenset(
        {
            "packet_path",
            "packet_sha256",
            "phase",
            "child_id",
            "engine",
            "ordinal",
            "reserved_metric_calls",
            "workspace",
        }
    ),
    "phase2_ready": frozenset(
        {
            "packet_path",
            "packet_sha256",
            "phase",
            "child_id",
            "engine",
            "reserved_metric_calls",
            "workspace",
            "seed_artifact_sha256",
        }
    ),
    "fair_compare_ready": frozenset(
        {"phase", "packet_path", "packet_sha256", "reserved_metric_calls"}
    ),
    "reporting_ready": frozenset(
        {"packet_path", "packet_sha256", "reserved_metric_calls"}
    ),
    "omni_done": frozenset(
        {"final_candidate", "artifact_path", "artifact_sha256", "state_path"}
    ),
    "error_escalation": frozenset({"error_path", "claim_path", "claim_sha256"}),
}


def _validate_event_payload(type_: str, payload: dict[str, Any]) -> None:
    required = _EVENT_REQUIRED_KEYS.get(type_)
    if required is None or set(payload) != required:
        raise ValueError(f"Invalid fixed outer event payload for {type_!r}.")
    if any(
        isinstance(value, (dict, list)) or not isinstance(value, (str, int))
        for value in payload.values()
    ):
        raise ValueError(
            "Outer event payloads contain paths and scalar routing metadata only."
        )


def _write_event(directory: Path, event: dict[str, Any]) -> None:
    """Write an already-durable outbox entry exactly once.

    ``state.json`` contains the full outbox before this is called.  A crash in
    between is repaired by ``next`` or the next transition flushing the same
    exact bytes, so observers never see work whose state was not committed.
    """
    path = directory / "events" / f"{event['sequence']:020d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(event, sort_keys=True, separators=(",", ":"))
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        # State/outbox is the authority. A crash can leave a partial event
        # projection behind after state committed, so atomically repair that
        # projection rather than treating the recoverable partial bytes as a
        # collision. Exact existing bytes remain a no-op.
        if path.read_text(encoding="utf-8") == serialized:
            return
        _atomic_text(path, serialized)
        return
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(serialized)


def _flush_outbox(directory: Path, state: dict[str, Any]) -> None:
    for event in state.get("outbox", []):
        _write_event(directory, event)


def _queue_event(
    directory: Path, state: dict[str, Any], type_: str, payload: dict[str, Any]
) -> str:
    _validate_event_payload(type_, payload)
    key = _event_key(type_, payload)
    known = state.setdefault("event_keys", {}).get(key)
    if known:
        _flush_outbox(directory, state)
        return str(known)
    sequence = int(state.get("event_sequence", 0)) + 1
    event_id = f"{sequence:020d}-{key[:16]}"
    event = {
        "id": event_id,
        "key": key,
        "sequence": sequence,
        "type": type_,
        "payload": payload,
    }
    state["event_sequence"] = sequence
    state.setdefault("event_keys", {})[key] = event_id
    state.setdefault("outbox", []).append(event)
    _save(directory, state)
    _flush_outbox(directory, state)
    return event_id


def _append_event_in_memory(
    state: dict[str, Any], type_: str, payload: dict[str, Any]
) -> None:
    """Build the initial outbox before its first all-or-nothing state commit."""
    _validate_event_payload(type_, payload)
    key = _event_key(type_, payload)
    sequence = int(state.get("event_sequence", 0)) + 1
    event_id = f"{sequence:020d}-{key[:16]}"
    state["event_sequence"] = sequence
    state.setdefault("event_keys", {})[key] = event_id
    state.setdefault("outbox", []).append(
        {
            "id": event_id,
            "key": key,
            "sequence": sequence,
            "type": type_,
            "payload": payload,
        }
    )


def _events(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((directory / "events").glob("*.json"))
    ]


def _pending(directory: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    _flush_outbox(directory, state)
    cursor = directory / "cursor.json"
    acknowledged = (
        int(json.loads(cursor.read_text(encoding="utf-8")).get("sequence", 0))
        if cursor.exists()
        else 0
    )
    return [
        event for event in _events(directory) if int(event["sequence"]) > acknowledged
    ]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ArtifactRef(_StrictModel):
    artifact_path: StrictStr
    sha256: StrictStr

    @field_validator("sha256")
    @classmethod
    def _digest_is_valid(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("sha256 must be a lower-case 64-character digest.")
        return value

    @model_validator(mode="after")
    def _frozen_bytes_match(self) -> "ArtifactRef":
        return self.model_copy(
            update={"artifact_path": _verify_artifact(self.artifact_path, self.sha256)}
        )


class EvidenceRef(ArtifactRef):
    pass


class MinibatchRef(ArtifactRef):
    case_count: StrictInt = Field(ge=1)
    case_ids: list[StrictStr] = Field(min_length=1)

    @model_validator(mode="after")
    def _case_ids_are_exact(self) -> "MinibatchRef":
        if len(self.case_ids) != self.case_count or len(set(self.case_ids)) != len(
            self.case_ids
        ):
            raise ValueError("minibatch case_ids must be unique and match case_count.")
        return self


class ChildDefinition(_StrictModel):
    child_id: StrictStr
    engine: StrictStr
    metric_calls: StrictInt = Field(ge=1)
    workspace: StrictStr
    driver_manifest: ArtifactRef

    @field_validator("child_id", "engine")
    @classmethod
    def _identifier_is_safe(cls, value: str) -> str:
        if not value or any(character not in _SAFE_IDENTIFIER for character in value):
            raise ValueError(
                "Identifiers may contain only letters, digits, '.', '_' and '-'."
            )
        return value

    @field_validator("workspace")
    @classmethod
    def _workspace_is_isolated(cls, value: str) -> str:
        path = Path(_safe_path(value))
        if path == repo_root().resolve() or (path.exists() and not path.is_dir()):
            raise ValueError(
                "Each child workspace must be a safe non-root directory path."
            )
        return str(path)


class ComparisonPlan(_StrictModel):
    repetitions: StrictInt = Field(ge=1, le=5)
    max_repetitions: StrictInt | None = Field(default=None, ge=1, le=5)
    metric_calls: StrictInt = Field(ge=1)
    phase_two_metric_calls: StrictInt | None = Field(default=None, ge=1)
    mode: Literal["instance", "objective", "hybrid", "cartesian"] = "instance"
    acceptance_confidence: StrictFloat = Field(default=0.9, gt=0.0, lt=1.0)
    acceptance_min_delta: StrictFloat = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _acceptance_bounds(self) -> "ComparisonPlan":
        if self.max_repetitions is not None and self.max_repetitions < self.repetitions:
            raise ValueError("max_repetitions must be >= repetitions.")
        if not math.isfinite(self.acceptance_min_delta):
            raise ValueError("acceptance_min_delta must be finite.")
        return self

    @property
    def maximum_repetitions(self) -> int:
        return self.max_repetitions or self.repetitions


class ReportingPlan(_StrictModel):
    test_set: MinibatchRef
    metric_calls: StrictInt = Field(ge=1)


class OmniPlan(_StrictModel):
    seed: ArtifactRef
    evaluator_identity: StrictStr
    evaluator_sha256: StrictStr
    minibatch: MinibatchRef
    phase_one: list[ChildDefinition] = Field(min_length=1)
    comparison: ComparisonPlan
    phase_two: ChildDefinition
    reporting: ReportingPlan | None = None

    @field_validator("evaluator_identity")
    @classmethod
    def _evaluator_identity_is_nonempty(cls, value: str) -> str:
        if not value:
            raise ValueError("evaluator_identity cannot be empty.")
        return value

    @field_validator("evaluator_sha256")
    @classmethod
    def _evaluator_digest_is_valid(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("evaluator_sha256 must be a lower-case SHA-256 digest.")
        return value

    @model_validator(mode="after")
    def _budgeted_distinct_children(self) -> "OmniPlan":
        ids = [child.child_id for child in self.phase_one]
        engines = [child.engine for child in self.phase_one]
        workspaces = [child.workspace for child in self.phase_one]
        if (
            len(set(ids)) != len(ids)
            or len(set(engines)) != len(engines)
            or len(set(workspaces)) != len(workspaces)
        ):
            raise ValueError(
                "phase_one child IDs, engine families, and workspaces must be distinct."
            )
        all_workspaces = [*workspaces, self.phase_two.workspace]
        if any(
            Path(left) != Path(right)
            and (Path(left) in Path(right).parents or Path(right) in Path(left).parents)
            for position, left in enumerate(all_workspaces)
            for right in all_workspaces[position + 1 :]
        ):
            raise ValueError("Child workspaces cannot be ancestor/descendant paths.")
        if self.phase_two.child_id in ids or self.phase_two.workspace in workspaces:
            raise ValueError("phase_two needs a fresh child ID and isolated workspace.")
        if len({child.metric_calls for child in self.phase_one}) != 1:
            raise ValueError(
                "phase_one children require equal matched metric-call slices."
            )
        phase_one_minimum = (
            (len(self.phase_one) + 1)
            * self.comparison.maximum_repetitions
            * self.minibatch.case_count
        )
        phase_two_minimum = (
            2 * self.comparison.maximum_repetitions * self.minibatch.case_count
        )
        if self.comparison.metric_calls < phase_one_minimum:
            raise ValueError(
                "comparison metric_calls cannot fund all phase-one matched samples."
            )
        if (
            self.comparison.phase_two_metric_calls or self.comparison.metric_calls
        ) < phase_two_minimum:
            raise ValueError(
                "phase-two comparison budget cannot fund both matched candidates."
            )
        if (
            self.reporting
            and self.reporting.metric_calls < self.reporting.test_set.case_count
        ):
            raise ValueError("reporting metric_calls cannot fund the frozen test set.")
        return self


class ChildReceipt(_StrictModel):
    phase: Literal["phase_one", "phase_two"]
    child_id: StrictStr
    engine: StrictStr
    plan_sha256: StrictStr
    seed_sha256: StrictStr
    packet_sha256: StrictStr
    candidate_artifact_path: StrictStr
    candidate_artifact_sha256: StrictStr
    metric_calls: StrictInt = Field(ge=0)
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @field_validator(
        "plan_sha256", "seed_sha256", "packet_sha256", "candidate_artifact_sha256"
    )
    @classmethod
    def _receipt_digest_is_valid(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("Receipt SHA-256 values must be lower-case digests.")
        return value


class SampleReceipt(_StrictModel):
    score: Annotated[StrictFloat | StrictInt, Field()]
    selectable: StrictBool = True
    infrastructure_valid: StrictBool = True
    case_scores: dict[StrictStr, Annotated[StrictFloat | StrictInt, Field()]]
    objective_scores: dict[StrictStr, Annotated[StrictFloat | StrictInt, Field()]] = (
        Field(default_factory=dict)
    )
    per_case_objective_scores: dict[
        StrictStr, dict[StrictStr, Annotated[StrictFloat | StrictInt, Field()]]
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _finite_scores(self) -> "SampleReceipt":
        values: list[float | int] = [
            self.score,
            *self.case_scores.values(),
            *self.objective_scores.values(),
        ]
        values.extend(
            value
            for row in self.per_case_objective_scores.values()
            for value in row.values()
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Scores must be finite numeric values.")
        if self.case_scores:
            mean = sum(float(value) for value in self.case_scores.values()) / len(
                self.case_scores
            )
            if not math.isclose(float(self.score), mean, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("sample score must equal the mean of case_scores.")
        if self.per_case_objective_scores:
            objective_names = set(self.objective_scores)
            if not objective_names or any(
                set(values) != objective_names
                for values in self.per_case_objective_scores.values()
            ):
                raise ValueError(
                    "per-case objective scores require the same aggregate objective keys."
                )
            for objective in objective_names:
                mean = sum(
                    float(values[objective])
                    for values in self.per_case_objective_scores.values()
                ) / len(self.per_case_objective_scores)
                if not math.isclose(
                    float(self.objective_scores[objective]),
                    mean,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "aggregate objective scores must equal their per-case means."
                    )
        return self


class ComparisonCandidate(_StrictModel):
    candidate_id: StrictStr
    artifact_path: StrictStr
    artifact_sha256: StrictStr
    samples: list[SampleReceipt]

    @field_validator("artifact_sha256")
    @classmethod
    def _candidate_digest_is_valid(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("candidate artifact SHA-256 must be a lower-case digest.")
        return value


class ComparisonReceipt(_StrictModel):
    phase: Literal["phase_one", "phase_two"]
    plan_sha256: StrictStr
    evaluator_identity: StrictStr
    evaluator_sha256: StrictStr
    minibatch_sha256: StrictStr
    packet_sha256: StrictStr
    metric_calls: StrictInt = Field(ge=0)
    candidates: list[ComparisonCandidate]
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @field_validator(
        "plan_sha256", "evaluator_sha256", "minibatch_sha256", "packet_sha256"
    )
    @classmethod
    def _comparison_digest_is_valid(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("Comparison SHA-256 values must be lower-case digests.")
        return value


class DispatchReceipt(_StrictModel):
    phase: Literal["phase_one", "phase_two"]
    child_id: StrictStr
    packet_sha256: StrictStr
    pid: StrictInt | None = Field(default=None, ge=1)
    dispatcher: StrictStr | None = None


class ReportingReceipt(_StrictModel):
    plan_sha256: StrictStr
    evaluator_identity: StrictStr
    evaluator_sha256: StrictStr
    test_set_sha256: StrictStr
    packet_sha256: StrictStr
    candidate_artifact_path: StrictStr
    candidate_artifact_sha256: StrictStr
    metric_calls: StrictInt = Field(ge=0)
    score: Annotated[StrictFloat | StrictInt, Field()]
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @field_validator(
        "plan_sha256",
        "evaluator_sha256",
        "test_set_sha256",
        "packet_sha256",
        "candidate_artifact_sha256",
    )
    @classmethod
    def _reporting_digest_is_valid(cls, value: str) -> str:
        if not _is_digest(value):
            raise ValueError("Reporting SHA-256 values must be lower-case digests.")
        return value

    @model_validator(mode="after")
    def _finite_score(self) -> "ReportingReceipt":
        if not math.isfinite(float(self.score)):
            raise ValueError("reporting score must be finite.")
        return self


class EscalationClaim(_StrictModel):
    kind: Literal["unattainable_evaluator_target"]
    source_receipt_sha256: StrictStr
    claim: StrictStr
    requested_action: StrictStr
    case_ids: list[StrictStr] = Field(min_length=1)
    evidence: list[EvidenceRef] = Field(min_length=1)


def _plan(directory: Path, state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the canonical plan and verify its frozen content digest."""
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    declared = plan.get("plan_sha256")
    actual = _digest(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if declared != actual or (state is not None and state.get("plan_sha256") != actual):
        raise RuntimeError("Frozen Omni plan digest mismatch.")
    return plan


def _receipt(directory: Path, payload: dict[str, Any]) -> str:
    digest = _digest(payload)
    path = directory / "receipts" / f"{digest}.json"
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
        # Publishing a complete temporary file through a hard link preserves
        # the immutable no-overwrite contract. Unlike a direct O_EXCL write,
        # it can never strand a partial final receipt after interruption.
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = path.read_text(encoding="utf-8")
            if existing != serialized:
                try:
                    parsed = json.loads(existing)
                except json.JSONDecodeError:
                    # Old/direct publication could have been interrupted
                    # before this atomic scheme existed. A malformed file
                    # cannot be an immutable receipt, so repair it from the
                    # complete temporary publication.
                    os.replace(temporary, path)
                    temporary = ""
                else:
                    if _digest(parsed) != digest:
                        raise RuntimeError("Receipt digest collision.")
        return digest
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _receipt_data(directory: Path, digest: str) -> dict[str, Any]:
    if not _is_digest(digest):
        raise RuntimeError("Immutable receipt key must be a SHA-256 digest.")
    payload = json.loads(
        (directory / "receipts" / f"{digest}.json").read_text(encoding="utf-8")
    )
    if _digest(payload) != digest:
        raise RuntimeError("Immutable receipt digest mismatch.")
    return payload


def _write_packet(
    directory: Path, state: dict[str, Any], filename: str, packet: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    path = directory / "packets" / filename
    _atomic(path, packet)
    state.setdefault("packet_digests", {})[filename] = _file_sha256(path)
    return path, packet


def _expected_packet_digest(
    directory: Path, state: dict[str, Any], filename: str
) -> str:
    expected = state.get("packet_digests", {}).get(filename)
    path = directory / "packets" / filename
    if not isinstance(expected, str) or not path.exists():
        raise typer.BadParameter("Missing durable packet binding.")
    actual = _file_sha256(path)
    if actual != expected:
        raise typer.BadParameter("Packet bytes changed after durable emission.")
    return expected


def _packet(
    directory: Path,
    phase: str,
    child: dict[str, Any],
    seed: dict[str, Any],
    state: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    packet = {
        "version": VERSION,
        "omni_id": state["omni_id"],
        "plan_sha256": state["plan_sha256"],
        "phase": phase,
        "child_id": child["child_id"],
        "engine": child["engine"],
        "workspace": child["workspace"],
        # Opaque engine-specific launch instructions/configuration. The
        # controller only freezes bytes and never executes this artifact.
        "driver_manifest": child["driver_manifest"],
        "seed": seed,
        "minibatch": _plan(directory, state)["minibatch"],
        "evaluator_identity": _plan(directory, state)["evaluator_identity"],
        "evaluator_sha256": _plan(directory, state)["evaluator_sha256"],
        "reserved_metric_calls": child["metric_calls"],
    }
    filename = f"{phase}-{child['child_id']}.json"
    return _write_packet(directory, state, filename, packet)


def _comparison_packet(
    directory: Path, state: dict[str, Any], phase: str
) -> tuple[Path, dict[str, Any]]:
    plan = _plan(directory, state)
    expected = _expected_candidates(directory, state, phase)
    budget = (
        plan["comparison"]["metric_calls"]
        if phase == "phase_one"
        else plan["comparison"].get("phase_two_metric_calls")
        or plan["comparison"]["metric_calls"]
    )
    packet = {
        "version": VERSION,
        "omni_id": state["omni_id"],
        "plan_sha256": state["plan_sha256"],
        "phase": phase,
        "evaluator_identity": plan["evaluator_identity"],
        "evaluator_sha256": plan["evaluator_sha256"],
        "minibatch": plan["minibatch"],
        "reserved_metric_calls": budget,
        "repetitions": plan["comparison"]["repetitions"],
        "max_repetitions": plan["comparison"].get("max_repetitions")
        or plan["comparison"]["repetitions"],
        "mode": plan["comparison"]["mode"],
        "acceptance_confidence": plan["comparison"]["acceptance_confidence"],
        "acceptance_min_delta": plan["comparison"]["acceptance_min_delta"],
        "candidates": [
            {"candidate_id": key, **value} for key, value in expected.items()
        ],
    }
    return _write_packet(directory, state, f"compare-{phase}.json", packet)


def _reporting_packet(
    directory: Path, state: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    plan = _plan(directory, state)
    packet = {
        "version": VERSION,
        "omni_id": state["omni_id"],
        "plan_sha256": state["plan_sha256"],
        "phase": "reporting",
        "candidate": {
            "artifact_path": state["final"]["artifact_path"],
            "artifact_sha256": state["final"]["artifact_sha256"],
        },
        "test_set": plan["reporting"]["test_set"],
        "evaluator_identity": plan["evaluator_identity"],
        "evaluator_sha256": plan["evaluator_sha256"],
        "reserved_metric_calls": plan["reporting"]["metric_calls"],
    }
    return _write_packet(directory, state, "reporting.json", packet)


def _expected_child(plan: dict[str, Any], phase: str, child_id: str) -> dict[str, Any]:
    children = plan["phase_one"] if phase == "phase_one" else [plan["phase_two"]]
    for child in children:
        if child["child_id"] == child_id:
            return child
    raise typer.BadParameter(f"Unknown {phase} child {child_id!r}.")


def _verify_child_frozen_artifacts(
    plan: dict[str, Any], child: dict[str, Any], seed: dict[str, Any]
) -> None:
    """Re-check every provider-visible frozen input at a child boundary."""
    seed_digest = seed.get("sha256", seed.get("artifact_sha256"))
    if not isinstance(seed_digest, str):
        raise ValueError("Packet seed is missing its artifact digest.")
    _verify_artifact(seed["artifact_path"], seed_digest)
    _verify_artifact(plan["minibatch"]["artifact_path"], plan["minibatch"]["sha256"])
    _verify_artifact(
        child["driver_manifest"]["artifact_path"],
        child["driver_manifest"]["sha256"],
    )


def _expected_candidates(
    directory: Path, state: dict[str, Any], phase: str
) -> dict[str, dict[str, str]]:
    plan = _plan(directory, state)
    if phase == "phase_one":
        expected = {
            "seed": {
                "artifact_path": plan["seed"]["artifact_path"],
                "artifact_sha256": plan["seed"]["sha256"],
            }
        }
        for child in plan["phase_one"]:
            digest = state["children"].get(child["child_id"])
            if not digest:
                raise typer.BadParameter(
                    "All phase-one child receipts are required before comparison."
                )
            receipt = _receipt_data(directory, digest)
            expected[child["child_id"]] = {
                "artifact_path": receipt["candidate_artifact_path"],
                "artifact_sha256": receipt["candidate_artifact_sha256"],
            }
        return expected
    if state.get("winner") is None:
        raise typer.BadParameter("Phase-two comparison has no durable incumbent.")
    continuation_digest = state["children"].get(plan["phase_two"]["child_id"])
    if not continuation_digest:
        raise typer.BadParameter(
            "Phase-two child receipt is required before comparison."
        )
    continuation = _receipt_data(directory, continuation_digest)
    return {
        "incumbent": {
            "artifact_path": state["winner"]["artifact_path"],
            "artifact_sha256": state["winner"]["artifact_sha256"],
        },
        "continuation": {
            "artifact_path": continuation["candidate_artifact_path"],
            "artifact_sha256": continuation["candidate_artifact_sha256"],
        },
    }


def _normalize_evidence(payload: dict[str, Any]) -> None:
    payload["evidence"] = [
        evidence
        | {
            "artifact_path": _verify_artifact(
                str(evidence["artifact_path"]), str(evidence["sha256"])
            )
        }
        for evidence in payload.get("evidence", [])
    ]


def _normalize_comparison_paths(payload: dict[str, Any]) -> None:
    for candidate in payload["candidates"]:
        candidate["artifact_path"] = _verify_artifact(
            candidate["artifact_path"], candidate["artifact_sha256"]
        )
    _normalize_evidence(payload)


def _sample_coordinates(
    sample: SampleReceipt, mode: str, case_ids: set[str]
) -> dict[str, float]:
    if set(sample.case_scores) != case_ids:
        raise typer.BadParameter(
            "Comparison case_scores must cover exactly the frozen minibatch case IDs."
        )
    if mode == "instance":
        return {
            f"case:{key}": float(value) for key, value in sample.case_scores.items()
        }
    if not sample.objective_scores:
        raise typer.BadParameter(
            "Objective frontier modes require aggregate objective_scores."
        )
    if mode == "objective":
        return {
            f"objective:{key}": float(value)
            for key, value in sample.objective_scores.items()
        }
    if mode == "hybrid":
        return {
            **{
                f"case:{key}": float(value) for key, value in sample.case_scores.items()
            },
            **{
                f"objective:{key}": float(value)
                for key, value in sample.objective_scores.items()
            },
        }
    if set(sample.per_case_objective_scores) != case_ids:
        raise typer.BadParameter(
            "Cartesian frontier requires objectives for every frozen case."
        )
    objective_keys = set(sample.objective_scores)
    if not objective_keys or any(
        set(row) != objective_keys for row in sample.per_case_objective_scores.values()
    ):
        raise typer.BadParameter(
            "Cartesian frontier requires an identical objective set for every case."
        )
    return {
        f"case:{case_id}:objective:{objective}": float(value)
        for case_id, row in sample.per_case_objective_scores.items()
        for objective, value in row.items()
    }


def _select_comparison(
    receipt: ComparisonReceipt,
    plan: dict[str, Any],
    expected: dict[str, dict[str, str]],
) -> tuple[str, dict[str, float], dict[str, Any]]:
    if len(receipt.candidates) != len(expected) or {
        candidate.candidate_id for candidate in receipt.candidates
    } != set(expected):
        raise typer.BadParameter(
            "Comparison receipt candidates must be exactly the durable expected candidate set."
        )
    if len({candidate.candidate_id for candidate in receipt.candidates}) != len(
        receipt.candidates
    ):
        raise typer.BadParameter("Comparison receipt contains duplicate candidates.")
    mode = plan["comparison"]["mode"]
    repetitions = plan["comparison"]["repetitions"]
    case_ids = set(plan["minibatch"]["case_ids"])
    coordinate_keys: set[str] | None = None
    means: dict[str, float] = {}
    coordinates: dict[str, dict[str, float]] = {}
    selectable: dict[str, bool] = {}
    sample_count: int | None = None
    for candidate in receipt.candidates:
        required = expected[candidate.candidate_id]
        if (
            _verify_artifact(candidate.artifact_path, candidate.artifact_sha256)
            != required["artifact_path"]
            or candidate.artifact_sha256 != required["artifact_sha256"]
        ):
            raise typer.BadParameter(
                "Comparison candidate artifact does not match its durable receipt."
            )
        if len(candidate.samples) < repetitions or len(candidate.samples) > (
            plan["comparison"].get("max_repetitions") or repetitions
        ):
            raise typer.BadParameter(
                "Every candidate requires the configured matched repetition range."
            )
        if sample_count is None:
            sample_count = len(candidate.samples)
        elif len(candidate.samples) != sample_count:
            raise typer.BadParameter(
                "Every comparison candidate requires the same matched sample count."
            )
        samples_coordinates = [
            _sample_coordinates(sample, mode, case_ids) for sample in candidate.samples
        ]
        own_keys = set(samples_coordinates[0])
        if any(set(value) != own_keys for value in samples_coordinates[1:]):
            raise typer.BadParameter(
                "Repeated samples have inconsistent frontier coordinates."
            )
        if coordinate_keys is None:
            coordinate_keys = own_keys
        elif own_keys != coordinate_keys:
            raise typer.BadParameter(
                "Candidates have inconsistent frontier coordinate sets."
            )
        actual_sample_count = len(candidate.samples)
        means[candidate.candidate_id] = (
            sum(float(sample.score) for sample in candidate.samples)
            / actual_sample_count
        )
        coordinates[candidate.candidate_id] = {
            key: sum(sample[key] for sample in samples_coordinates)
            / actual_sample_count
            for key in own_keys
        }
        selectable[candidate.candidate_id] = all(
            sample.selectable and sample.infrastructure_valid
            for sample in candidate.samples
        )
    valid = [candidate_id for candidate_id in expected if selectable[candidate_id]]
    if "seed" in expected and "seed" not in valid:
        raise typer.BadParameter(
            "The frozen seed must be selectable in phase-one comparison."
        )
    if "incumbent" in expected and "incumbent" not in valid:
        raise typer.BadParameter(
            "The durable incumbent must be selectable in phase-two comparison."
        )
    if not valid:
        raise typer.BadParameter("No selectable candidates in comparison receipt.")

    def dominates(left: str, right: str) -> bool:
        return all(
            coordinates[left][key] >= coordinates[right][key]
            for key in coordinates[left]
        ) and any(
            coordinates[left][key] > coordinates[right][key]
            for key in coordinates[left]
        )

    frontier = [
        candidate_id
        for candidate_id in valid
        if not any(
            other != candidate_id and dominates(other, candidate_id) for other in valid
        )
    ]
    provisional = sorted(
        frontier, key=lambda candidate_id: (-means[candidate_id], candidate_id)
    )[0]
    baseline = "seed" if "seed" in expected else "incumbent"
    acceptance: dict[str, Any]
    if provisional == baseline:
        acceptance = {
            "verdict": "baseline",
            "confidence": plan["comparison"]["acceptance_confidence"],
            "min_delta": plan["comparison"]["acceptance_min_delta"],
        }
        selected = baseline
    else:
        baseline_samples = next(
            candidate.samples
            for candidate in receipt.candidates
            if candidate.candidate_id == baseline
        )
        candidate_samples = next(
            candidate.samples
            for candidate in receipt.candidates
            if candidate.candidate_id == provisional
        )
        acceptance = compare_candidate_samples(
            [float(sample.score) for sample in baseline_samples],
            [float(sample.score) for sample in candidate_samples],
            confidence=float(plan["comparison"]["acceptance_confidence"]),
            min_delta=float(plan["comparison"]["acceptance_min_delta"]),
        ).to_dict()
        if (
            acceptance["verdict"] == "inconclusive"
            and sample_count is not None
            and (
                sample_count
                < (plan["comparison"].get("max_repetitions") or repetitions)
            )
        ):
            raise typer.BadParameter(
                "Comparison remains inconclusive; submit matched samples through max_repetitions."
            )
        selected = provisional if acceptance["verdict"] == "accepted" else baseline
    return (
        selected,
        means,
        {
            "mode": mode,
            "frontier": frontier,
            "selectable": selectable,
            "coordinates": coordinates,
            "provisional_winner": provisional,
            "baseline": baseline,
            "sample_count": sample_count,
            "acceptance": acceptance,
        },
    )


def _usage(directory: Path, state: dict[str, Any]) -> dict[str, int]:
    optimization = sum(
        int(_receipt_data(directory, digest).get("metric_calls", 0))
        for digest in state.get("children", {}).values()
    )
    comparison = sum(
        int(_receipt_data(directory, digest).get("metric_calls", 0))
        for digest in state.get("comparisons", {}).values()
    )
    reporting = (
        int(_receipt_data(directory, state["reporting"]).get("metric_calls", 0))
        if state.get("reporting")
        else 0
    )
    return {
        "optimization_metric_calls": optimization,
        "comparison_metric_calls": comparison,
        "reporting_metric_calls": reporting,
        "accounted_metric_calls": optimization + comparison + reporting,
    }


@app.command("start")
def start(plan: Path = typer.Option(..., "--plan")) -> None:
    """Freeze a strict plan and emit one independent phase-one packet per child."""
    try:
        parsed = OmniPlan.model_validate_json(plan.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    raw = parsed.model_dump(mode="json")
    canonical = {"version": VERSION, **raw}
    canonical["plan_sha256"] = _digest(canonical)
    omni_id = f"omni-{uuid.uuid4().hex[:12]}"
    directory = _root() / omni_id
    _atomic(directory / "plan.json", canonical)
    state: dict[str, Any] = {
        "version": VERSION,
        "revision": 0,
        "omni_id": omni_id,
        "plan_sha256": canonical["plan_sha256"],
        "phase": "phase_one",
        "children": {},
        "dispatches": {},
        "comparisons": {},
        "reporting": None,
        "winner": None,
        "final": None,
        "receipts": [],
        "outbox": [],
        "event_keys": {},
        "event_sequence": 0,
        "packet_digests": {},
        "usage_attestation": "child_receipt",
    }
    for ordinal, child in enumerate(canonical["phase_one"], 1):
        path, packet = _packet(directory, "phase_one", child, canonical["seed"], state)
        _append_event_in_memory(
            state,
            "child_ready",
            {
                "packet_path": str(path),
                "packet_sha256": _file_sha256(path),
                "phase": "phase_one",
                "child_id": child["child_id"],
                "engine": child["engine"],
                "ordinal": ordinal,
                "reserved_metric_calls": child["metric_calls"],
                "workspace": child["workspace"],
            },
        )
    # Every phase-one packet and every initial event is durable together. A
    # crash before this point leaves no discoverable run; after it, `next`
    # repairs any event files that were not flushed yet.
    _save(directory, state)
    _flush_outbox(directory, state)
    typer.echo(
        json.dumps({"omni_id": omni_id, "state_path": str(directory / "state.json")})
    )


@app.command("status")
def status(omni_id: str, json_: bool = typer.Option(False, "--json")) -> None:
    """Show durable outer state without inspecting child traces."""
    _, state = _load(omni_id)
    typer.echo(
        json.dumps(state, sort_keys=True) if json_ else f"{omni_id}: {state['phase']}"
    )


@app.command("next")
def next_(
    omni_id: str,
    json_: bool = typer.Option(False, "--json"),
    wait: bool = typer.Option(False, "--wait"),
    timeout: float = typer.Option(0.0, "--timeout"),
) -> None:
    """Return the oldest unacked event; redelivery is byte-for-byte exact."""
    deadline = time.monotonic() + timeout
    while True:
        with _locked(omni_id) as (directory, state):
            pending = _pending(directory, state)
            if pending:
                typer.echo(
                    json.dumps(pending[0], sort_keys=True)
                    if json_
                    else pending[0]["id"]
                )
                return
        if not wait or time.monotonic() >= deadline:
            raise typer.Exit(code=3)
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


@app.command("ack")
@_serialized_transition
def ack(omni_id: str, event_id: str) -> None:
    """Ack only the next event; re-acknowledging an older known ID is harmless."""
    _directory, _state = _transition()
    events = _events(_directory)
    matching = next((event for event in events if event["id"] == event_id), None)
    if matching is None:
        raise typer.BadParameter("Unknown Omni event ID.")
    cursor = _directory / "cursor.json"
    acknowledged = (
        int(json.loads(cursor.read_text()).get("sequence", 0)) if cursor.exists() else 0
    )
    sequence = int(matching["sequence"])
    if sequence <= acknowledged:
        return
    if sequence != acknowledged + 1:
        raise typer.BadParameter("Only the oldest unacked event may be acknowledged.")
    _atomic(cursor, {"sequence": sequence, "event_id": event_id})


@app.command("child-dispatched")
@_serialized_transition
def child_dispatched(
    omni_id: str,
    receipt: Path = typer.Option(..., "--receipt"),
) -> None:
    """Record orchestrator dispatch metadata without executing arbitrary commands."""
    _directory, _state = _transition()
    try:
        raw = DispatchReceipt.model_validate_json(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    plan = _plan(_directory, _state)
    child = _expected_child(plan, raw.phase, raw.child_id)
    workspace = Path(child["workspace"])
    if not workspace.is_dir() or workspace.resolve() == repo_root().resolve():
        raise typer.BadParameter(
            "Dispatch requires the planned isolated workspace directory."
        )
    packet_digest = _expected_packet_digest(
        _directory, _state, f"{raw.phase}-{raw.child_id}.json"
    )
    packet = json.loads(
        (_directory / "packets" / f"{raw.phase}-{raw.child_id}.json").read_text(
            encoding="utf-8"
        )
    )
    try:
        _verify_child_frozen_artifacts(plan, child, packet["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise typer.BadParameter(
            "Frozen child input changed after plan start."
        ) from exc
    if raw.packet_sha256 != packet_digest:
        raise typer.BadParameter(
            "Dispatch receipt does not bind the exact child packet."
        )
    digest = _receipt(_directory, raw.model_dump(mode="json"))
    old = _state["dispatches"].get(raw.child_id)
    if old and old != digest:
        raise typer.BadParameter("Conflicting duplicate child dispatch.")
    if not old:
        _state["dispatches"][raw.child_id] = digest
        _state["receipts"].append(digest)
        _save(_directory, _state)


@app.command("child-submit")
@_serialized_transition
def child_submit(
    omni_id: str,
    receipt: Path = typer.Option(..., "--receipt"),
) -> None:
    """Record one immutable child result; duplicates are no-ops, conflicts fail."""
    _directory, _state = _transition()
    try:
        raw = ChildReceipt.model_validate_json(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    plan = _plan(_directory, _state)
    child = _expected_child(plan, raw.phase, raw.child_id)
    payload = raw.model_dump(mode="json")
    packet_sha256 = _expected_packet_digest(
        _directory, _state, f"{raw.phase}-{raw.child_id}.json"
    )
    packet = json.loads(
        (_directory / "packets" / f"{raw.phase}-{raw.child_id}.json").read_text(
            encoding="utf-8"
        )
    )
    try:
        _verify_child_frozen_artifacts(plan, child, packet["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise typer.BadParameter(
            "Frozen child input changed after plan start."
        ) from exc
    if raw.packet_sha256 != packet_sha256:
        raise typer.BadParameter(
            "Child receipt does not bind the exact durable packet."
        )
    dispatch_digest = _state["dispatches"].get(raw.child_id)
    if dispatch_digest is None:
        raise typer.BadParameter(
            "Child receipt requires a prior durable child-dispatched receipt."
        )
    if _receipt_data(_directory, dispatch_digest).get("packet_sha256") != packet_sha256:
        raise typer.BadParameter("Child dispatch does not bind the submitted packet.")
    try:
        payload["candidate_artifact_path"] = _verify_artifact(
            raw.candidate_artifact_path, raw.candidate_artifact_sha256
        )
        _normalize_evidence(payload)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    digest = _digest(payload)
    old = _state["children"].get(raw.child_id)
    if old:
        if old == digest:
            _receipt_data(_directory, old)
            return
        raise typer.BadParameter("Conflicting duplicate child submission.")
    if raw.phase != _state["phase"]:
        raise typer.BadParameter("Child receipt phase does not match durable state.")
    expected_seed = (
        plan["seed"]["sha256"]
        if raw.phase == "phase_one"
        else _state.get("winner", {}).get("artifact_sha256")
    )
    if raw.plan_sha256 != _state["plan_sha256"] or raw.seed_sha256 != expected_seed:
        raise typer.BadParameter(
            "Child receipt does not bind the frozen plan and seed."
        )
    if raw.engine != child["engine"] or raw.metric_calls > child["metric_calls"]:
        raise typer.BadParameter(
            "Child receipt engine or metric usage exceeds its reserved packet slice."
        )
    stored = _receipt(_directory, payload)
    _state["children"][raw.child_id] = stored
    _state["receipts"].append(stored)
    if raw.phase == "phase_one" and all(
        child_def["child_id"] in _state["children"] for child_def in plan["phase_one"]
    ):
        path, packet = _comparison_packet(_directory, _state, "phase_one")
        _queue_event(
            _directory,
            _state,
            "fair_compare_ready",
            {
                "phase": "phase_one",
                "packet_path": str(path),
                "packet_sha256": _file_sha256(path),
                "reserved_metric_calls": plan["comparison"]["metric_calls"],
            },
        )
    elif raw.phase == "phase_two":
        path, packet = _comparison_packet(_directory, _state, "phase_two")
        _queue_event(
            _directory,
            _state,
            "fair_compare_ready",
            {
                "phase": "phase_two",
                "packet_path": str(path),
                "packet_sha256": _file_sha256(path),
                "reserved_metric_calls": plan["comparison"].get(
                    "phase_two_metric_calls"
                )
                or plan["comparison"]["metric_calls"],
            },
        )
    else:
        _save(_directory, _state)


@app.command("compare-submit")
@_serialized_transition
def compare_submit(
    omni_id: str,
    receipt: Path = typer.Option(..., "--receipt"),
) -> None:
    """Accept exact matched comparison work and advance one durable phase."""
    _directory, _state = _transition()
    try:
        raw = ComparisonReceipt.model_validate_json(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = raw.model_dump(mode="json")
    try:
        _normalize_comparison_paths(payload)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    digest = _digest(payload)
    plan = _plan(_directory, _state)
    try:
        _verify_artifact(
            plan["minibatch"]["artifact_path"], plan["minibatch"]["sha256"]
        )
    except ValueError as exc:
        raise typer.BadParameter(
            "Frozen comparison minibatch changed after plan start."
        ) from exc
    expected_packet_digest = _expected_packet_digest(
        _directory, _state, f"compare-{raw.phase}.json"
    )
    if raw.packet_sha256 != expected_packet_digest:
        raise typer.BadParameter(
            "Comparison receipt does not bind the exact comparison packet."
        )
    allowed_calls = (
        plan["comparison"]["metric_calls"]
        if raw.phase == "phase_one"
        else plan["comparison"].get("phase_two_metric_calls")
        or plan["comparison"]["metric_calls"]
    )
    sample_counts = {len(candidate.samples) for candidate in raw.candidates}
    if len(sample_counts) != 1:
        raise typer.BadParameter("Comparison candidates must have equal repetitions.")
    submitted_repetitions = next(iter(sample_counts))
    semantic_calls = (
        len(raw.candidates) * submitted_repetitions * plan["minibatch"]["case_count"]
    )
    if (
        raw.plan_sha256 != _state["plan_sha256"]
        or raw.evaluator_identity != plan["evaluator_identity"]
        or raw.evaluator_sha256 != plan["evaluator_sha256"]
        or raw.minibatch_sha256 != plan["minibatch"]["sha256"]
        or raw.metric_calls != semantic_calls
        or semantic_calls > allowed_calls
    ):
        raise typer.BadParameter(
            "Comparison receipt does not match frozen identity or reserved budget."
        )
    expected = _expected_candidates(_directory, _state, raw.phase)
    # A duplicate remains a no-op only after all frozen inputs and the
    # previously named immutable receipt still validate. This keeps retries
    # from masking plan/packet/receipt tampering after a phase transition.
    known = _state["comparisons"].get(raw.phase)
    if known:
        if known != digest:
            raise typer.BadParameter("Conflicting duplicate comparison submission.")
        _receipt_data(_directory, known)
        _select_comparison(raw, plan, expected)
        return
    if raw.phase != _state["phase"]:
        raise typer.BadParameter(
            "Comparison receipt phase does not match durable state."
        )
    selected, means, decision = _select_comparison(raw, plan, expected)
    stored = _receipt(_directory, payload)
    _state["comparisons"][raw.phase] = stored
    _state["receipts"].append(stored)
    if raw.phase == "phase_one":
        artifact = expected[selected]
        _state["winner"] = {
            "candidate": selected,
            **artifact,
            "mean_score": means[selected],
            "comparison_receipt": stored,
            "decision": decision,
        }
        _state["phase"] = "phase_two"
        path, packet = _packet(
            _directory, "phase_two", plan["phase_two"], artifact, _state
        )
        _queue_event(
            _directory,
            _state,
            "phase2_ready",
            {
                "packet_path": str(path),
                "packet_sha256": _file_sha256(path),
                "phase": "phase_two",
                "child_id": plan["phase_two"]["child_id"],
                "engine": plan["phase_two"]["engine"],
                "reserved_metric_calls": plan["phase_two"]["metric_calls"],
                "workspace": plan["phase_two"]["workspace"],
                "seed_artifact_sha256": artifact["artifact_sha256"],
            },
        )
        return
    artifact = expected[selected]
    _state["final"] = {
        "candidate": selected,
        **artifact,
        "mean_score": means[selected],
        "comparison_receipt": stored,
        "decision": decision,
    }
    if plan.get("reporting"):
        _state["phase"] = "reporting"
        reporting = plan["reporting"]
        path, packet = _reporting_packet(_directory, _state)
        _queue_event(
            _directory,
            _state,
            "reporting_ready",
            {
                "packet_path": str(path),
                "packet_sha256": _file_sha256(path),
                "reserved_metric_calls": reporting["metric_calls"],
            },
        )
        return
    _state["phase"] = "done"
    _state["usage"] = _usage(_directory, _state)
    _queue_event(
        _directory,
        _state,
        "omni_done",
        {
            "final_candidate": selected,
            "artifact_path": artifact["artifact_path"],
            "artifact_sha256": artifact["artifact_sha256"],
            "state_path": str(_directory / "state.json"),
        },
    )


@app.command("reporting-submit")
@_serialized_transition
def reporting_submit(
    omni_id: str,
    receipt: Path = typer.Option(..., "--receipt"),
) -> None:
    """Record optional reporting-only test work and complete the Omni run."""
    _directory, _state = _transition()
    try:
        raw = ReportingReceipt.model_validate_json(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = raw.model_dump(mode="json")
    plan = _plan(_directory, _state)
    reporting = plan.get("reporting")
    final = _state.get("final") or {}
    expected_packet_digest = _expected_packet_digest(
        _directory, _state, "reporting.json"
    )
    if raw.packet_sha256 != expected_packet_digest:
        raise typer.BadParameter(
            "Reporting receipt does not bind the exact reporting packet."
        )
    if reporting is not None:
        try:
            _verify_artifact(
                reporting["test_set"]["artifact_path"],
                reporting["test_set"]["sha256"],
            )
        except ValueError as exc:
            raise typer.BadParameter(
                "Frozen reporting test set changed after plan start."
            ) from exc
    try:
        payload["candidate_artifact_path"] = _verify_artifact(
            raw.candidate_artifact_path, raw.candidate_artifact_sha256
        )
        _normalize_evidence(payload)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    digest = _digest(payload)
    if _state.get("reporting"):
        if _state["reporting"] == digest:
            _receipt_data(_directory, _state["reporting"])
            return
        raise typer.BadParameter("Conflicting duplicate reporting submission.")
    if _state["phase"] != "reporting":
        raise typer.BadParameter("No reporting work is pending.")
    if (
        not reporting
        or raw.plan_sha256 != _state["plan_sha256"]
        or raw.evaluator_identity != plan["evaluator_identity"]
        or raw.evaluator_sha256 != plan["evaluator_sha256"]
        or raw.test_set_sha256 != reporting["test_set"]["sha256"]
        or raw.metric_calls != reporting["test_set"]["case_count"]
        or raw.metric_calls > reporting["metric_calls"]
        or raw.candidate_artifact_sha256 != final.get("artifact_sha256")
    ):
        raise typer.BadParameter(
            "Reporting receipt does not match frozen final candidate or reporting plan."
        )
    if payload["candidate_artifact_path"] != final.get("artifact_path"):
        raise typer.BadParameter(
            "Reporting receipt candidate path does not match final artifact."
        )
    stored = _receipt(_directory, payload)
    _state["reporting"] = stored
    _state["receipts"].append(stored)
    _state["final"]["reporting_receipt"] = stored
    _state["final"]["reporting_score"] = float(raw.score)
    _state["phase"] = "done"
    _state["usage"] = _usage(_directory, _state)
    _queue_event(
        _directory,
        _state,
        "omni_done",
        {
            "final_candidate": final["candidate"],
            "artifact_path": final["artifact_path"],
            "artifact_sha256": final["artifact_sha256"],
            "state_path": str(_directory / "state.json"),
        },
    )


@app.command("error-submit")
@_serialized_transition
def error_submit(
    omni_id: str,
    claim: Path = typer.Option(..., "--claim"),
) -> None:
    """Persist only bound evaluator-unattainability claims as ERROR artifacts."""
    _directory, _state = _transition()
    try:
        raw = EscalationClaim.model_validate_json(claim.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = raw.model_dump(mode="json")
    semantic_receipts = {
        *_state.get("children", {}).values(),
        *_state.get("comparisons", {}).values(),
        *([_state["reporting"]] if _state.get("reporting") else []),
    }
    if raw.source_receipt_sha256 not in semantic_receipts:
        raise typer.BadParameter(
            "Escalation must bind a semantic child, comparison, or reporting receipt."
        )
    plan = _plan(_directory, _state)
    case_ids = set(
        plan["reporting"]["test_set"]["case_ids"]
        if raw.source_receipt_sha256 == _state.get("reporting")
        else plan["minibatch"]["case_ids"]
    )
    if not set(raw.case_ids) <= case_ids:
        raise typer.BadParameter("Escalation case IDs must be in the frozen minibatch.")
    try:
        _normalize_evidence(payload)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    source = _receipt_data(_directory, raw.source_receipt_sha256)
    declared_evidence = {
        (item["artifact_path"], item["sha256"]) for item in source.get("evidence", [])
    }
    if not declared_evidence or any(
        (item["artifact_path"], item["sha256"]) not in declared_evidence
        for item in payload["evidence"]
    ):
        raise typer.BadParameter(
            "Escalation evidence must be declared by its bound semantic receipt."
        )
    digest = _digest(payload)
    if digest in _state.setdefault("escalations", []):
        return
    stored = _receipt(_directory, payload)
    claim_receipt_path = _directory / "receipts" / f"{stored}.json"
    markdown = _directory / "errors" / f"{stored}.md"
    _atomic_text(
        markdown,
        f"# Evaluation target escalation\n\nClaim SHA: `{stored}`\n\n{raw.claim}\n\nCase IDs: {', '.join(raw.case_ids)}\n\nEvidence:\n"
        + "\n".join(
            f"- {item['artifact_path']} ({item['sha256']})"
            for item in payload["evidence"]
        )
        + f"\n\nRequested action: {raw.requested_action}\n",
    )
    index = _directory / "ERROR.md"
    existing = (
        index.read_text(encoding="utf-8")
        if index.exists()
        else "# Evaluation target escalations\n"
    )
    if f"{stored}.md" not in existing:
        _atomic_text(index, existing + f"\n- [{stored}](errors/{stored}.md)\n")
    _state["escalations"].append(stored)
    _state["receipts"].append(stored)
    _queue_event(
        _directory,
        _state,
        "error_escalation",
        {
            "error_path": str(markdown),
            "claim_path": str(claim_receipt_path),
            "claim_sha256": stored,
        },
    )

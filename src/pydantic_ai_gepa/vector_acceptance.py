"""Generic keyed-vector acceptance contracts and durable rollout records.

The optimizer deliberately treats assertion keys and latency payloads as opaque.
Harnesses own their meaning through a comparator configured from ``gepa.toml``.
In pinned-scorer mode, component-map keys are the exact relative file paths
declared by ``acceptance.component_files``; harnesses must accept those paths
as component ids.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable


VectorVerdict = Literal["accepted", "rejected", "equivalent", "needs_escalation"]
VECTOR_RECORD_VERSION = 1


@dataclass(frozen=True, slots=True)
class VectorRecordKey:
    """Identity fields that make vector samples safe to pool."""

    run_id: str
    inventory_hash: str
    scorer_identity: str
    incumbent_hash: str
    candidate_hash: str
    repetition: int
    vector_schema_version: str
    telemetry_schema_version: str


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One scored rollout's opaque assertion vector and latency payload."""

    key: VectorRecordKey
    assertions: Mapping[str, Any]
    latency: Mapping[str, Any]
    display_score: float
    outcome: Literal["scored", "infra_error"] = "scored"
    error: Mapping[str, Any] | None = None
    record_version: int = VECTOR_RECORD_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_version": self.record_version,
            "key": asdict(self.key),
            "assertions": dict(self.assertions),
            "latency": dict(self.latency),
            "display_score": self.display_score,
            "outcome": self.outcome,
            "error": dict(self.error) if self.error else None,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "VectorRecord":
        key = raw.get("key")
        if not isinstance(key, Mapping):
            raise ValueError("Vector record is missing its key.")
        outcome = raw.get("outcome", "scored")
        if outcome not in {"scored", "infra_error"}:
            raise ValueError("Vector record outcome must be 'scored' or 'infra_error'.")
        return cls(
            key=VectorRecordKey(
                run_id=str(key["run_id"]),
                inventory_hash=str(key["inventory_hash"]),
                scorer_identity=str(key["scorer_identity"]),
                incumbent_hash=str(key["incumbent_hash"]),
                candidate_hash=str(key["candidate_hash"]),
                repetition=int(key["repetition"]),
                vector_schema_version=str(key["vector_schema_version"]),
                telemetry_schema_version=str(key["telemetry_schema_version"]),
            ),
            assertions=dict(raw.get("assertions") or {}),
            latency=dict(raw.get("latency") or {}),
            display_score=float(raw["display_score"]),
            outcome=outcome,
            error=dict(raw["error"]) if isinstance(raw.get("error"), Mapping) else None,
            record_version=int(raw.get("record_version", VECTOR_RECORD_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class VectorComparisonRequest:
    """Opaque samples supplied to a harness-defined acceptance comparator.

    ``journal_context`` carries durable orchestration facts without assigning
    meaning to assertion keys. Vector lane comparisons provide
    ``accepted_promotion_count`` and ``run_start_baseline``; scheduled
    comparisons additionally set ``comparison_kind = 'run_start_rebaseline'``.
    """

    incumbent: tuple[VectorRecord, ...]
    candidate: tuple[VectorRecord, ...]
    attempt: int
    escalation: int
    journal_context: Mapping[str, Any] = field(default_factory=dict)

    def validate_compatible(self) -> None:
        records = (*self.incumbent, *self.candidate)
        if not self.incumbent or not self.candidate:
            raise ValueError("Comparator requires incumbent and candidate records.")
        reference = self.incumbent[0].key
        candidate_hashes = {record.key.candidate_hash for record in self.candidate}
        if len(candidate_hashes) != 1:
            raise ValueError(
                "Refusing vector comparison across more than one candidate hash."
            )
        for record in records:
            key = record.key
            if (
                key.run_id != reference.run_id
                or key.inventory_hash != reference.inventory_hash
                or key.scorer_identity != reference.scorer_identity
                or key.vector_schema_version != reference.vector_schema_version
                or key.telemetry_schema_version != reference.telemetry_schema_version
                or key.incumbent_hash != reference.incumbent_hash
            ):
                raise ValueError(
                    "Refusing vector comparison across run, inventory, scorer, "
                    "incumbent, or schema identities."
                )


@dataclass(frozen=True, slots=True)
class VectorComparison:
    """Comparator result consumed generically by lanes and selection."""

    verdict: VectorVerdict
    ranking_key: tuple[float, ...] = ()
    display_score: float = 0.0
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        return self.verdict == "accepted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "ranking_key": list(self.ranking_key),
            "display_score": self.display_score,
            "detail": dict(self.detail),
            "improved": self.improved,
        }


@runtime_checkable
class VectorComparator(Protocol):
    """Harness extension point for keyed-vector acceptance."""

    def compare(self, request: VectorComparisonRequest) -> VectorComparison: ...


def resolve_vector_comparator(
    ref: str, *, expected_root: Path | None = None
) -> VectorComparator:
    """Resolve a ``module:factory`` comparator and validate its narrow API."""
    if ":" not in ref:
        raise ValueError("Comparator must be a 'module:factory' reference.")
    module_name, attr = ref.split(":", 1)
    module = importlib.import_module(module_name)
    if expected_root is not None:
        module_file = Path(str(getattr(module, "__file__", ""))).resolve()
        try:
            module_file.relative_to(expected_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Comparator {ref!r} is outside pinned scorer root."
            ) from exc
    factory = getattr(module, attr)
    if inspect.isclass(factory):
        try:
            comparator = factory()
        except TypeError as exc:
            raise TypeError(
                "Comparator classes must be constructible without arguments."
            ) from exc
    else:
        comparator = (
            factory()
            if callable(factory) and not hasattr(factory, "compare")
            else factory
        )
    if inspect.isclass(comparator):
        raise TypeError("Comparator factory returned a class instead of an instance.")
    if not isinstance(comparator, VectorComparator):
        raise TypeError(
            "Comparator factory must return an object with compare(request)."
        )
    return comparator


def compare_vectors(
    comparator: VectorComparator, request: VectorComparisonRequest
) -> VectorComparison:
    """Invoke a comparator after enforcing record compatibility and result shape."""
    request.validate_compatible()
    result = comparator.compare(request)
    if not isinstance(result, VectorComparison):
        raise TypeError("Vector comparator must return VectorComparison.")
    return result


class VectorRecordStore:
    """Append-only per-run vector ledger, including pooled incumbent samples."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: VectorRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def records(self) -> list[VectorRecord]:
        if not self.path.exists():
            return []
        output: list[VectorRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            output.append(VectorRecord.from_dict(json.loads(line)))
        return output

    def matching(
        self, key: VectorRecordKey, *, candidate_hash: str | None = None
    ) -> list[VectorRecord]:
        target = candidate_hash or key.candidate_hash
        matches = [
            record
            for record in self.records()
            if record.key.run_id == key.run_id
            and record.key.inventory_hash == key.inventory_hash
            and record.key.scorer_identity == key.scorer_identity
            and record.key.incumbent_hash == key.incumbent_hash
            and record.key.candidate_hash == target
            and record.key.vector_schema_version == key.vector_schema_version
            and record.key.telemetry_schema_version == key.telemetry_schema_version
            and record.outcome == "scored"
        ]
        return sorted(matches, key=lambda item: item.key.repetition)

    def matching_incumbent_for_case(
        self, key: VectorRecordKey, *, case_id: str
    ) -> list[VectorRecord]:
        """Find pooled incumbent reps containing ``case_id`` across inventories.

        A probe intentionally has a one-case inventory hash, while its
        incumbent was scored on the frozen full inventory. All other identity
        and schema dimensions remain binding.
        """
        matches = [
            record
            for record in self.records()
            if record.key.run_id == key.run_id
            and record.key.scorer_identity == key.scorer_identity
            and record.key.incumbent_hash == key.incumbent_hash
            and record.key.candidate_hash == key.incumbent_hash
            and record.key.vector_schema_version == key.vector_schema_version
            and record.key.telemetry_schema_version == key.telemetry_schema_version
            and record.outcome == "scored"
            and case_id in record.assertions
        ]
        return sorted(matches, key=lambda item: item.key.repetition)

    def records_for_keys(
        self, keys: Sequence[Mapping[str, Any]]
    ) -> list[VectorRecord]:
        """Load the most recent persisted record for each exact vector key.

        Re-baseline scheduling stores run-start keys, rather than deriving a
        baseline from whatever compatible records happen to exist later.
        Selecting the last matching append makes a crash/retry replacement
        deterministic while preserving the requested key order.
        """
        requested = [json.dumps(dict(key), sort_keys=True) for key in keys]
        latest = {
            json.dumps(asdict(record.key), sort_keys=True): record
            for record in self.records()
            if record.outcome == "scored"
        }
        missing = [key for key in requested if key not in latest]
        if missing:
            raise ValueError("A stored run-start vector record is unavailable.")
        return [latest[key] for key in requested]


def inventory_hash(case_ids: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(case_ids)).encode()).hexdigest()


def scorer_identity(refs: Sequence[str | None], *, root: Path | None = None) -> str:
    """Hash configured scorer references and their resolved module source bytes."""
    digest = hashlib.sha256()
    for ref in refs:
        digest.update((ref or "").encode())
        digest.update(b"\0")
        if ref is None:
            continue
        module_name = ref.split(":", 1)[0]
        source_path = _resolve_module_source(module_name, root=root)
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_module_source(module_name: str, *, root: Path | None) -> Path:
    """Resolve one module without making an import cache part of its identity."""
    relative = Path(*module_name.split("."))
    candidates: list[Path] = []
    if root is not None:
        resolved_root = root.resolve()
        for prefix in (resolved_root, resolved_root / "src"):
            candidates.extend(
                (
                    prefix / relative.with_suffix(".py"),
                    prefix / relative / "__init__.py",
                )
            )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin or spec.origin in {"built-in", "frozen"}:
        raise ValueError(
            f"Could not resolve source bytes for scorer module {module_name!r}."
        )
    source = Path(spec.origin).resolve()
    if not source.is_file():
        raise ValueError(f"Scorer module {module_name!r} has no readable source file.")
    return source


def side_info_vector(records: Sequence[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract opaque per-case assertion and latency data from metric records."""
    assertions: dict[str, Any] = {}
    latency: dict[str, Any] = {}
    for record in records:
        payload = getattr(record, "payload", {})
        info = payload.get("side_info") if isinstance(payload, Mapping) else None
        if not isinstance(info, Mapping):
            trajectory = (
                payload.get("trajectory") if isinstance(payload, Mapping) else None
            )
            info = getattr(trajectory, "metric_side_info", None)
        if not isinstance(info, Mapping):
            continue
        case_id = str(getattr(record, "case_id"))
        if "assertions" in info:
            assertions[case_id] = info["assertions"]
        if "latency" in info:
            latency[case_id] = info["latency"]
    return assertions, latency

"""Per-run on-disk state: minibatch persistence + Pareto append-only log.

Each `gepa eval` invocation reads/writes through `MinibatchStore` and
`ParetoLog`. The Pareto row schema records the tuple required by
pydanticaigepa-dec-xd6 — `(candidate_id, commit_sha, component_overrides_id,
minibatch_id, per_case_scores, mean_score, status, summary, timestamp)` — so
historical runs can be reconstructed by checking out the commit and replaying
the component overrides file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .layout import (
    minibatch_path,
    pareto_log_path,
    run_dir,
)


# ----------------------------- helpers ---------------------------------


def current_commit_sha(root: Path | None = None) -> str | None:
    """Return the short commit sha at HEAD, or None if not in a git repo."""
    cwd = str(root) if root is not None else None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return completed.stdout.strip() or None


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp suitable for the Pareto log."""
    return datetime.now(timezone.utc).isoformat()


# ----------------------------- minibatches -----------------------------


@dataclass(frozen=True)
class Minibatch:
    """A frozen sampling of case ids drawn from the dataset."""

    id: str
    case_ids: list[str]
    seed: int
    epoch: int
    size: int
    sampled_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "case_ids": list(self.case_ids),
            "seed": self.seed,
            "epoch": self.epoch,
            "size": self.size,
            "sampled_at": self.sampled_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Minibatch:
        return Minibatch(
            id=data["id"],
            case_ids=list(data["case_ids"]),
            seed=int(data["seed"]),
            epoch=int(data["epoch"]),
            size=int(data["size"]),
            sampled_at=str(data["sampled_at"]),
        )


def _hash_minibatch(case_ids: Sequence[str], seed: int, epoch: int) -> str:
    payload = json.dumps(
        {"case_ids": list(case_ids), "seed": seed, "epoch": epoch}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


class MinibatchStore:
    """Persist deterministic minibatch references under `.gepa/runs/<run_id>/minibatches/`."""

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self._run_id = run_id
        self._root = root
        self._dir = run_dir(run_id, root) / "minibatches"

    @property
    def dir(self) -> Path:
        return self._dir

    def sample(
        self,
        case_ids: Iterable[str],
        size: int,
        *,
        seed: int = 0,
        epoch: int = 0,
    ) -> Minibatch:
        """Deterministically sample ``size`` case ids from ``case_ids``.

        Re-running with the same ``(seed, epoch, case_id_set, size)`` always
        returns the same minibatch id and the same ordering.
        """
        pool = sorted(set(case_ids))
        if size > len(pool):
            size = len(pool)
        if size < 0:
            raise ValueError(f"size must be non-negative, got {size}")

        # Combining seed + epoch keeps successive epochs deterministic but distinct.
        # `random.Random` accepts int/str/bytes seeds; we derive a deterministic
        # int from the (seed, epoch) tuple.
        combined_seed = hashlib.sha256(
            json.dumps([seed, epoch], sort_keys=True).encode("utf-8")
        ).hexdigest()
        rng = random.Random(combined_seed)
        chosen: list[str] = rng.sample(pool, size) if size > 0 else []

        mb_id = _hash_minibatch(chosen, seed, epoch)
        minibatch = Minibatch(
            id=mb_id,
            case_ids=chosen,
            seed=seed,
            epoch=epoch,
            size=size,
            sampled_at=utc_now_iso(),
        )
        self.save(minibatch)
        return minibatch

    def save(self, minibatch: Minibatch) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = minibatch_path(self._run_id, minibatch.id, self._root)
        path.write_text(json.dumps(minibatch.to_dict(), indent=2), encoding="utf-8")
        return path

    def load(self, mb_id: str) -> Minibatch:
        path = minibatch_path(self._run_id, mb_id, self._root)
        if not path.exists():
            raise FileNotFoundError(f"No minibatch at {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return Minibatch.from_dict(data)

    def list_ids(self) -> list[str]:
        if not self._dir.is_dir():
            return []
        return sorted(p.stem for p in self._dir.iterdir() if p.suffix == ".json")


# ----------------------------- Pareto log ------------------------------


ParetoStatus = str  # 'baseline' | 'proposal' | 'evaluated' | 'candidate' | other


@dataclass(frozen=True)
class ParetoRow:
    """A single row of the Pareto history."""

    candidate_id: str
    commit_sha: str | None
    component_overrides_id: str | None
    minibatch_id: str
    per_case_scores: dict[str, float]
    mean_score: float
    status: ParetoStatus
    summary: str
    timestamp: str
    lane: str | None = None
    objective_scores: dict[str, float] = field(default_factory=dict)
    per_case_objective_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "candidate_id": self.candidate_id,
            "commit_sha": self.commit_sha,
            "component_overrides_id": self.component_overrides_id,
            "minibatch_id": self.minibatch_id,
            "per_case_scores": dict(self.per_case_scores),
            "mean_score": self.mean_score,
            "status": self.status,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "lane": self.lane,
            "objective_scores": dict(self.objective_scores),
            "per_case_objective_scores": {
                case_id: dict(scores)
                for case_id, scores in self.per_case_objective_scores.items()
            },
        }
        if self.extra:
            out["extra"] = dict(self.extra)
        return out

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ParetoRow:
        # Rows written before lanes existed carry no "lane" key; default to None.
        raw_per_case_objectives = data.get("per_case_objective_scores", {})
        if not isinstance(raw_per_case_objectives, dict):
            raise ValueError("per_case_objective_scores must be a mapping.")
        return ParetoRow(
            candidate_id=str(data["candidate_id"]),
            commit_sha=data.get("commit_sha"),
            component_overrides_id=data.get("component_overrides_id"),
            minibatch_id=str(data["minibatch_id"]),
            per_case_scores={
                str(k): float(v) for k, v in data.get("per_case_scores", {}).items()
            },
            mean_score=float(data["mean_score"]),
            status=str(data["status"]),
            summary=str(data.get("summary", "")),
            timestamp=str(data["timestamp"]),
            lane=data.get("lane"),
            objective_scores=_finite_objective_scores(data.get("objective_scores", {})),
            per_case_objective_scores={
                str(case_id): _finite_objective_scores(scores)
                for case_id, scores in raw_per_case_objectives.items()
            },
            extra=dict(data.get("extra", {})),
        )


def new_candidate_id() -> str:
    """Return a short stable identifier suitable for a candidate or proposal."""
    return uuid.uuid4().hex[:12]


def new_eval_id() -> str:
    """Return a unique per-eval identifier.

    Report and trace artifacts are keyed by ``(eval_id, candidate_id)`` rather
    than the global iteration ordinal so concurrent lane evals of identical
    candidate trees can never collide on filenames (pydanticaigepa-dec-msy).
    """
    return uuid.uuid4().hex[:8]


class ParetoLog:
    """Append-only JSONL ledger of evaluation events for a run."""

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self._run_id = run_id
        self._root = root
        self._path = pareto_log_path(run_id, root)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, row: ParetoRow) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row.to_dict(), sort_keys=True)
        # One unbuffered O_APPEND write per row: concurrent writers never
        # interleave bytes within a row, and a killed writer can only ever
        # leave one trailing partial line (which readers tolerate below) —
        # never poison the rows that follow (spec-1do: the append is the sole
        # budget authority; row count must equal completed evals).
        # Defensive newline: if a previous writer was killed mid-row (the
        # file does not end in a newline), terminate its partial line first
        # so the torn bytes can never merge with and destroy this row. A race
        # between two writers doing this yields at worst a blank line, which
        # readers skip.
        prefix = b""
        try:
            probe = os.open(self._path, os.O_RDONLY)
            try:
                size = os.fstat(probe).st_size
                if size > 0 and os.pread(probe, 1, size - 1) != b"\n":
                    prefix = b"\n"
            finally:
                os.close(probe)
        except FileNotFoundError:
            pass
        fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, prefix + (line + "\n").encode("utf-8"))
        finally:
            os.close(fd)

    @staticmethod
    def _parse_line(line: str) -> ParetoRow | None:
        stripped = line.strip()
        if not stripped:
            return None
        try:
            return ParetoRow.from_dict(json.loads(stripped))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _is_valid_row_line(line: str) -> bool:
        return ParetoLog._parse_line(line) is not None

    def _read_lines(self) -> list[str]:
        if not self._path.exists():
            return []
        return self._path.read_text(encoding="utf-8").splitlines()

    def iter_rows(self) -> list[ParetoRow]:
        rows: list[ParetoRow] = []
        # Unparseable lines (a writer killed mid-append leaves a torn partial
        # row) are skipped with a warning, never fatal: the ledger stays
        # readable for `gepa pareto`, final reports, and select.
        for line in self._read_lines():
            row = self._parse_line(line)
            if row is None:
                if line.strip():
                    print(
                        f"warning: ignoring unparseable pareto row in "
                        f"{self._path} (a writer was likely killed mid-append)",
                        file=sys.stderr,
                    )
                continue
            rows.append(row)
        return rows

    def count_rows(self, *, scope: str = "acceptance") -> int:
        """Count completed evals: the number of parseable rows in the log.

        A torn partial row from a killed writer does not count — the eval did
        not complete, so it did not finalize its budget spend. This is the
        single budget count source (pydanticaigepa-dec-msy).
        """
        count = 0
        for line in self._read_lines():
            row = self._parse_line(line)
            if row is not None and row.extra.get("row_scope", "acceptance") == scope:
                count += 1
        return count

    def count_budget_rows(self) -> int:
        """Count training and validation evaluations charged to run budget."""

        return sum(
            self.count_rows(scope=scope) for scope in ("acceptance", "validation")
        )

    def front(self, *, mode: str = "instance") -> list[ParetoRow]:
        """Return the selectable Pareto front with complete matching coordinates.

        Rows must expose identical coordinates for the requested frontier mode;
        partial rows fail closed instead of being treated as incomparable.
        """
        if mode not in {"instance", "objective", "hybrid", "cartesian"}:
            raise ValueError(f"Unknown Pareto frontier mode: {mode!r}.")
        rows = self.selectable_rows()
        if not rows:
            return []
        coordinates = [_row_coordinates(row, mode) for row in rows]
        if any(coordinate is None for coordinate in coordinates):
            raise ValueError(
                f"{mode} frontier requires complete coordinates for every selectable row."
            )
        first_coordinates = set(coordinates[0] or {})
        if any(
            set(coordinate or {}) != first_coordinates for coordinate in coordinates[1:]
        ):
            raise ValueError(
                "Pareto frontier rows must have identical coordinate keys; refusing incomparable partial rows."
            )

        front: list[ParetoRow] = []
        for candidate in rows:
            dominated = False
            front_after: list[ParetoRow] = []
            for existing in front:
                cmp = _row_dominance(existing, candidate, mode=mode)
                if cmp == "existing":
                    dominated = True
                    front_after.append(existing)
                elif cmp == "candidate":
                    # existing is dominated, drop it
                    continue
                else:
                    front_after.append(existing)
            if not dominated:
                front_after.append(candidate)
            front = front_after
        return front

    def selectable_rows(self) -> list[ParetoRow]:
        """Return rows whose evaluations may participate in quality ranking."""

        return [
            row
            for row in self.iter_rows()
            if row.extra.get("selectable", True) is not False
            and row.extra.get("outcome") != "infrastructure_failure"
            and row.extra.get("row_scope", "acceptance") == "acceptance"
        ]

    def validation_rows(self) -> list[ParetoRow]:
        """Return held-out selection rows, excluding infrastructure failures."""

        return [
            row
            for row in self.iter_rows()
            if row.extra.get("selectable", True) is not False
            and row.extra.get("outcome") != "infrastructure_failure"
            and row.extra.get("row_scope") == "validation"
        ]


def _dominance(a: dict[str, float], b: dict[str, float]) -> str | None:
    """Return 'existing' if a dominates b, 'candidate' if b dominates a, else None.

    Returns ``None`` for incomparable rows (different case sets or mutually
    non-dominating scores).
    """
    shared = set(a) & set(b)
    if not shared:
        return None
    a_ge_all = True
    b_ge_all = True
    a_gt_any = False
    b_gt_any = False
    for case in shared:
        if a[case] < b[case]:
            a_ge_all = False
        elif a[case] > b[case]:
            a_gt_any = True
        if b[case] < a[case]:
            b_ge_all = False
        elif b[case] > a[case]:
            b_gt_any = True
    if a_ge_all and a_gt_any:
        return "existing"
    if b_ge_all and b_gt_any:
        return "candidate"
    return None


def _finite_objective_scores(raw: Any) -> dict[str, float]:
    """Validate named higher-is-better objective coordinates at the ledger edge."""
    if not isinstance(raw, dict):
        raise ValueError("objective_scores must be a mapping.")
    values: dict[str, float] = {}
    for name, value in raw.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"Objective score {name!r} must be finite numeric.")
        values[str(name)] = float(value)
    return values


def _row_dominance(
    existing: ParetoRow, candidate: ParetoRow, *, mode: str
) -> str | None:
    """Compare instance/objective/hybrid/cartesian frontier coordinates."""
    if mode not in {"instance", "objective", "hybrid", "cartesian"}:
        raise ValueError(
            "front mode must be instance, objective, hybrid, or cartesian."
        )
    left = _row_coordinates(existing, mode)
    right = _row_coordinates(candidate, mode)
    if left is None or right is None or set(left) != set(right):
        return None
    comparison = _dominance(left, right)
    if comparison == "existing":
        return "existing"
    if comparison == "candidate":
        return "candidate"
    return None


def _row_coordinates(row: ParetoRow, mode: str) -> dict[str, float] | None:
    coordinates: dict[str, float] = {}
    if mode in {"instance", "hybrid"}:
        coordinates.update(
            {f"case:{case_id}": score for case_id, score in row.per_case_scores.items()}
        )
    if mode in {"objective", "hybrid"}:
        if not row.objective_scores:
            return None
        coordinates.update(
            {f"objective:{name}": score for name, score in row.objective_scores.items()}
        )
    if mode == "cartesian":
        for case_id, objectives in row.per_case_objective_scores.items():
            for name, score in objectives.items():
                coordinates[f"case:{case_id}:objective:{name}"] = score
        if not coordinates:
            return None
    return coordinates or None

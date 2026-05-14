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
import random
import subprocess
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
        }
        if self.extra:
            out["extra"] = dict(self.extra)
        return out

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ParetoRow:
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
            extra=dict(data.get("extra", {})),
        )


def new_candidate_id() -> str:
    """Return a short stable identifier suitable for a candidate or proposal."""
    return uuid.uuid4().hex[:12]


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
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def iter_rows(self) -> list[ParetoRow]:
        if not self._path.exists():
            return []
        rows: list[ParetoRow] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(ParetoRow.from_dict(json.loads(stripped)))
        return rows

    def front(self) -> list[ParetoRow]:
        """Return rows that are Pareto-dominant across per-case scores.

        A row dominates another if its per-case scores are >= the other's on
        every shared case and strictly > on at least one. Rows without any
        overlapping cases are kept (they're incomparable). Higher score = better.
        """
        rows = self.iter_rows()
        if not rows:
            return []

        front: list[ParetoRow] = []
        for candidate in rows:
            dominated = False
            front_after: list[ParetoRow] = []
            for existing in front:
                cmp = _dominance(existing.per_case_scores, candidate.per_case_scores)
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

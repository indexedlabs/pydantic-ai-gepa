"""Harness-private validation front and replay-stable parent sampling."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random
import re
import shlex
import os
import subprocess
import tempfile
from typing import Any

import typer

from ..gepa_graph.evaluation.pareto import remove_dominated_programs
from .eval import EvalOutcome
from .layout import components_dir, gepa_dir, run_dir
from .safe_git import Repository, pin_commit
from .validation import (
    check_heldout_pin,
    read_validation_evidence,
    validation_evidence_path,
)


def _read_front(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Resume fronts written before they had a dedicated storage format.
    if set(data) == {"identity", "scores"}:
        data = data["identity"]
    return data


def _write_front(path: Path, data: dict[str, Any]) -> None:
    """Atomically publish a private front, independently of paired evidence."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def _pin_git_candidate(root: Path, run_id: str, candidate_id: str, sha: str) -> None:
    """Retain validated commits even after a reflector resets its own branch."""
    if not all(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value)
        for value in (run_id, candidate_id)
    ) or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", sha):
        raise typer.BadParameter("Invalid public candidate identity for retention.")
    ref = f"refs/gepa/{run_id}/{candidate_id}"
    try:
        pin_commit(root, ref, sha)
    except (OSError, subprocess.CalledProcessError):
        raise typer.BadParameter(
            "Could not retain the validated candidate commit."
        ) from None


def parent_restore_command(state: Any, root: Path) -> str:
    """Describe a checkout change without making it inside a nomination."""
    if state.candidate_source == "git":
        return f"git reset --hard {state.next_parent_commit_sha}"
    path = (
        run_dir(state.run_id, root)
        / "parents"
        / f"{state.next_parent_candidate_id}.json"
    )
    command = (
        f"gepa --gepa-dir {shlex.quote(str(gepa_dir(root)))} "
        f"apply --candidate-file {shlex.quote(str(path))}"
    )
    try:
        tree = Repository.discover(root).root
    except OSError:
        return command
    if components_dir(root).resolve().is_relative_to(tree):
        command += " --commit"
    return command


def write_parent_packet(path: Path, state: Any, root: Path) -> None:
    """Add the pending parent action to an otherwise training-only packet."""
    if (
        not state.next_parent_candidate_id
        or state.status != "paused_after_candidate_eval"
    ):
        return
    packet = json.loads(path.read_text())
    packet["next_parent"] = {
        "candidate_id": state.next_parent_candidate_id,
        "commit_sha": state.next_parent_commit_sha,
        "restore_command": parent_restore_command(state, root),
    }
    packet["instructions"] = (
        f"Restore next parent {state.next_parent_candidate_id} with "
        f"{parent_restore_command(state, root)}, then run next_command to sample its training minibatch."
    )
    packet["discard_command"] = None
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(packet, handle, indent=2)
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def snapshot_components(outcome: EvalOutcome, root: Path) -> None:
    """Archive every evaluated component candidate, without selection evidence."""
    if outcome.summary.get("candidate_source") != "components":
        return
    from .candidates import Candidate, candidate_id_from_components
    from .layout import GepaConfig, config_path, resolve_agent, resolve_skills
    from .store import ComponentStore

    cfg = GepaConfig.load(config_path(root))
    components = ComponentStore(root).effective_candidate(
        resolve_agent(cfg), skills_fs=resolve_skills(cfg, root=root)
    )
    candidate_id = str(outcome.summary["candidate_id"])
    if candidate_id_from_components(components) != candidate_id:
        raise ValueError("Component candidate changed before its parent snapshot.")
    path = (
        run_dir(str(outcome.summary["run_id"]), root)
        / "parents"
        / f"{candidate_id}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Candidate(id=candidate_id, components=components).write(path)


class ValidationFront:
    """Keep scores and sampling history beside the harness's pinned dataset.

    Only ``select``'s candidate identity may be published to the reflector.
    The controller serializes updates under the existing run lock.
    """

    def __init__(self, root: Path, run_id: str):
        self.root = root
        self.run_id = run_id
        self.dataset, digest = check_heldout_pin(root, run_id)
        self.key = f"{run_id}:parent-front"
        self.path = validation_evidence_path(
            self.dataset, project_root=root, run_id=self.key
        )
        try:
            self.data = _read_front(self.path)
        except FileNotFoundError:
            self.data = {"digest": digest, "evaluations": {}, "selections": {}}
        except OSError:
            raise typer.BadParameter(
                "Private parent front storage must be readable by the harness."
            ) from None
        if self.data["digest"] != digest:
            raise ValueError("Private parent front does not match the pinned dataset.")

    def _save(self) -> None:
        try:
            _write_front(self.path, self.data)
        except OSError:
            raise typer.BadParameter(
                "Private parent front storage must be writable by the harness."
            ) from None

    def preflight(self) -> None:
        """Verify the real write/replace operation before paying for validation."""
        self._save()

    def record(self, outcome: EvalOutcome) -> None:
        summary = outcome.summary
        if (
            summary.get("selectable") is False
            or summary.get("evaluation_outcome", "valid") != "valid"
        ):
            return
        eval_id = str(summary["eval_id"])
        if eval_id in self.data["evaluations"]:
            if summary.get("candidate_source") == "git":
                _pin_git_candidate(
                    self.root,
                    self.run_id,
                    str(summary["candidate_id"]),
                    str(summary.get("commit_sha")),
                )
            return
        candidate = str(summary["candidate_id"])
        scores = {record.case_id: record.score for record in outcome.records}
        if not scores:
            scores = read_validation_evidence(
                self.dataset,
                project_root=self.root,
                run_id=f"{self.run_id}:eval:{eval_id}",
                identity={"candidate_id": candidate, "eval_id": eval_id},
            )
        if not scores or not all(math.isfinite(score) for score in scores.values()):
            return
        evaluations = self.data["evaluations"]
        if evaluations and set(scores) != set(
            next(iter(evaluations.values()))["scores"]
        ):
            return
        if summary.get("candidate_source") == "git":
            _pin_git_candidate(
                self.root, self.run_id, candidate, str(summary.get("commit_sha"))
            )
        evaluations[eval_id] = {
            "candidate_id": candidate,
            "commit_sha": summary.get("commit_sha"),
            "scores": scores,
        }
        self._save()

    def weights(self) -> dict[str, int]:
        """Compute private frequencies from mean per-case repeated measurements."""
        samples: dict[str, list[dict[str, float]]] = {}
        for evaluation in self.data["evaluations"].values():
            samples.setdefault(evaluation["candidate_id"], []).append(
                evaluation["scores"]
            )
        if not samples:
            return {}
        scores = {
            candidate: {
                case: sum(sample[case] for sample in values) / len(values)
                for case in values[0]
            }
            for candidate, values in samples.items()
        }
        fronts = {}
        for case in next(iter(scores.values())):
            best = max(values[case] for values in scores.values())
            fronts[case] = {
                candidate
                for candidate, values in scores.items()
                if values[case] == best
            }
        fronts = remove_dominated_programs(
            fronts,
            {
                candidate: sum(values.values()) / len(values)
                for candidate, values in scores.items()
            },
        )
        return dict(
            Counter(
                candidate
                for winners in fronts.values()
                for candidate in sorted(winners)
            )
        )

    def retire_replay(self, outcome: EvalOutcome) -> None:
        """The front now holds nonpaired replay scores; discard the temporary copy."""
        eval_id = str(outcome.summary["eval_id"])
        if eval_id in self.data["evaluations"]:
            validation_evidence_path(
                self.dataset,
                project_root=self.root,
                run_id=f"{self.run_id}:eval:{eval_id}",
            ).unlink(missing_ok=True)

    def select(self, *, seed: int, round_id: int) -> dict[str, Any] | None:
        """Return only an identity, memoizing each controller transition on disk."""
        selections = self.data["selections"]
        key = f"{seed}:{round_id}"
        if key in selections:
            return dict(selections[key])
        weights = self.weights()
        if not weights:
            return None
        # Reconstruct the seeded stream without putting its state or weights in
        # the public run state. Retrying a transition reuses its saved draw.
        rng = random.Random(seed)
        for _ in selections:
            rng.random()
        candidates = list(
            dict.fromkeys(
                value["candidate_id"]
                for value in self.data["evaluations"].values()
                if value["candidate_id"] in weights
            )
        )
        candidate = rng.choices(
            candidates, weights=[weights[c] for c in candidates], k=1
        )[0]
        evaluation = next(
            value
            for value in self.data["evaluations"].values()
            if value["candidate_id"] == candidate
        )
        selected = {"candidate_id": candidate, "commit_sha": evaluation["commit_sha"]}
        selections[key] = selected
        self._save()
        return dict(selected)

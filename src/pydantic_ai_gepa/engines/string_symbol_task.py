"""An optimization task whose candidate is a Git commit's declared string constants.

``StringSymbolTask`` lets the in-process text engines (``gepa``,
``coding_agent``) and the composition helpers, including ``optimize_omni``,
optimize a Git candidate: the seed is read from ``base_commit`` with
:func:`~pydantic_ai_gepa.string_symbols.candidate_from_commit`, every
candidate is materialized as a verified commit on top of that base, and a
caller-supplied evaluator scores the commit.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from pydantic_evals import Case

from .._validation import validation_active
from ..adapter import SharedReflectiveDataset
from ..adapters.agent_adapter import _serialize_for_reflection
from ..evaluation import EvaluationRecord
from ..evaluation_models import EvaluationBatch
from ..gepa_graph.datasets import DatasetInput
from ..gepa_graph.models import CandidateMap
from ..string_symbols import (
    candidate_from_commit,
    commit_from_candidate,
    normalize_string_symbols,
    resolve_commit,
)
from ..types import MetricResult, RolloutOutput
from .base import SELECTION_SIDE_INFO_KEYS, OptimizationTask

if TYPE_CHECKING:
    from ..gepa_graph.example_bank import InMemoryExampleBank

CommitEvaluator: TypeAlias = Callable[
    [str, Sequence[Case[Any, Any, Any]]], Awaitable[Sequence[MetricResult]]
]
"""Score a commit SHA on cases, returning one ``MetricResult`` per case, in order.

Return a low score with feedback for a failed case; an exception aborts the
evaluation. ``side_info['scores']`` feeds objective selection as usual.
"""


def _agentless_metric(*_: Any) -> MetricResult:
    raise TypeError("StringSymbolTask scores commits through its evaluator.")


class StringSymbolTask(OptimizationTask):
    """Optimize declared Python string constants of a Git commit.

    ``symbols`` uses the ``[acceptance.python_string_symbols]`` shape, with
    paths relative to ``repo``. Each candidate's component texts are written
    as one commit on top of ``base_commit`` (memoized per candidate; the seed
    is ``base_commit`` itself) and passed to ``evaluator``. Use
    :meth:`commit_for` to materialize a result's ``best_candidate``.
    """

    def __init__(
        self,
        *,
        repo: str | Path,
        base_commit: str,
        symbols: Mapping[str, Sequence[str]],
        evaluator: CommitEvaluator,
        trainset: DatasetInput,
        valset: DatasetInput | None = None,
        test_set: DatasetInput | None = None,
        concurrency: int = 20,
        evaluation_cache_identity: str | None = None,
        commit_message: str = "Try a pydantic-ai-gepa string-symbol candidate",
    ) -> None:
        """Resolve ``base_commit`` and read its seed constants eagerly."""
        super().__init__(
            agent=cast(Any, None),
            trainset=trainset,
            metric=_agentless_metric,
            valset=valset,
            test_set=test_set,
            concurrency=concurrency,
            evaluation_cache_identity=evaluation_cache_identity,
        )
        self.repo = Path(repo)
        self.symbols = normalize_string_symbols(symbols)
        self.evaluator = evaluator
        self.commit_message = commit_message
        self.base_commit = resolve_commit(self.repo, base_commit)
        self._seed = candidate_from_commit(self.repo, self.base_commit, self.symbols)
        self._commits: dict[tuple[tuple[str, str], ...], str] = {}
        self._commit_lock = asyncio.Lock()

    async def seed_candidate(self) -> CandidateMap:
        """Return the base commit's declared constants."""
        return {name: value.model_copy() for name, value in self._seed.items()}

    async def commit_for(self, candidate: CandidateMap) -> str:
        """Return the verified commit holding ``candidate``'s texts."""
        key = tuple(sorted((name, value.text) for name, value in candidate.items()))
        async with self._commit_lock:
            sha = self._commits.get(key)
            if sha is None:
                sha = await asyncio.to_thread(
                    commit_from_candidate,
                    self.repo,
                    self.base_commit,
                    self.symbols,
                    candidate,
                    self.commit_message,
                )
                self._commits[key] = sha
        return sha

    def create_adapter(self) -> StringSymbolAdapter:
        """Return the GEPA graph adapter that scores candidates as commits."""
        return StringSymbolAdapter(self)

    async def score_commit(
        self, commit: str, cases: Sequence[Case[Any, Any, Any]]
    ) -> list[MetricResult]:
        """Run the evaluator and check it returned one MetricResult per case."""
        results = list(await self.evaluator(commit, cases))
        if len(results) != len(cases) or not all(
            isinstance(result, MetricResult)
            and not isinstance(result.score, bool)
            and isinstance(result.score, (int, float))
            and math.isfinite(result.score)
            for result in results
        ):
            raise ValueError(
                "A CommitEvaluator must return one MetricResult per case, in "
                "order, each with a finite numeric score."
            )
        return results

    async def _evaluate_cases(
        self,
        candidate: CandidateMap,
        cases: Sequence[Case[Any, Any, Any]],
        *,
        capture_traces: bool,
    ) -> tuple[list[EvaluationRecord], list[EvaluationRecord]]:
        sha = await self.commit_for(candidate)
        results = await self.score_commit(sha, cases)
        withheld = validation_active()
        records: list[EvaluationRecord] = []
        selection: list[EvaluationRecord] = []
        for index, (case, result) in enumerate(zip(cases, results)):
            case_id = case.name or f"case-{index}"
            info = result.side_info or {}
            if withheld:
                records.append(
                    EvaluationRecord(
                        case_id=case_id,
                        score=result.score,
                        feedback=None,
                        payload={"commit": sha},
                    )
                )
                selection.append(
                    EvaluationRecord(
                        case_id=case_id,
                        score=result.score,
                        feedback=None,
                        payload={
                            "side_info": {
                                key: info[key]
                                for key in SELECTION_SIDE_INFO_KEYS
                                if key in info
                            }
                        },
                    )
                )
                continue
            payload: dict[str, Any] = {"commit": sha}
            if result.side_info is not None:
                payload["side_info"] = result.side_info
            record = EvaluationRecord(
                case_id=case_id,
                score=result.score,
                feedback=result.feedback,
                payload=payload,
            )
            records.append(record)
            selection.append(record)
        return records, selection


@dataclass
class StringSymbolTrajectory:
    """Reflection evidence for one case scored on a candidate commit."""

    case: Case[Any, Any, Any]
    commit: str
    metric_feedback: str | None = None
    metric_side_info: dict[str, Any] | None = None
    instructions: str | None = None
    final_output: Any | None = field(default=None)

    def to_reflective_record(self) -> dict[str, Any]:
        return {
            "case": self.case.name,
            "inputs": _serialize_for_reflection(self.case.inputs),
            "commit": self.commit,
        }


class StringSymbolAdapter:
    """``Adapter`` implementation that scores CandidateMaps as Git commits."""

    trace_collector = None

    def __init__(self, task: StringSymbolTask) -> None:
        self.task = task
        self.spend_meter: Any = None

    def close(self) -> None:
        """Nothing to release; commits are retained in the repository."""

    async def evaluate(
        self,
        batch: Sequence[Case[Any, Any, Any]],
        candidate: CandidateMap,
        capture_traces: bool,
        example_bank: InMemoryExampleBank | None = None,
    ) -> EvaluationBatch:
        sha = await self.task.commit_for(candidate)
        results = await self.task.score_commit(sha, batch)
        # Withheld validation rollouts never reach reflection.
        traced = capture_traces and not validation_active()
        return EvaluationBatch(
            outputs=[RolloutOutput.from_success(sha) for _ in results],
            scores=[result.score for result in results],
            trajectories=[
                StringSymbolTrajectory(
                    case=case,
                    commit=sha,
                    metric_feedback=result.feedback,
                    metric_side_info=result.side_info,
                )
                for case, result in zip(batch, results)
            ]
            if traced
            else None,
        )

    def make_reflective_dataset(
        self,
        *,
        candidate: CandidateMap,
        eval_batch: EvaluationBatch,
        components_to_update: Sequence[str],
        include_case_metadata: bool = False,
        include_expected_output: bool = False,
    ) -> SharedReflectiveDataset:
        records: list[dict[str, Any]] = []
        for trajectory, output, score in zip(
            eval_batch.trajectories or (), eval_batch.outputs, eval_batch.scores
        ):
            if not isinstance(trajectory, StringSymbolTrajectory):
                continue
            record = trajectory.to_reflective_record()
            record["score"] = score
            record["success"] = output.success
            record["feedback"] = trajectory.metric_feedback or f"Score: {score:.2f}"
            if trajectory.metric_side_info:
                record["side_info"] = _serialize_for_reflection(
                    trajectory.metric_side_info
                )
            case = trajectory.case
            if include_case_metadata and case.metadata:
                record["case_metadata"] = _serialize_for_reflection(case.metadata)
            if include_expected_output and case.expected_output is not None:
                record["expected_output"] = _serialize_for_reflection(
                    case.expected_output
                )
            records.append(record)
        return SharedReflectiveDataset(records=records)

    def get_components(self) -> CandidateMap:
        return {name: value.model_copy() for name, value in self.task._seed.items()}


__all__ = [
    "CommitEvaluator",
    "StringSymbolAdapter",
    "StringSymbolTask",
    "StringSymbolTrajectory",
]

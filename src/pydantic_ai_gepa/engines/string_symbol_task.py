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
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from pydantic_evals import Case

from .._concurrency import gather_cancelling_on_provider_stop
from .._validation import validation_active
from ..adapter import SharedReflectiveDataset
from ..adapters.agent_adapter import _serialize_for_reflection
from ..evaluation import EvaluationRecord, _rollout_scope
from ..exceptions import UsageBudgetExceeded
from ..evaluation_models import EvaluationBatch
from ..gepa_graph.datasets import DatasetInput
from ..gepa_graph.models import CandidateMap
from ..string_symbols import (
    candidate_from_commit,
    commit_from_candidate,
    normalize_string_symbols,
    resolve_commit,
)
from ..provider_errors import is_provider_stop_error
from ..spend import SpendMeter, current_rollout_capability, rollout_spend
from ..types import MetricResult, RolloutOutput
from .base import SELECTION_SIDE_INFO_KEYS, OptimizationTask

if TYPE_CHECKING:
    from ..gepa_graph.example_bank import InMemoryExampleBank

CommitEvaluator: TypeAlias = Callable[
    [str, Case[Any, Any, Any]], Awaitable[MetricResult]
]
"""Score one case on a commit SHA: ``async (commit, case) -> MetricResult``.

Engines call it once per case, and calls overlap, often on the same SHA, so it
must be reentrant: materialize each commit once (for example a per-SHA worktree
created under an ``asyncio.Lock``) or per call, never at a shared fixed path.
Attach ``current_rollout_capability()`` to model calls so they are metered.
``side_info['scores']`` feeds objective selection as usual.
"""


@dataclass(frozen=True, slots=True)
class CommitScore:
    """One case's evaluator result and the rollout output recorded for it."""

    result: MetricResult
    output: RolloutOutput[Any]


def _agentless_metric(*_: Any) -> MetricResult:
    raise TypeError("StringSymbolTask scores commits through its evaluator.")


_FAILED_CASE_SIDE_INFO = {"selectable": False, "infrastructure_valid": False}


class _CommitScorer:
    """Commit materialization and scoring, with no access to any dataset.

    The graph adapter holds only this, so an engine holding the adapter cannot
    reach the task's held-out ``test_set`` around the trusted task view.
    """

    def __init__(
        self,
        *,
        repo: Path,
        base_commit: str,
        symbols: Mapping[str, Sequence[str]],
        evaluator: CommitEvaluator,
        concurrency: int,
        commit_message: str,
    ) -> None:
        self.repo = repo
        self.base_commit = base_commit
        self.symbols = symbols
        self.evaluator = evaluator
        self.concurrency = concurrency
        self.commit_message = commit_message
        self.seed = candidate_from_commit(repo, base_commit, symbols)
        self._commits: dict[tuple[tuple[str, str], ...], str] = {}
        self._commit_lock = asyncio.Lock()

    async def commit_for(self, candidate: CandidateMap) -> str:
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

    async def score_commit(
        self,
        commit: str,
        cases: Sequence[Case[Any, Any, Any]],
        *,
        meter: SpendMeter | None = None,
    ) -> list[CommitScore]:
        active = current_rollout_capability()
        meter = meter or (active.meter if active is not None else None)
        semaphore = asyncio.Semaphore(max(1, self.concurrency))

        async def score(case: Case[Any, Any, Any]) -> CommitScore:
            async with semaphore:
                with rollout_spend(meter) if meter is not None else nullcontext():
                    async with _rollout_scope():
                        try:
                            result = await self.evaluator(commit, case)
                        except Exception as error:
                            if isinstance(
                                error, UsageBudgetExceeded
                            ) or is_provider_stop_error(error):
                                raise
                            # An infrastructure failure, not a measured 0.0:
                            # the evaluation becomes non-selectable.
                            return CommitScore(
                                MetricResult(
                                    score=0.0,
                                    feedback=f"Commit evaluator failed: {error}",
                                    side_info=dict(_FAILED_CASE_SIDE_INFO),
                                ),
                                RolloutOutput.from_error(error, kind="system"),
                            )
            if not (
                isinstance(result, MetricResult)
                and not isinstance(result.score, bool)
                and isinstance(result.score, (int, float))
                and math.isfinite(result.score)
            ):
                raise ValueError(
                    "A CommitEvaluator must return a MetricResult with a finite "
                    "numeric score."
                )
            return CommitScore(result, RolloutOutput.from_success(commit))

        return list(
            await gather_cancelling_on_provider_stop(*(score(case) for case in cases))
        )


class StringSymbolTask(OptimizationTask):
    """Optimize declared Python string constants of a Git commit.

    ``symbols`` uses the ``[acceptance.python_string_symbols]`` shape, with
    paths relative to ``repo``. Each candidate's component texts are written
    as one commit on top of ``base_commit`` (memoized per candidate; the seed
    is ``base_commit`` itself) and passed to ``evaluator``. Use
    :meth:`commit_for` to materialize a result's ``best_candidate``.
    Proposals must keep to the declared component keys; an undeclared key
    raises :class:`~pydantic_ai_gepa.string_symbols.StringSymbolError`.

    Candidate commits are not referenced by any ref, so ``git gc`` eventually
    prunes them; point a ref at the commit you keep
    (``git update-ref refs/heads/<name> <sha>``).
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
        self.base_commit = resolve_commit(self.repo, base_commit)
        self._scorer = _CommitScorer(
            repo=self.repo,
            base_commit=self.base_commit,
            symbols=self.symbols,
            evaluator=evaluator,
            concurrency=concurrency,
            commit_message=commit_message,
        )

    async def seed_candidate(self) -> CandidateMap:
        """Return the base commit's declared constants."""
        return {name: value.model_copy() for name, value in self._scorer.seed.items()}

    async def commit_for(self, candidate: CandidateMap) -> str:
        """Return the verified commit holding ``candidate``'s texts."""
        return await self._scorer.commit_for(candidate)

    def create_adapter(self) -> StringSymbolAdapter:
        """Return the GEPA graph adapter that scores candidates as commits."""
        return StringSymbolAdapter(self._scorer)

    async def score_commit(
        self,
        commit: str,
        cases: Sequence[Case[Any, Any, Any]],
        *,
        meter: SpendMeter | None = None,
    ) -> list[CommitScore]:
        """Score ``commit`` on each case, one evaluator call per case.

        Calls run concurrently (up to ``concurrency``), each as one metered
        rollout: ``current_rollout_capability()`` inside the evaluator returns
        ``meter`` or, when it is None, the enclosing evaluation's meter. An
        evaluator exception scores that case 0.0 with a failed output and
        ``side_info`` marking it non-selectable, so the evaluation cannot be
        selected, as the CLI treats a failed rollout. Budget and provider
        stops propagate.
        """
        return await self._scorer.score_commit(commit, cases, meter=meter)

    async def _evaluate_cases(
        self,
        candidate: CandidateMap,
        cases: Sequence[Case[Any, Any, Any]],
        *,
        capture_traces: bool,
    ) -> tuple[list[EvaluationRecord], list[EvaluationRecord]]:
        sha = await self.commit_for(candidate)
        scored = await self.score_commit(sha, cases)
        withheld = validation_active()
        records: list[EvaluationRecord] = []
        selection: list[EvaluationRecord] = []
        for index, (case, item) in enumerate(zip(cases, scored)):
            result = item.result
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
            payload: dict[str, Any] = {"commit": sha, "output": item.output}
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

    def __init__(self, scorer: _CommitScorer) -> None:
        self._scorer = scorer
        self.spend_meter: Any = None

    def close(self) -> None:
        """Nothing to release; candidate commits stay in the object store."""

    async def evaluate(
        self,
        batch: Sequence[Case[Any, Any, Any]],
        candidate: CandidateMap,
        capture_traces: bool,
        example_bank: InMemoryExampleBank | None = None,
    ) -> EvaluationBatch:
        sha = await self._scorer.commit_for(candidate)
        scored = await self._scorer.score_commit(sha, batch, meter=self.spend_meter)
        # Withheld validation rollouts never reach reflection.
        traced = capture_traces and not validation_active()
        return EvaluationBatch(
            outputs=[item.output for item in scored],
            scores=[item.result.score for item in scored],
            trajectories=[
                StringSymbolTrajectory(
                    case=case,
                    commit=sha,
                    metric_feedback=item.result.feedback,
                    metric_side_info=item.result.side_info,
                )
                for case, item in zip(batch, scored)
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
        return {name: value.model_copy() for name, value in self._scorer.seed.items()}


__all__ = [
    "CommitEvaluator",
    "CommitScore",
    "StringSymbolAdapter",
    "StringSymbolTask",
    "StringSymbolTrajectory",
]

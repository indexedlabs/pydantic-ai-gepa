"""Shared contracts and evaluation support for optimization engines."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.agent import AbstractAgent
from pydantic_evals import Case

from ..adapters.agent_adapter import CaseFactory
from ..components import extract_seed_candidate_with_input_type
from ..evaluation import EvaluationRecord, evaluate_candidate_dataset
from ..gepa_graph.datasets import DataLoader, DatasetInput, resolve_dataset
from ..gepa_graph.models import CandidateMap
from ..input_type import InputSpec
from ..skills import SkillsFS
from ..skills.models import SkillCapability
from ..types import MetricResult, RolloutOutput

Metric = Callable[
    [Case[Any, Any, Any], RolloutOutput[Any]],
    MetricResult | Awaitable[MetricResult],
]


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Result of evaluating one candidate on an optimization task's valset."""

    score: float
    records: list[EvaluationRecord]
    side_info: dict[str, Any]
    num_cases: int
    objective_scores: dict[str, float] = field(default_factory=dict)
    per_case_objective_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    selectable: bool = True
    per_case_scores: dict[str, float] = field(default_factory=dict)


class OptimizationTask:
    """The shared dataset evaluator and seed source used by optimization engines."""

    def __init__(
        self,
        *,
        agent: AbstractAgent[Any, Any],
        trainset: DatasetInput,
        metric: Metric,
        valset: DatasetInput | None = None,
        input_type: InputSpec[BaseModel] | None = None,
        skills_fs: SkillsFS | None = None,
        skills_capabilities: set[SkillCapability] | None = None,
        case_factory: CaseFactory | None = None,
        concurrency: int = 20,
        test_set: DatasetInput | None = None,
        evaluation_cache_identity: str | None = None,
    ) -> None:
        """Create an optimization task without eagerly materializing datasets."""
        if concurrency < 1:
            raise ValueError("concurrency must be greater than zero.")

        self.agent = agent
        self.trainset = trainset
        self.metric = metric
        self.valset = valset
        self.test_set = test_set
        self.input_type = input_type
        self.skills_fs = skills_fs
        self.skills_capabilities = skills_capabilities
        self.case_factory = case_factory
        self.concurrency = concurrency
        self.evaluation_cache_identity = evaluation_cache_identity

        self._train_loader: DataLoader[Any, Case[Any, Any, Any]] | None = None
        self._val_loader: DataLoader[Any, Case[Any, Any, Any]] | None = None
        self._val_cases: list[Case[Any, Any, Any]] | None = None
        self._seed_candidate: CandidateMap | None = None
        self._train_loader_lock = asyncio.Lock()
        self._val_loader_lock = asyncio.Lock()
        self._val_cases_lock = asyncio.Lock()
        self._test_loader: DataLoader[Any, Case[Any, Any, Any]] | None = None
        self._test_cases: list[Case[Any, Any, Any]] | None = None
        self._test_loader_lock = asyncio.Lock()
        self._test_cases_lock = asyncio.Lock()
        self._evaluation_cache: dict[str, CandidateEvaluation] = {}
        self._evaluation_cache_lock = asyncio.Lock()
        self._seed_candidate_lock = asyncio.Lock()

    async def train_loader(self) -> DataLoader[Any, Case[Any, Any, Any]]:
        """Resolve and memoize the training dataset loader."""
        if self._train_loader is None:
            async with self._train_loader_lock:
                if self._train_loader is None:
                    self._train_loader = await resolve_dataset(
                        self.trainset, name="trainset"
                    )
        return self._train_loader

    async def val_loader(self) -> DataLoader[Any, Case[Any, Any, Any]]:
        """Resolve and memoize the validation dataset loader."""
        if self.valset is None:
            return await self.train_loader()

        if self._val_loader is None:
            async with self._val_loader_lock:
                if self._val_loader is None:
                    self._val_loader = await resolve_dataset(self.valset, name="valset")
        return self._val_loader

    async def seed_candidate(self) -> CandidateMap:
        """Extract and memoize the task agent's initial editable candidate."""
        if self._seed_candidate is None:
            async with self._seed_candidate_lock:
                if self._seed_candidate is None:
                    self._seed_candidate = extract_seed_candidate_with_input_type(
                        self.agent,
                        input_type=self.input_type,
                        optimize_output_type=False,
                    )
        return self._seed_candidate

    async def evaluate(
        self,
        candidate: CandidateMap,
        *,
        budget: BudgetTracker | None = None,
        capture_traces: bool = False,
        dataset: str = "validation",
        cache: bool = False,
    ) -> CandidateEvaluation:
        """Evaluate a candidate through the shared eval server.

        ``dataset='test'`` is deliberately reporting-only: engines only receive
        this task API and composition never invokes it until a final report.
        Caching is opt-in and requires a caller supplied deterministic evaluator
        identity; stochastic evaluators must include their seed/control in it.
        """
        if dataset not in {"validation", "test"}:
            raise ValueError("dataset must be 'validation' or 'test'.")
        cases = await (
            self._validation_cases()
            if dataset == "validation"
            else self._test_cases_for_reporting()
        )
        cache_key = self._cache_key(candidate, dataset) if cache else None
        if cache_key is not None:
            async with self._evaluation_cache_lock:
                cached = self._evaluation_cache.get(cache_key)
            if cached is not None:
                return cached

        # Charge before making a call. This closes the evaluate-then-spend race
        # and means an exhausted budget cannot trigger an invisible evaluator call.
        if budget is not None:
            budget.spend(len(cases))
        records = await evaluate_candidate_dataset(
            agent=self.agent,
            metric=self.metric,
            dataset=cases,
            candidate=candidate,
            concurrency=self.concurrency,
            input_type=self.input_type,
            case_factory=self.case_factory,
            skills_fs=self.skills_fs,
            skills_capabilities=self.skills_capabilities,
            capture_traces=capture_traces,
        )
        score = (
            sum(record.score for record in records) / len(records) if records else 0.0
        )
        objectives, per_case_objectives, selectable = _objective_scores(records)
        result = CandidateEvaluation(
            score=score,
            records=records,
            side_info=_aggregate_side_info(records),
            num_cases=len(records),
            objective_scores=objectives,
            per_case_objective_scores=per_case_objectives,
            selectable=selectable,
            per_case_scores={record.case_id: record.score for record in records},
        )
        if cache_key is not None:
            async with self._evaluation_cache_lock:
                self._evaluation_cache.setdefault(cache_key, result)
        return result

    async def _validation_cases(self) -> Sequence[Case[Any, Any, Any]]:
        """Materialize and memoize the validation loader's cases for evaluation."""
        if self._val_cases is None:
            async with self._val_cases_lock:
                if self._val_cases is None:
                    loader = await self.val_loader()
                    self._val_cases = await loader.fetch(await loader.all_ids())
        return self._val_cases

    async def _test_cases_for_reporting(self) -> Sequence[Case[Any, Any, Any]]:
        """Materialize the optional held-out reporting set, never used by engines."""
        if self.test_set is None:
            raise ValueError("This task has no test_set.")
        if self._test_cases is None:
            async with self._test_cases_lock:
                if self._test_cases is None:
                    if self._test_loader is None:
                        async with self._test_loader_lock:
                            if self._test_loader is None:
                                self._test_loader = await resolve_dataset(
                                    self.test_set, name="test_set"
                                )
                    self._test_cases = await self._test_loader.fetch(
                        await self._test_loader.all_ids()
                    )
        return self._test_cases

    def _cache_key(self, candidate: CandidateMap, dataset: str) -> str | None:
        if self.evaluation_cache_identity is None:
            raise ValueError(
                "Evaluation caching requires evaluation_cache_identity; include a deterministic evaluator version and seed."
            )
        payload = json.dumps(
            {
                "identity": self.evaluation_cache_identity,
                "dataset": dataset,
                "cases": [
                    {
                        "id": case.name,
                        "inputs": repr(case.inputs),
                        "expected_output": repr(case.expected_output),
                        "metadata": repr(case.metadata),
                    }
                    for case in (
                        self._val_cases if dataset == "validation" else self._test_cases
                    )
                    or []
                ],
                "candidate": {
                    name: value.model_dump(mode="json")
                    for name, value in sorted(candidate.items())
                },
            },
            sort_keys=True,
            default=str,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _aggregate_side_info(records: Sequence[EvaluationRecord]) -> dict[str, Any]:
    """Collect optional per-case side information and failing feedback."""
    case_side_info: list[dict[str, Any]] = []
    feedback: list[str] = []
    for record in records:
        raw_side_info = record.payload.get("side_info")
        if isinstance(raw_side_info, dict):
            case_side_info.append(
                {"case_id": record.case_id, "side_info": dict(raw_side_info)}
            )
        if record.score < 1.0 and record.feedback:
            feedback.append(record.feedback)

    side_info: dict[str, Any] = {}
    if case_side_info:
        side_info["cases"] = case_side_info
    if feedback:
        side_info["feedback"] = feedback
    return side_info


def _objective_scores(
    records: Sequence[EvaluationRecord],
) -> tuple[dict[str, float], dict[str, dict[str, float]], bool]:
    """Extract finite higher-is-better objectives from ``side_info['scores']``.

    The regular scalar score remains the compatibility/reporting score.  An
    invalid objective is a caller error rather than silently becoming a NaN
    frontier coordinate.  A metric can mark a result non-selectable with
    ``side_info['selectable'] = False`` (for infrastructure-invalid outcomes).
    """
    values: dict[str, list[float]] = {}
    per_case: dict[str, dict[str, float]] = {}
    selectable = True
    objective_key_sets: list[set[str] | None] = []
    for record in records:
        info = record.payload.get("side_info")
        if not isinstance(info, dict):
            info = record.payload.get("metric_side_info")
        if not isinstance(info, dict):
            info = getattr(record.payload.get("trajectory"), "metric_side_info", None)
        if not isinstance(info, dict):
            objective_key_sets.append(None)
            continue
        if info.get("selectable") is False or info.get("infrastructure_valid") is False:
            selectable = False
        raw_scores = info.get("scores")
        if raw_scores is None:
            objective_key_sets.append(None)
            continue
        if not isinstance(raw_scores, dict):
            raise ValueError(
                "MetricResult.side_info['scores'] must be a mapping of names to finite numbers."
            )
        case_scores: dict[str, float] = {}
        objective_key_sets.append({str(name) for name in raw_scores})
        for name, raw in raw_scores.items():
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
            ):
                raise ValueError(
                    f"Objective score {name!r} for case {record.case_id!r} must be finite numeric."
                )
            value = float(raw)
            key = str(name)
            values.setdefault(key, []).append(value)
            case_scores[key] = value
        if case_scores:
            per_case[record.case_id] = case_scores
    if any(keys is not None for keys in objective_key_sets):
        first = objective_key_sets[0]
        if first is None or any(keys != first for keys in objective_key_sets[1:]):
            raise ValueError(
                "MetricResult.side_info['scores'] must expose identical objective keys for every evaluated case."
            )
    return (
        {name: sum(scores) / len(scores) for name, scores in values.items()},
        per_case,
        selectable,
    )


class EngineConfig(BaseModel):
    """Configuration shared by all optimization engine implementations."""

    engine: str
    max_metric_calls: int = Field(default=200, gt=0)
    max_iterations: int | None = Field(default=None, gt=0)
    stop_at_score: float | None = None
    seed: int = 0
    engine_config: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class EngineEvent(BaseModel):
    """A progress or diagnostic event emitted by an optimization engine."""

    kind: str
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class EngineResult(BaseModel):
    """The best candidate and execution summary returned by an engine."""

    engine: str
    best_candidate: CandidateMap
    best_score: float
    num_metric_calls: int
    history: list[EngineEvent] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BudgetExhausted(Exception):
    """Raised when an evaluation would exceed the shared metric-call budget."""


class BudgetTracker:
    """Track metric calls consumed across one or more optimization engines."""

    def __init__(self, max_metric_calls: int) -> None:
        """Create a tracker with a positive total metric-call limit."""
        if max_metric_calls <= 0:
            raise ValueError("max_metric_calls must be greater than zero.")
        self.max_metric_calls = max_metric_calls
        self._spent = 0

    @property
    def spent(self) -> int:
        """Return the number of metric calls consumed so far."""
        return self._spent

    @property
    def remaining(self) -> int:
        """Return the number of metric calls still available."""
        return self.max_metric_calls - self._spent

    @property
    def exhausted(self) -> bool:
        """Return whether no metric calls remain."""
        return self._spent >= self.max_metric_calls

    def spend(self, n: int) -> None:
        """Record ``n`` metric calls or raise if doing so exceeds the budget."""
        if n < 0:
            raise ValueError("n must be greater than or equal to zero.")
        if self._spent + n > self.max_metric_calls:
            raise BudgetExhausted(
                "Metric-call budget exhausted: "
                f"requested {n} with {self.remaining} remaining."
            )
        self._spent += n

    def preflight(self, n: int) -> None:
        """Fail before an evaluator call when ``n`` calls are not affordable."""
        if n < 0:
            raise ValueError("n must be greater than or equal to zero.")
        if n > self.remaining:
            raise BudgetExhausted(
                f"Metric-call budget exhausted: requested {n} with {self.remaining} remaining."
            )

    def refund(self, n: int) -> None:
        """Return a pre-reserved amount that was not actually consumed."""
        if n < 0 or n > self._spent:
            raise ValueError("Cannot refund more metric calls than have been spent.")
        self._spent -= n

    def reserve_slice(self, n: int) -> "BudgetTracker":
        """Reserve an isolated child budget before concurrent work starts.

        The parent is charged immediately; callers release unused capacity with
        :meth:`release_slice` after the engine finishes. This makes concurrent
        composition deterministic and prevents two engines from racing for the
        final calls.
        """
        self.spend(n)
        return BudgetTracker(n)

    def release_slice(self, slice_budget: "BudgetTracker") -> None:
        """Return unused capacity from a completed child slice to this budget."""
        unused = slice_budget.remaining
        if unused:
            self._spent -= unused

    def check(self) -> None:
        """Raise if the shared metric-call budget has been exhausted."""
        if self.exhausted:
            raise BudgetExhausted("Metric-call budget exhausted.")


class OptimizationEngine(Protocol):
    """Protocol implemented by every pluggable optimization engine."""

    name: str

    async def run(
        self,
        task: OptimizationTask,
        config: EngineConfig,
        budget: BudgetTracker,
    ) -> EngineResult:
        """Run optimization against ``task`` under the shared ``budget``."""
        ...


__all__ = [
    "BudgetExhausted",
    "BudgetTracker",
    "CandidateEvaluation",
    "EngineConfig",
    "EngineEvent",
    "EngineResult",
    "OptimizationEngine",
    "OptimizationTask",
]

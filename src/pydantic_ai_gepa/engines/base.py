"""Shared contracts and evaluation support for optimization engines."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
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
    ) -> None:
        """Create an optimization task without eagerly materializing datasets."""
        if concurrency < 1:
            raise ValueError("concurrency must be greater than zero.")

        self.agent = agent
        self.trainset = trainset
        self.metric = metric
        self.valset = valset
        self.input_type = input_type
        self.skills_fs = skills_fs
        self.skills_capabilities = skills_capabilities
        self.case_factory = case_factory
        self.concurrency = concurrency

        self._train_loader: DataLoader[Any, Case[Any, Any, Any]] | None = None
        self._val_loader: DataLoader[Any, Case[Any, Any, Any]] | None = None
        self._val_cases: list[Case[Any, Any, Any]] | None = None
        self._seed_candidate: CandidateMap | None = None
        self._train_loader_lock = asyncio.Lock()
        self._val_loader_lock = asyncio.Lock()
        self._val_cases_lock = asyncio.Lock()
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
    ) -> CandidateEvaluation:
        """Evaluate ``candidate`` on the valset through the shared eval server."""
        records = await evaluate_candidate_dataset(
            agent=self.agent,
            metric=self.metric,
            dataset=await self._validation_cases(),
            candidate=candidate,
            concurrency=self.concurrency,
            input_type=self.input_type,
            case_factory=self.case_factory,
            skills_fs=self.skills_fs,
            skills_capabilities=self.skills_capabilities,
            capture_traces=capture_traces,
        )
        if budget is not None:
            budget.spend(len(records))

        score = (
            sum(record.score for record in records) / len(records) if records else 0.0
        )
        return CandidateEvaluation(
            score=score,
            records=records,
            side_info=_aggregate_side_info(records),
            num_cases=len(records),
        )

    async def _validation_cases(self) -> Sequence[Case[Any, Any, Any]]:
        """Materialize and memoize the validation loader's cases for evaluation."""
        if self._val_cases is None:
            async with self._val_cases_lock:
                if self._val_cases is None:
                    loader = await self.val_loader()
                    self._val_cases = await loader.fetch(await loader.all_ids())
        return self._val_cases


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

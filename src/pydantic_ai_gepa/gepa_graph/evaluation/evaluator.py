"""Parallel evaluation utilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Sequence, TypeVar
from ...evaluation_models import EvaluationBatch
from ...types import RolloutOutput, Trajectory
from pydantic_evals import Case
from ..models import CandidateMap, CandidateProgram
from ..._concurrency import gather_cancelling_on_provider_stop

if TYPE_CHECKING:
    from ...adapter import Adapter
    from ..example_bank import InMemoryExampleBank

DataIdT = TypeVar("DataIdT")


@dataclass(slots=True, kw_only=True)
class EvaluationResults(Generic[DataIdT]):
    """Container for flattened evaluation outputs."""

    data_ids: list[DataIdT]
    scores: list[float]
    outputs: list[RolloutOutput[Any]]
    trajectories: list[Trajectory | None] | None = None

    def __post_init__(self) -> None:
        lengths = {len(self.data_ids), len(self.scores), len(self.outputs)}
        if self.trajectories is not None:
            lengths.add(len(self.trajectories))
        if len(lengths) > 1:
            raise ValueError("EvaluationResults fields must have matching lengths.")

    def __len__(self) -> int:
        return len(self.data_ids)

    def __iter__(self):
        return iter(zip(self.data_ids, self.scores, self.outputs))

    def has_trajectories(self) -> bool:
        return bool(self.trajectories) and any(
            trace is not None for trace in self.trajectories
        )


class ParallelEvaluator:
    """Evaluate candidates on datasets with asyncio-powered parallelism."""

    async def evaluate_batch(
        self,
        *,
        candidate: CandidateProgram,
        batch: Sequence[Case[Any, Any, Any]],
        adapter: "Adapter[Any, Any, Any]",
        max_concurrent: int = 10,
        capture_traces: bool = False,
    ) -> EvaluationResults[str]:
        """Evaluate ``candidate`` for every instance in ``batch``."""
        if not batch:
            return EvaluationResults(
                data_ids=[],
                scores=[],
                outputs=[],
                trajectories=[] if capture_traces else None,
            )

        semaphore = asyncio.Semaphore(max(1, max_concurrent))
        candidate_payload = candidate.components
        example_bank = candidate.example_bank

        async def run_one(index: int, instance: Case[Any, Any, Any]):
            async with semaphore:
                eval_batch = await self._call_adapter(
                    adapter=adapter,
                    instance=instance,
                    candidate_payload=candidate_payload,
                    capture_traces=capture_traces,
                    example_bank=example_bank,
                )
            data_id = self._data_id(instance, index)
            return data_id, eval_batch

        tasks = [run_one(idx, instance) for idx, instance in enumerate(batch)]

        results = await gather_cancelling_on_provider_stop(*tasks)
        return self._merge_results(results, capture_traces=capture_traces)

    async def _call_adapter(
        self,
        *,
        adapter: "Adapter[Any, Any, Any]",
        instance: Case[Any, Any, Any],
        candidate_payload: CandidateMap,
        capture_traces: bool,
        example_bank: "InMemoryExampleBank | None" = None,
    ) -> EvaluationBatch:
        return await adapter.evaluate(
            [instance],
            candidate_payload,
            capture_traces,
            example_bank=example_bank,
        )

    def _merge_results(
        self,
        results: list[tuple[str, EvaluationBatch]],
        *,
        capture_traces: bool,
    ) -> EvaluationResults[str]:
        data_ids: list[str] = []
        scores: list[float] = []
        outputs: list[RolloutOutput[Any]] = []
        trajectories: list[Trajectory | None] | None = [] if capture_traces else None

        for data_id, batch in results:
            batch_scores = list(batch.scores)
            batch_outputs = list(batch.outputs)

            if len(batch_scores) != len(batch_outputs):
                raise ValueError("Adapter returned mismatched scores and outputs.")

            data_ids.extend([data_id] * len(batch_scores))
            scores.extend(batch_scores)
            outputs.extend(batch_outputs)

            if trajectories is not None:
                batch_traces = batch.trajectories
                if batch_traces is None:
                    trajectories.extend([None] * len(batch_scores))
                else:
                    batch_traces_list = list(batch_traces)
                    if len(batch_traces_list) != len(batch_scores):
                        raise ValueError("Adapter returned mismatched trace count.")
                    trajectories.extend(batch_traces_list)

        trajectory_payload: list[Trajectory | None] | None
        if trajectories is None:
            trajectory_payload = None
        else:
            trajectory_payload = trajectories

        return EvaluationResults(
            data_ids=data_ids,
            scores=scores,
            outputs=outputs,
            trajectories=trajectory_payload,
        )

    @staticmethod
    def _data_id(instance: Case[Any, Any, Any], index: int) -> str:
        return instance.name or f"case-{index}"


__all__ = ["EvaluationResults", "ParallelEvaluator"]

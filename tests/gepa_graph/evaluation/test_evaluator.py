"""Tests for the ParallelEvaluator helper."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_evals import Case

from pydantic_ai_gepa.gepa_graph.evaluation import EvaluationBatch, ParallelEvaluator
from pydantic_ai_gepa.gepa_graph.models import (
    CandidateMap,
    CandidateProgram,
    ComponentValue,
)
from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapterTrajectory
from pydantic_ai_gepa.adapter import SharedReflectiveDataset
from pydantic_ai_gepa.types import RolloutOutput


def _make_candidate() -> CandidateProgram:
    return CandidateProgram(
        idx=0,
        components={"system": ComponentValue(name="system", text="Hello")},
        creation_type="seed",
        discovered_at_iteration=0,
        discovered_at_evaluation=0,
    )


def _make_data_inst(case_id: str) -> Case[str, str, dict[str, str]]:
    return Case(name=case_id, inputs=f"prompt-{case_id}", metadata={})


class _RecordingAdapter:
    def __init__(self, delay: float = 0.01) -> None:
        self._delay = delay
        self.inflight = 0
        self.max_inflight = 0
        self.call_count = 0

    async def evaluate(self, batch, candidate, capture_traces, example_bank=None):
        self.call_count += 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(self._delay)
            trajectories = None
            if capture_traces:
                trajectories = [
                    AgentAdapterTrajectory(
                        messages=[],
                        final_output=None,
                        case=batch[0],
                    )
                ]

            return EvaluationBatch(
                outputs=[RolloutOutput.from_success(candidate["system"].text)],
                scores=[1.0],
                trajectories=trajectories,
            )
        finally:
            self.inflight -= 1

    def make_reflective_dataset(
        self,
        *,
        candidate,
        eval_batch,
        components_to_update,
        include_case_metadata: bool = False,
        include_expected_output: bool = False,
    ):
        return SharedReflectiveDataset(records=[])

    def get_components(self) -> CandidateMap:
        return {"instructions": ComponentValue(name="instructions", text="seed")}


class _CachingAdapter:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, tuple[tuple[str, str], ...]], EvaluationBatch] = {}
        self.llm_calls = 0
        self.cache_hits = 0

    async def evaluate(self, batch, candidate, capture_traces, example_bank=None):
        key = (
            batch[0].name,
            tuple(sorted((name, value.text) for name, value in candidate.items())),
        )
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        self.llm_calls += 1
        batch_result = EvaluationBatch(
            outputs=[RolloutOutput.from_success(f"out-{self.llm_calls}")],
            scores=[float(self.llm_calls)],
        )
        self._cache[key] = batch_result
        return batch_result

    def make_reflective_dataset(
        self,
        *,
        candidate,
        eval_batch,
        components_to_update,
        include_case_metadata: bool = False,
        include_expected_output: bool = False,
    ):
        return SharedReflectiveDataset(records=[])

    def get_components(self) -> CandidateMap:
        return {"instructions": ComponentValue(name="instructions", text="seed")}


@pytest.mark.asyncio
async def test_parallel_evaluator_limits_concurrency() -> None:
    evaluator = ParallelEvaluator()
    adapter = _RecordingAdapter(delay=0.05)
    batch = [_make_data_inst(str(i)) for i in range(5)]

    result = await evaluator.evaluate_batch(
        candidate=_make_candidate(),
        batch=batch,
        adapter=adapter,
        max_concurrent=2,
    )

    assert len(result) == len(batch)
    assert adapter.max_inflight == 2


@pytest.mark.asyncio
async def test_parallel_evaluator_integrates_with_adapter_cache() -> None:
    evaluator = ParallelEvaluator()
    adapter = _CachingAdapter()
    batch = [_make_data_inst(str(i)) for i in range(4)]
    candidate = _make_candidate()

    await evaluator.evaluate_batch(
        candidate=candidate,
        batch=batch,
        adapter=adapter,
    )
    await evaluator.evaluate_batch(
        candidate=candidate,
        batch=batch,
        adapter=adapter,
    )

    assert adapter.llm_calls == len(batch)
    assert adapter.cache_hits == len(batch)


@pytest.mark.asyncio
async def test_parallel_evaluator_collects_trajectories() -> None:
    evaluator = ParallelEvaluator()
    adapter = _RecordingAdapter(delay=0)
    batch = [_make_data_inst(str(i)) for i in range(3)]

    result = await evaluator.evaluate_batch(
        candidate=_make_candidate(),
        batch=batch,
        adapter=adapter,
        capture_traces=True,
    )

    assert result.trajectories is not None
    assert len(result.trajectories) == len(batch)


class _StoppingAdapter(_RecordingAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled: list[str] = []

    async def evaluate(self, batch, candidate, capture_traces, example_bank=None):
        from pydantic_ai_gepa.provider_errors import ProviderStopError

        if batch[0].name == "stops":
            await self.started.wait()
            raise ProviderStopError("insufficient_quota")
        self.started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled.append(batch[0].name)
            raise
        raise AssertionError("not reached")


@pytest.mark.asyncio
async def test_parallel_evaluator_cancels_in_flight_cases_on_provider_stop() -> None:
    from pydantic_ai_gepa.provider_errors import ProviderStopError

    adapter = _StoppingAdapter()
    with pytest.raises(ProviderStopError):
        await ParallelEvaluator().evaluate_batch(
            candidate=_make_candidate(),
            batch=[_make_data_inst("slow"), _make_data_inst("stops")],
            adapter=adapter,
            max_concurrent=2,
        )
    assert adapter.cancelled == ["slow"]

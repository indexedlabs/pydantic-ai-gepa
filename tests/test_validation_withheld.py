"""Validation evidence never becomes a reflector artifact."""

import json
import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa._validation import validation_active, validation_evaluation
from pydantic_ai_gepa.adapters.agent_adapter import _BaseAgentAdapter, create_adapter
from pydantic_ai_gepa.cache import CacheManager
from pydantic_ai_gepa.gepa_graph.proposal.instruction import ProposalResult
from pydantic_ai_gepa.runner import optimize_agent
from pydantic_ai_gepa.types import MetricResult, ReflectionConfig


@pytest.mark.asyncio
async def test_optimize_withholds_validation_from_traces_cache_and_reflection(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    seen_validation = []
    process_case = _BaseAgentAdapter.process_case

    async def observe(self, *args, **kwargs):
        result = await process_case(self, *args, **kwargs)
        if validation_active():
            seen_validation.append(result)
        return result

    monkeypatch.setattr(_BaseAgentAdapter, "process_case", observe)
    reflected = []

    class Proposer:
        async def propose_texts(self, *, reflective_data, component_toolsets, **kwargs):
            reflected.append(reflective_data)
            for toolset in component_toolsets:
                if "run_python_repl" in toolset.tools:
                    repl = toolset.tools["run_python_repl"].function
                    listed = await repl("list_dir('traces')")
                    assert "traces.jsonl" in listed
                    traces = await repl("read_file('traces/traces.jsonl')")
                    assert "TRAINING_INPUT" in traces
                    assert "WITHHELD" not in traces
            return ProposalResult(
                texts={"instructions": "improved"},
                component_metadata={},
                reasoning=None,
            )

    def metric(case, output):
        return MetricResult(
            score=0.25,
            feedback=f"{case.name} feedback",
            side_info={"detail": f"{case.name} side info"},
        )

    result = await optimize_agent(
        Agent(TestModel(custom_output_text="answer"), instructions="seed"),
        [Case(name="TRAINING", inputs="TRAINING_INPUT")],
        metric=metric,
        valset=[Case(name="WITHHELD_CASE", inputs="WITHHELD_INPUT")],
        reflection_config=ReflectionConfig(model=TestModel()),
        deterministic_proposer=Proposer(),
        max_iterations=1,
        max_metric_calls=10,
        reflection_minibatch_size=1,
        show_progress=False,
        enable_cache=True,
    )
    assert result.original_score == 0.25
    assert seen_validation
    for item in seen_validation:
        assert item["feedback"] is None
        assert "metric_side_info" not in item
        assert "trajectory" not in item
    assert reflected
    records = json.dumps(reflected[0].records, default=str)
    assert "TRAINING feedback" in records
    assert "TRAINING side info" in records
    assert "WITHHELD" not in records
    files = list((tmp_path / ".gepa_cache").rglob("*"))
    assert any(path.suffix == ".pkl" for path in files)
    assert any(path.name == "traces.jsonl" for path in files)
    for path in files:
        if path.is_file():
            assert b"WITHHELD" not in path.read_bytes(), path


@pytest.mark.asyncio
async def test_validation_discards_spans_and_metric_details_even_if_capture_requested(
    tmp_path,
):
    metric = MetricResult(0.5, feedback="WITHHELD", side_info={"secret": "WITHHELD"})
    adapter = create_adapter(
        agent=Agent(TestModel(custom_output_text="WITHHELD"), instructions="seed"),
        metric=lambda *args: metric,
        cache_manager=CacheManager(cache_dir=tmp_path),
    )
    try:
        with validation_evaluation(adapter.trace_collector.exporter):
            result = await adapter.process_case(
                Case(name="WITHHELD", inputs="WITHHELD"),
                0,
                capture_traces=True,
                candidate=adapter.get_components(),
            )
            assert adapter.trace_collector.exporter.get_finished_spans()
        assert result["score"] == 0.5
        assert result["feedback"] is None
        assert "metric_side_info" not in result
        assert "trajectory" not in result
        assert metric.feedback == "WITHHELD"  # Never mutate a caller-owned metric.
        assert not adapter.trace_collector.exporter.get_finished_spans()
        assert not list(tmp_path.iterdir())
    finally:
        adapter.close()


def test_validation_context_drains_on_failure():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with pytest.raises(RuntimeError), validation_evaluation(exporter):
            with provider.get_tracer(__name__).start_as_current_span("WITHHELD"):
                pass
            raise RuntimeError("scoring failed")
        assert not exporter.get_finished_spans()
        assert not validation_active()
    finally:
        provider.shutdown()


@pytest.mark.asyncio
async def test_validation_failure_drains_pending_rollouts_before_context_exit():
    from pydantic_ai_gepa._concurrency import gather_cancelling_on_provider_stop

    started = asyncio.Event()
    finished = asyncio.Event()

    async def pending():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            assert validation_active()
            finished.set()

    async def failing():
        await started.wait()
        raise ValueError("validation failure")

    with pytest.raises(ValueError), validation_evaluation():
        await gather_cancelling_on_provider_stop(pending(), failing())
    assert finished.is_set()
    assert not validation_active()


@pytest.mark.parametrize("stage", ["validation", "merge"])
def test_validation_error_summaries_withhold_case_and_error_details(stage):
    from pydantic_ai_gepa.gepa_graph.models import GepaConfig, GepaState
    from pydantic_ai_gepa.gepa_graph.datasets import ListDataLoader
    from pydantic_ai_gepa.types import RolloutOutput

    state = GepaState(
        config=GepaConfig(),
        iteration=0,
        training_set=ListDataLoader([Case(inputs="training")]),
    )
    state.record_evaluation_errors(
        candidate_idx=0,
        stage=stage,
        data_ids=["WITHHELD_CASE"],
        outputs=[RolloutOutput.from_error(ValueError("WITHHELD_GOLD"))],
    )
    assert len(state.evaluation_errors) == 1
    assert "WITHHELD" not in state.evaluation_errors[0].model_dump_json()

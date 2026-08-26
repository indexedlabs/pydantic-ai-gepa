"""Run-scoped trace correlation and collection for GEPA evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any, Literal

from opentelemetry import trace
from opentelemetry.context import Context, attach, detach
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic_ai import RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    Instrumentation,
    WrapRunHandler,
)
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.run import AgentRunResult

TraceCompleteness = Literal["root-only", "full"]
DEFAULT_MAX_RETAINED_SPANS = 100_000


def select_spans_by_trace_id(
    exporter: InMemorySpanExporter,
    trace_ids: set[str],
) -> tuple[ReadableSpan, ...]:
    """Select finished spans without clearing or timing-partitioning the exporter."""
    numeric_trace_ids: set[int] = set()
    for trace_id in trace_ids:
        try:
            numeric_trace_ids.add(int(trace_id, 16))
        except ValueError:
            continue
    return tuple(
        span
        for span in exporter.get_finished_spans()
        if span.context is not None and span.context.trace_id in numeric_trace_ids
    )


def drain_spans_by_trace_id(
    exporter: InMemorySpanExporter,
    trace_ids: set[str],
) -> tuple[ReadableSpan, ...]:
    """Select one completed batch and release all retained batch spans."""
    spans = select_spans_by_trace_id(exporter, trace_ids)
    exporter.clear()
    return spans


@dataclass(slots=True)
class GepaTraceContext:
    """Mutable result populated by one trace-context capability invocation."""

    gepa_run_id: str
    candidate_id: str
    case_id: str
    agent_run_id: str
    expected_nested_agents: frozenset[str] = frozenset()
    trace_id: str | None = None
    root_span_id: str | None = None
    seen_agent_names: frozenset[str] = frozenset()
    completeness: TraceCompleteness = "root-only"


@dataclass
class GepaTraceContextCapability(AbstractCapability[Any]):
    """Attach GEPA correlation attributes inside Pydantic AI instrumentation."""

    context: GepaTraceContext = field(
        default_factory=lambda: GepaTraceContext("", "", "", "")
    )

    @classmethod
    def get_serialization_name(cls) -> None:
        return None

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(
            wrapped_by=(Instrumentation,),
            requires=(Instrumentation,),
        )

    async def wrap_run(
        self,
        ctx: RunContext[Any],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        span = trace.get_current_span()
        span_context = span.get_span_context()
        if span_context.is_valid:
            self.context.trace_id = format(span_context.trace_id, "032x")
            self.context.root_span_id = format(span_context.span_id, "016x")
        if span.is_recording():
            span.set_attributes(
                {
                    "gepa.run.id": self.context.gepa_run_id,
                    "gepa.candidate.id": self.context.candidate_id,
                    "gepa.case.id": self.context.case_id,
                    "gepa.agent_run.id": self.context.agent_run_id,
                }
            )
        return await handler()


class GepaTraceCollector:
    """Own an isolated tracer provider and select spans by correlated trace ID."""

    def __init__(
        self,
        *,
        include_content: bool = True,
        max_retained_spans: int = DEFAULT_MAX_RETAINED_SPANS,
    ) -> None:
        self._exporter = InMemorySpanExporter(max_spans=max_retained_spans)
        self._provider = TracerProvider(shutdown_on_exit=False)
        self._provider.add_span_processor(SimpleSpanProcessor(self._exporter))
        self._settings = InstrumentationSettings(
            tracer_provider=self._provider,
            include_content=include_content,
            include_model_request_parameters=True,
        )

    @property
    def exporter(self) -> InMemorySpanExporter:
        """Expose the isolated exporter for legacy trace serialization."""
        return self._exporter

    @property
    def settings(self) -> InstrumentationSettings:
        """Settings nested agents can share to participate in the same trace."""
        return self._settings

    def capabilities(
        self,
        context: GepaTraceContext,
    ) -> tuple[Instrumentation, GepaTraceContextCapability]:
        """Return instrumentation plus GEPA correlation for one root run."""
        return (
            Instrumentation(settings=self._settings),
            GepaTraceContextCapability(context=context),
        )

    def nested_instrumentation(self) -> Instrumentation:
        """Return instrumentation a participating child agent can share."""
        return Instrumentation(settings=self._settings)

    @contextmanager
    def root_context(self) -> Iterator[None]:
        """Detach a rollout from ambient OTel parents so each case owns a trace."""
        token = attach(Context())
        try:
            yield
        finally:
            detach(token)

    def spans_for(self, context: GepaTraceContext) -> tuple[ReadableSpan, ...]:
        """Return only spans belonging to ``context`` and finalize completeness."""
        if context.trace_id is None:
            return ()
        spans = select_spans_by_trace_id(self._exporter, {context.trace_id})
        agent_names = frozenset(
            str(name)
            for span in spans
            if span.attributes is not None
            and (name := span.attributes.get("gen_ai.agent.name"))
        )
        context.seen_agent_names = agent_names
        context.completeness = (
            "full"
            if context.expected_nested_agents
            and context.expected_nested_agents.issubset(agent_names)
            else "root-only"
        )
        return spans

    def shutdown(self) -> None:
        """Flush and shut down this collector's run-owned tracer provider."""
        self._provider.shutdown()

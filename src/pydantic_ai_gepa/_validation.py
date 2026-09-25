"""Harness-owned context for evaluations whose evidence must be withheld."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_VALIDATION_ACTIVE: ContextVar[bool] = ContextVar(
    "gepa_validation_active", default=False
)


def validation_active() -> bool:
    """Whether the current rollout is withheld from reflection."""
    return _VALIDATION_ACTIVE.get()


@contextmanager
def validation_evaluation(
    exporter: InMemorySpanExporter | None = None,
) -> Iterator[None]:
    """Suppress persistent evidence and discard spans, including on failure."""
    token = _VALIDATION_ACTIVE.set(True)
    try:
        yield
    finally:
        if exporter is not None:
            exporter.clear()
        _VALIDATION_ACTIVE.reset(token)

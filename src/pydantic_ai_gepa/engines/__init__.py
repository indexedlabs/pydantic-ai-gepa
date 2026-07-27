"""Pluggable optimization engine contracts and registry helpers."""

from .base import (
    BudgetExhausted,
    BudgetTracker,
    CandidateEvaluation,
    EngineConfig,
    EngineEvent,
    EngineResult,
    OptimizationEngine,
    OptimizationTask,
)
from .registry import (
    EngineFactory,
    get_engine,
    list_engines,
    register_engine,
    unregister_engine,
)

__all__ = [
    "BudgetExhausted",
    "BudgetTracker",
    "CandidateEvaluation",
    "EngineConfig",
    "EngineEvent",
    "EngineFactory",
    "EngineResult",
    "OptimizationEngine",
    "OptimizationTask",
    "get_engine",
    "list_engines",
    "register_engine",
    "unregister_engine",
]

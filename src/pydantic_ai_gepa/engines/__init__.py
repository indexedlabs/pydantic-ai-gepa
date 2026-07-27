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
from .gepa_engine import GepaEngine

__all__ = [
    "BudgetExhausted",
    "BudgetTracker",
    "CandidateEvaluation",
    "EngineConfig",
    "EngineEvent",
    "EngineFactory",
    "EngineResult",
    "GepaEngine",
    "OptimizationEngine",
    "OptimizationTask",
    "get_engine",
    "list_engines",
    "register_engine",
    "unregister_engine",
]

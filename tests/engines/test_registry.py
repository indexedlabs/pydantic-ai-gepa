"""Tests for optimization engine registration."""

from __future__ import annotations

import pytest

from pydantic_ai_gepa.engines.base import (
    BudgetTracker,
    EngineConfig,
    EngineResult,
    OptimizationTask,
)
from pydantic_ai_gepa.engines.registry import (
    get_engine,
    list_engines,
    register_engine,
    unregister_engine,
)


class _TestEngine:
    name = "test_engine"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    async def run(
        self,
        task: OptimizationTask,
        config: EngineConfig,
        budget: BudgetTracker,
    ) -> EngineResult:
        raise NotImplementedError


def test_registry_registers_gets_lists_replaces_and_unregisters() -> None:
    """Factories are discoverable and replacement is explicit."""
    name = "test_engine_registry_unit"
    unregister_engine(name)
    config = EngineConfig(engine=name)

    try:
        register_engine(name, _TestEngine)
        assert name in list_engines()
        engine = get_engine(name, config)
        assert isinstance(engine, _TestEngine)
        assert engine.config is config

        with pytest.raises(ValueError, match="already registered"):
            register_engine(name, _TestEngine)

        register_engine(name, _TestEngine, replace=True)
        assert isinstance(get_engine(name, config), _TestEngine)
    finally:
        unregister_engine(name)

    assert name not in list_engines()


def test_registry_rejects_empty_and_describes_unknown_engine() -> None:
    """Invalid and unknown names yield actionable errors."""
    with pytest.raises(ValueError, match="non-empty"):
        register_engine("", _TestEngine)

    with pytest.raises(KeyError, match="Registered engines"):
        get_engine("missing_engine_registry_unit", EngineConfig(engine="missing"))

"""Autonomous-research engine seam for caller-owned long-horizon loops."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeAlias, cast

from .base import BudgetTracker, EngineConfig, EngineResult, OptimizationTask
from .registry import register_engine


AutonomousResearchDriver: TypeAlias = Callable[
    [OptimizationTask, EngineConfig, BudgetTracker], Awaitable[EngineResult]
]


class AutonomousResearchEngine:
    """Delegate an autonomous optimization loop without hardcoding an agent CLI.

    The driver owns its long-lived coding-agent/eval-server interaction and
    must consume the supplied budget. This keeps vendor/model process policy
    out of the library while retaining the common result/accounting contract.
    """

    name = "autoresearch"

    def __init__(self, config: EngineConfig) -> None:
        driver = config.engine_config.get("driver")
        if not callable(driver):
            raise TypeError(
                "engine_config['driver'] must be an async callable accepting "
                "(OptimizationTask, EngineConfig, BudgetTracker)."
            )
        self._driver = cast(AutonomousResearchDriver, driver)

    async def run(
        self, task: OptimizationTask, config: EngineConfig, budget: BudgetTracker
    ) -> EngineResult:
        before = budget.spent
        result = await self._driver(task, config, budget)
        consumed = budget.spent - before
        if result.num_metric_calls != consumed:
            raise RuntimeError(
                "Autonomous-research driver accounting mismatch: its EngineResult "
                f"reports {result.num_metric_calls}, but it spent {consumed}."
            )
        if result.engine != self.name:
            return result.model_copy(update={"engine": self.name})
        return result


register_engine(AutonomousResearchEngine.name, AutonomousResearchEngine, replace=True)


__all__ = ["AutonomousResearchDriver", "AutonomousResearchEngine"]

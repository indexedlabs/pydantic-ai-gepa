"""Registry for discovering pluggable optimization engine implementations."""

from __future__ import annotations

from collections.abc import Callable

from .base import EngineConfig, OptimizationEngine

EngineFactory = Callable[[EngineConfig], OptimizationEngine]

_ENGINES: dict[str, EngineFactory] = {}


def register_engine(
    name: str,
    factory: EngineFactory,
    *,
    replace: bool = False,
) -> None:
    """Register an optimization engine factory under ``name``."""
    if not name:
        raise ValueError("Engine name must be non-empty.")
    if name in _ENGINES and not replace:
        raise ValueError(f"An engine named {name!r} is already registered.")
    _ENGINES[name] = factory


def get_engine(name: str, config: EngineConfig) -> OptimizationEngine:
    """Create the engine registered under ``name`` using ``config``."""
    try:
        factory = _ENGINES[name]
    except KeyError as exc:
        registered = ", ".join(list_engines()) or "(none)"
        raise KeyError(
            f"Unknown optimization engine {name!r}. Registered engines: {registered}."
        ) from exc
    return factory(config)


def list_engines() -> tuple[str, ...]:
    """Return registered engine names in deterministic order."""
    return tuple(sorted(_ENGINES))


def unregister_engine(name: str) -> None:
    """Remove an engine registration when it exists."""
    _ENGINES.pop(name, None)


__all__ = [
    "EngineFactory",
    "get_engine",
    "list_engines",
    "register_engine",
    "unregister_engine",
]

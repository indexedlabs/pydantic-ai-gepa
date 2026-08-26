"""Instruction rendering for Pydantic AI's resolved run capabilities."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

from pydantic_ai import RunContext


def _normalize_instructions(instructions: Any) -> list[Any]:
    if instructions is None:
        return []
    if isinstance(instructions, Sequence) and not isinstance(instructions, str):
        return list(instructions)
    return [instructions]


async def resolved_capability_instructions(ctx: RunContext[Any]) -> str | None:
    """Render instructions after capability functions have resolved for a run."""
    rendered: list[str] = []
    for capability in ctx.capabilities.values():
        if capability.defer_loading is True:
            continue
        for instruction in _normalize_instructions(capability.get_instructions()):
            if isinstance(instruction, str):
                rendered.append(instruction)
                continue
            if not callable(instruction):
                continue
            try:
                parameters = inspect.signature(instruction).parameters
            except (TypeError, ValueError):
                parameters = {"ctx": None}
            value = instruction() if not parameters else instruction(ctx)
            if inspect.isawaitable(value):
                value = await value
            if value is not None:
                rendered.append(str(value))
    return "\n".join(rendered) or None

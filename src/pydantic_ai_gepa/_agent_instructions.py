"""Read an agent's declared instructions as raw strings and callables."""

from __future__ import annotations

from typing import Any

from pydantic_ai._instructions import SourcedInstruction


def unwrap_instruction(item: Any) -> Any:
    """Return the raw string or callable behind one stored instruction.

    pydantic-ai wraps every entry of ``Agent._instructions`` in a
    ``SourcedInstruction`` that records where it came from. GEPA mutates the
    literal text and re-passes callbacks through ``override(instructions=...)``,
    which only accepts the raw values, so the wrapper must be removed first.
    """
    if isinstance(item, SourcedInstruction):
        return item.instruction
    return item


def declared_instructions(agent: Any) -> list[Any] | None:
    """Return the agent's constructor/decorator instructions as raw values."""
    raw = getattr(agent, "_instructions", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        return [raw]
    return [unwrap_instruction(item) for item in raw]

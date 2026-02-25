"""Helpers for exposing the mcp-run-python sandbox as a PydanticAI tool."""

from __future__ import annotations

import json
import time
from typing import Literal, TypeAlias, TYPE_CHECKING

from pydantic import BaseModel, Field
from pydantic_ai import Tool

from typing_extensions import TypeAliasType

import ouros
from typing import TYPE_CHECKING, Literal, TypeAlias
from typing_extensions import TypeAliasType
from pydantic import BaseModel, Field
import json
import time

if TYPE_CHECKING:
    JsonValue: TypeAlias = (
        str | bool | int | float | None | list["JsonValue"] | dict[str, "JsonValue"]
    )
else:
    JsonValue = TypeAliasType(
        "JsonValue",
        str | bool | int | float | None | list["JsonValue"] | dict[str, "JsonValue"],
    )

class SandboxExecutionResult(BaseModel):
    """Structured result returned by the sandbox tool."""

    status: Literal["success", "install-error", "run-error"] = Field(
        description="Overall execution status returned by the sandbox.",
    )
    return_value: JsonValue | None = Field(
        default=None,
        description="JSON-serialisable return value from the script, if any.",
    )
    error: str | None = Field(
        default=None,
        description="Error message when the sandbox reports an installation or runtime failure.",
    )
    output_lines: list[str] = Field(
        default_factory=list,
        description="Ordered stdout/stderr lines emitted by the executed code.",
    )
    answer_text: str | None = Field(
        default=None,
        description="Best-effort single-line summary of the computed answer.",
    )
    answer_source: Literal["return_value", "stdout", "none"] = Field(
        default="none",
        description="Where the answer_text was derived from.",
    )
    elapsed_seconds: float = Field(
        description="Total wall-clock time spent creating the sandbox and executing the script.",
    )

def _summarize_answer(
    return_value: JsonValue | None,
    output_lines: list[str],
) -> tuple[str | None, Literal["return_value", "stdout", "none"]]:
    if return_value is not None:
        if isinstance(return_value, str):
            answer = return_value
        else:
            answer = json.dumps(return_value, ensure_ascii=False, separators=(",", ":"))
        return answer, "return_value"

    for line in reversed(output_lines):
        stripped = line.strip()
        if stripped:
            return stripped, "stdout"

    return None, "none"

async def _run_python_in_sandbox(
    code: str,
    *,
    inputs: dict[str, JsonValue] | None = None,
) -> SandboxExecutionResult:
    """Execute arbitrary Python inside the ouros sandbox.

    Args:
        code: Complete Python script to run. Include all required imports and print the final answer.
        inputs: Optional JSON-compatible mapping that will be injected as global variables when the script starts.
    """
    started = time.perf_counter()
    import logging
    print(f"Executing sandbox code:\\n{code}\\n")

    try:
        sandbox = ouros.Sandbox(code)
        output_lines_buffer = []
        def _print_callback(fd: Literal["stdout"], text: str) -> None:
            output_lines_buffer.append(text)
            
        result = await ouros.run_async(
            sandbox,
            inputs=inputs if inputs else None,
            print_callback=_print_callback,
            limits=ouros.ResourceLimits(
                timeout_ms=10000, 
                memory_bytes=500_000_000, 
                instruction_count=100_000_000
            )
        )
        sandbox_result = {
            "status": "success",
            "return_value": result,
            "output": "".join(output_lines_buffer).splitlines()
        }
    except Exception as exc:  # pragma: no cover - surfaced to the model instead
        elapsed = time.perf_counter() - started
        return SandboxExecutionResult(
            status="run-error",
            return_value=None,
            error=f"Sandbox execution failed: {exc}",
            output_lines=[],
            answer_text=None,
            answer_source="none",
            elapsed_seconds=elapsed,
        )

    elapsed = time.perf_counter() - started
    output_lines: list[str] = sandbox_result.get("output", [])
    answer_text, answer_source = _summarize_answer(
        sandbox_result.get("return_value"),
        output_lines,
    )

    return SandboxExecutionResult(
        status="success",
        return_value=sandbox_result.get("return_value"),
        error=None,
        output_lines=output_lines,
        answer_text=answer_text,
        answer_source=answer_source,
        elapsed_seconds=elapsed,
    )


run_python_tool = Tool(
    _run_python_in_sandbox,
    name="run_python",
    description=(
        "Execute Python code inside an isolated sandbox using only the Python standard library. "
        "Provide fully self-contained scripts that print their final answer."
    ),
)

__all__ = ["JsonValue", "SandboxExecutionResult", "run_python_tool"]

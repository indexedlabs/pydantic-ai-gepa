from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, FunctionToolset

from ...types import ReflectionConfig


def create_trace_toolset(run_id: str, candidate_idx: int, reflection_model: Any = "gpt-4o-mini") -> FunctionToolset[None]:
    toolset = FunctionToolset[None]()
    traces_dir = Path(f".gepa_cache/runs/{run_id}/candidates/{candidate_idx}/traces")

    def read_traces() -> list[dict[str, Any]]:
        traces_file = traces_dir / "traces.jsonl"
        if not traces_file.exists():
            return []
        traces = []
        with open(traces_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return traces

    @toolset.tool
    async def run_trace_analysis(python_script: str) -> str:
        """Execute a Python script to analyze the execution traces using the ouros data plane.
        
        The script has access to two external functions:
        - `get_traces()`: Returns a list of dictionaries representing the OTel spans for all evaluations.
        
        Example:
        ```python
        traces = get_traces()
        failed = [t for t in traces if "error" in str(t)]
        return f"Found {len(failed)} failed traces"
        ```
        """
        try:
            import ouros
        except ImportError:
            return "Error: ouros is not installed. Trace analysis is unavailable."

        def get_traces() -> list[dict[str, Any]]:
            return read_traces()

        sandbox = ouros.Sandbox(python_script, external_functions=["get_traces"])
        print(f"Executing Ouros trace analysis script:\\n{python_script}")
        try:
            result = await ouros.run_async(
                sandbox,
                external_functions={"get_traces": get_traces},
                limits=ouros.ResourceLimits(timeout_ms=10000, memory_bytes=500_000_000, instruction_count=10_000_000),
            )
            return str(result)
        except Exception as e:
            return f"Error executing script: {e}"

    @toolset.tool
    async def analyze_trace_with_llm(trace_id: str, prompt: str) -> str:
        """Spawn a lightweight sub-agent to analyze a specific trace.
        
        Args:
            trace_id: The `context.trace_id` from the span you want to analyze (find this using `run_trace_analysis`).
            prompt: The specific semantic question for the sub-agent (e.g. "Did the agent misunderstand the API tool?").
        """
        traces = read_traces()
        target_trace = None
        for trace in traces:
            ctx = trace.get("context", {})
            if str(ctx.get("trace_id", "")) == str(trace_id):
                target_trace = trace
                break
        
        if not target_trace:
            # Maybe the trace_id is formatted differently, let's just do a string search
            for trace in traces:
                if str(trace_id) in str(trace):
                    target_trace = trace
                    break

        if not target_trace:
            return f"Error: trace {trace_id} not found."

        agent = Agent(reflection_model, system_prompt="You are a senior debugging engineer analyzing an execution trace.")
        full_prompt = f"""Analyze the following trace to answer the question.

Question: {prompt}

Trace Data:
{json.dumps(target_trace, indent=2)}"""
        
        try:
            result = await agent.run(full_prompt)
            return result.data
        except Exception as e:
            return f"Error running sub-agent: {e}"

    return toolset

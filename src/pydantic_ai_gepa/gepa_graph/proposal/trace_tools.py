from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, FunctionToolset


def create_trace_toolset(
    run_id: str, candidate_idx: int, reflection_model: Any = "gpt-4o-mini"
) -> FunctionToolset[None]:
    toolset = FunctionToolset[None]()
    base_dir = Path(f".gepa_cache/runs/{run_id}/candidates/{candidate_idx}")
    traces_dir = base_dir / "traces"

    def _read_file(path: str) -> str:
        safe_path = (base_dir / path).resolve()
        if not safe_path.is_relative_to(base_dir.resolve()):
            return f"Error: Path {path} is outside the allowed directory."
        if not safe_path.exists():
            return f"Error: File {path} not found."
        try:
            return safe_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"

    def _list_dir(path: str) -> list[str]:
        safe_path = (base_dir / path).resolve()
        if not safe_path.is_relative_to(base_dir.resolve()):
            return []
        if not safe_path.exists() or not safe_path.is_dir():
            return []
        try:
            return [str(p.relative_to(base_dir)) for p in safe_path.iterdir()]
        except Exception:
            return []

    def _json_loads(data: str) -> Any:
        return json.loads(data)

    @toolset.tool
    async def run_python_script(python_script: str) -> str:
        """Execute a Python script to explore structured files (RLM architecture).
        
        The script is executed via pydantic_monty (a safe Python subset) and has access to:
        - `read_file(path: str) -> str`: Reads a file relative to the context directory.
        - `list_dir(path: str) -> list[str]`: Lists directory contents.
        - `json_loads(data: str) -> Any`: Parses a JSON string into a Python object.
        
        Available structured files:
        - `components.json`: Contains the candidate components.
        - `traces/traces.jsonl`: Contains the execution traces.

        The script MUST return its output by returning the value from the last expression (or by assigning to a variable that is the last expression).

        Example finding a failed trace:
        ```python
        lines = read_file('traces/traces.jsonl').strip().split('\\n')
        failed_trace_ids = []
        for line in lines:
            if not line: continue
            data = json_loads(line)
            if not data.get('success'):
                failed_trace_ids.append(data.get('context', {}).get('trace_id'))
        failed_trace_ids
        ```
        """
        try:
            import pydantic_monty
        except ImportError:
            return "Error: pydantic_monty is not installed. File exploration unavailable."

        try:
            m = pydantic_monty.Monty(
                python_script,
                external_functions=["read_file", "list_dir", "json_loads"]
            )
            result = m.run(
                external_functions={
                    "read_file": _read_file,
                    "list_dir": _list_dir,
                    "json_loads": _json_loads,
                }
            )
            return str(result)
        except Exception as e:
            return f"Error executing script: {e}"

    @toolset.tool
    async def analyze_trace_with_llm(trace_id: str, prompt: str) -> str:
        """Spawn a lightweight sub-agent to deeply analyze a specific trace.
        
        Use this when a python script isn't enough to answer semantic questions about a trace.

        Args:
            trace_id: The `context.trace_id` from the span you want to analyze.
            prompt: The specific semantic question for the sub-agent (e.g. "Did the agent misunderstand the API tool?").
        """
        # Create a dedicated toolset for the sub-agent that contains the same run_python_script
        subagent_toolset = FunctionToolset[None]()
        subagent_toolset.tools["run_python_script"] = run_python_script

        agent = Agent(
            reflection_model,
            system_prompt=(
                f"You are a senior debugging engineer analyzing an execution trace (trace_id: {trace_id}).\\n"
                "You MUST use your `run_python_script` tool to read the file `traces/traces.jsonl` "
                "(and `components.json` if needed) to extract the context you need. "
                "Do NOT guess. Write python scripts to parse the JSONL and examine the exact conversation messages."
            ),
        )

        full_prompt = f"Analyze the trace to answer this question:\\n\\n{prompt}"

        try:
            result = await agent.run(full_prompt, toolsets=[subagent_toolset])
            return result.data
        except Exception as e:
            return f"Error running sub-agent: {e}"

    return toolset

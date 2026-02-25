from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, FunctionToolset


class ClearMessageHistoryException(Exception):
    def __init__(self, next_context: str):
        super().__init__("Clear message history requested")
        self.next_context = next_context


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
    async def analyze_trace_with_llm(trace_id: str, prompt: str) -> str:
        """Spawn a lightweight sub-agent to deeply analyze a specific trace.
        
        Use this when a python script isn't enough to answer semantic questions about a trace.

        Args:
            trace_id: The `context.trace_id` from the span you want to analyze.
            prompt: The specific semantic question for the sub-agent (e.g. "Did the agent misunderstand the API tool?").
        """
        # Maintain a persistent scratchpad dictionary across history clears to emulate a stateful REPL
        repl_state: dict[str, Any] = {}

        def _set_state(key: str, value: Any) -> str:
            repl_state[key] = value
            return f"Saved {key} to REPL state."

        def _get_state(key: str) -> Any:
            return repl_state.get(key)
            
        def _list_state() -> list[str]:
            return list(repl_state.keys())

        subagent_toolset = FunctionToolset[None]()

        @subagent_toolset.tool
        async def run_python_script(python_script: str) -> str:
            """Execute a Python script to explore structured files (RLM architecture).
            
            The script is executed via pydantic_monty (a safe Python subset) and has access to:
            - `read_file(path: str) -> str`: Reads a file relative to the context directory.
            - `list_dir(path: str) -> list[str]`: Lists directory contents.
            - `json_loads(data: str) -> Any`: Parses a JSON string into a Python object.
            - `set_state(key: str, value: Any) -> str`: Persist a variable between script executions.
            - `get_state(key: str) -> Any`: Retrieve a persisted variable from a previous script execution.
            - `list_state() -> list[str]`: List keys currently persisted in state.
            
            Available structured files:
            - `components.json`: Contains the candidate components.
            - `traces/traces.jsonl`: Contains the execution traces.

            The script MUST return its output by returning the value from the last expression.

            Example persisting data:
            ```python
            data = json_loads(read_file('traces/traces.jsonl').split('\\n')[0])
            set_state('first_trace', data)
            "Saved first trace"
            ```
            """
            try:
                import pydantic_monty
            except ImportError:
                return "Error: pydantic_monty is not installed. File exploration unavailable."

            try:
                m = pydantic_monty.Monty(
                    python_script,
                    external_functions=[
                        "read_file", "list_dir", "json_loads", 
                        "set_state", "get_state", "list_state"
                    ]
                )
                result = m.run(
                    external_functions={
                        "read_file": _read_file,
                        "list_dir": _list_dir,
                        "json_loads": _json_loads,
                        "set_state": _set_state,
                        "get_state": _get_state,
                        "list_state": _list_state,
                    }
                )
                return str(result)
            except Exception as e:
                return f"Error executing script: {e}"

        @subagent_toolset.tool
        def clear_message_history(next_context: str) -> str:
            """Clear your conversation history to free up context window space. 
            Execution will restart with `next_context` as your new starting prompt.
            Ensure you have persisted important data using `set_state` in a python script before calling this.
            """
            raise ClearMessageHistoryException(next_context)

        @subagent_toolset.tool
        async def spawn_agent(instructions: str) -> str:
            """Spawn a recursive sub-agent with a fresh context window to investigate a sub-problem. 
            It has access to its own python script tool and its own isolated REPL state, and its message history 
            does NOT affect your context window. It returns a string answer to your instructions.
            """
            return await _run_agent_loop(instructions, depth=1)

        async def _run_agent_loop(current_prompt: str, depth: int) -> str:
            system_prompt = (
                f"You are a senior debugging engineer analyzing an execution trace (trace_id: {trace_id}).\n"
                "You MUST use your `run_python_script` tool to read `traces/traces.jsonl` "
                "(and `components.json` if needed). Do NOT guess. Write python scripts to extract context.\n"
                "Use `set_state` inside your python scripts to persist data across script runs.\n"
                "If your conversation gets too long, save findings to state and call `clear_message_history`."
            ) if depth == 0 else (
                f"You are a recursive sub-agent exploring a sub-problem for trace {trace_id}.\n"
                "Use `run_python_script` to read files and maintain your own state via `set_state`.\n"
                "Return your final answer to the parent agent.\n"
                "If your conversation gets too long, save findings to state and call `clear_message_history`."
            )

            agent = Agent(
                reflection_model,
                system_prompt=system_prompt,
            )

            while True:
                try:
                    result = await agent.run(current_prompt, toolsets=[subagent_toolset])
                    return result.data
                except ClearMessageHistoryException as e:
                    current_prompt = (
                        f"History cleared. You previously left yourself this note to continue:\n\n{e.next_context}\n\n"
                        "Your REPL state is intact. You may use `get_state` in your python scripts to retrieve your saved data."
                    )
                except Exception as e:
                    return f"Error running sub-agent: {e}"

        return await _run_agent_loop(f"Analyze the trace to answer this question:\n\n{prompt}", depth=0)

    return toolset

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import uuid

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
    async def run_python_script(python_script: str) -> str:
        """Execute a Python script across ALL traces using the stateless ouros Sandbox.
        
        The script is executed via ouros.Sandbox and has access to:
        - `get_traces() -> list[dict]`: Returns all execution traces from disk.

        This is a global map-reduce tool. It is stateless.

        Example finding a failed trace:
        ```python
        traces = get_traces()
        failed = [t['context']['trace_id'] for t in traces if not t.get('success')]
        failed
        ```
        """
        try:
            import ouros
        except ImportError:
            return "Error: ouros is not installed. File exploration unavailable."

        def get_traces() -> list[dict[str, Any]]:
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

        sandbox = ouros.Sandbox(python_script, external_functions=["get_traces"])
        try:
            result = await ouros.run_async(
                sandbox,
                external_functions={"get_traces": get_traces},
                limits=ouros.ResourceLimits(
                    timeout_ms=10000,
                    memory_bytes=500_000_000,
                    instruction_count=10_000_000,
                ),
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
        try:
            import ouros
        except ImportError:
            return "Error: ouros is not installed."
        
        mgr = ouros.SessionManager()

        async def _run_agent_loop(current_prompt: str, depth: int) -> str:
            session_id = f"repl_{uuid.uuid4().hex[:8]}"
            session = mgr.create_session(session_id)

            subagent_toolset = FunctionToolset[None]()

            # Because Session.execute does not support passing `external_functions` natively like Sandbox,
            # we provide standard LLM tools for file operations that the agent can use alongside the REPL.
            @subagent_toolset.tool
            def read_file(path: str) -> str:
                """Read a file relative to the context directory."""
                return _read_file(path)

            @subagent_toolset.tool
            def list_dir(path: str) -> list[str]:
                """List directory contents relative to the context directory."""
                return _list_dir(path)

            @subagent_toolset.tool
            def run_python_repl(python_code: str) -> str:
                """Execute Python code in your persistent REPL environment.
                
                This is a stateful Jupyter-style REPL. Variables assigned here will persist 
                in memory for future `run_python_repl` calls within this agent loop.

                The script MUST return its output by returning the value from the last expression 
                (or by assigning to a variable that is the last expression).

                Example persisting data:
                ```python
                import json
                # Assuming you fetched json string via the `read_file` tool first
                trace_data = json.loads(my_json_string)
                first_trace_id = trace_data.get('context', {}).get('trace_id')
                first_trace_id
                ```
                """
                try:
                    result = session.execute(python_code)
                    if isinstance(result, dict) and 'result' in result:
                        return str(result['result'])
                    return str(result)
                except Exception as e:
                    return f"Error executing REPL code: {e}"

            @subagent_toolset.tool
            def clear_message_history(next_context: str) -> str:
                """Clear your conversation history to free up context window space. 
                Execution will restart with `next_context` as your new starting prompt.
                Because your Python REPL is stateful, any variables you declared previously 
                will still be available in memory when you call `run_python_repl` again.
                """
                raise ClearMessageHistoryException(next_context)
            
            @subagent_toolset.tool
            async def spawn_agent(instructions: str) -> str:
                """Spawn a recursive sub-agent with a fresh context window to investigate a sub-problem. 
                It has access to its own isolated Python REPL session. Its message history does NOT affect 
                your context window. It returns a string answer to your instructions.
                """
                return await _run_agent_loop(instructions, depth=depth + 1)

            system_prompt = (
                f"You are a senior debugging engineer analyzing an execution trace (trace_id: {trace_id}).\n"
                "You MUST use your `read_file` tool to read `traces/traces.jsonl` "
                "(and `components.json` if needed). Do NOT guess. Write python code via `run_python_repl` to parse and extract context.\n"
                "Your python environment is persistent. Variables stay in memory.\n"
                "If your conversation gets too long, store big lists/dicts in REPL variables and call `clear_message_history`."
            ) if depth == 0 else (
                f"You are a recursive sub-agent exploring a sub-problem for trace {trace_id}.\n"
                "Your python environment is persistent.\n"
                "Return your final answer to the parent agent.\n"
                "If your conversation gets too long, store findings in REPL variables and call `clear_message_history`."
            )

            agent = Agent(
                reflection_model,
                system_prompt=system_prompt,
            )

            # Execution loop to handle history clears
            while True:
                try:
                    result = await agent.run(current_prompt, toolsets=[subagent_toolset])
                    return result.data
                except ClearMessageHistoryException as e:
                    current_prompt = (
                        f"History cleared. You previously left yourself this note to continue:\n\n{e.next_context}\n\n"
                        "Your Python REPL state is intact. You may query the variables you defined earlier."
                    )
                except Exception as e:
                    return f"Error running sub-agent: {e}"

        return await _run_agent_loop(f"Analyze the trace to answer this question:\n\n{prompt}", depth=0)

    return toolset

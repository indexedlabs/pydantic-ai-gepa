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
    def read_file(path: str) -> str:
        """Read a file relative to the context directory."""
        return _read_file(path)

    @toolset.tool
    def list_dir(path: str) -> list[str]:
        """List directory contents relative to the context directory."""
        return _list_dir(path)

    try:
        import ouros
        import uuid
        mgr = ouros.SessionManager()
        
        session_id = f"repl_{uuid.uuid4().hex[:8]}"
        session = mgr.create_session(session_id)

        # Pre-load files into the session to simulate external functions
        import base64
        traces_content = ""
        traces_file = traces_dir / "traces.jsonl"
        if traces_file.exists():
            traces_content = base64.b64encode(traces_file.read_bytes()).decode("utf-8")
            
        components_content = ""
        components_file = base_dir / "components.json"
        if components_file.exists():
            components_content = base64.b64encode(components_file.read_bytes()).decode("utf-8")
            
        setup_script = f"""
import base64
import json
def read_file(path: str) -> str:
    if 'traces.jsonl' in path:
        return base64.b64decode('{traces_content}').decode('utf-8')
    if 'components.json' in path:
        return base64.b64decode('{components_content}').decode('utf-8')
    return "Error: File not found"

def json_loads(data: str):
    return json.loads(data)

def list_dir(path: str):
    return ['traces/traces.jsonl', 'components.json']
"""
        session.execute(setup_script)

        @toolset.tool
        async def run_python_repl(python_code: str) -> str:
            """Execute Python code in your persistent REPL environment.
            
            This is a stateful Jupyter-style REPL. Variables assigned here will persist 
            in memory for future `run_python_repl` calls.
            
            You have access to:
            - `read_file(path: str) -> str`: Reads a file relative to the context directory.
            - `list_dir(path: str) -> list[str]`: Lists directory contents.
            - `json_loads(data: str) -> Any`: Parses a JSON string into a Python object.
            
            Available structured files:
            - `components.json`: Contains the candidate components.
            - `traces/traces.jsonl`: Contains the execution traces.

            The script MUST return its output by returning the value from the last expression 
            (or by assigning to a variable that is the last expression).
            """
            try:
                result = session.execute(python_code)
                if isinstance(result, dict) and 'result' in result:
                    return str(result['result'])
                return str(result)
            except Exception as e:
                return f"Error executing REPL code: {e}"

        @toolset.tool
        def clear_message_history(next_context: str) -> str:
            """Clear your conversation history to free up context window space. 
            Execution will restart with `next_context` as your new starting prompt.
            Because your Python REPL is stateful, any variables you declared previously 
            will still be available in memory when you call `run_python_repl` again.
            """
            raise ClearMessageHistoryException(next_context)

        @toolset.tool
        async def spawn_agent(instructions: str) -> str:
            """Spawn a recursive sub-agent with a fresh context window to investigate a sub-problem. 
            It has access to its own isolated Python REPL session. Its message history does NOT affect 
            your context window. It returns a string answer to your instructions.
            """
            return await _run_child_agent(instructions)

        async def _run_child_agent(current_prompt: str) -> str:
            child_session_id = f"repl_{uuid.uuid4().hex[:8]}"
            child_session = mgr.create_session(child_session_id)

            # Pre-load files into the session to simulate external functions
            # without hitting the Ouros Session external function limitation.
            import base64
            traces_content = ""
            traces_file = traces_dir / "traces.jsonl"
            if traces_file.exists():
                traces_content = base64.b64encode(traces_file.read_bytes()).decode("utf-8")
                
            components_content = ""
            components_file = base_dir / "components.json"
            if components_file.exists():
                components_content = base64.b64encode(components_file.read_bytes()).decode("utf-8")
                
            setup_script = f"""
import base64
import json
def read_file(path: str) -> str:
    if 'traces.jsonl' in path:
        return base64.b64decode('{traces_content}').decode('utf-8')
    if 'components.json' in path:
        return base64.b64decode('{components_content}').decode('utf-8')
    return "Error: File not found"

def json_loads(data: str):
    return json.loads(data)

def list_dir(path: str):
    return ['traces/traces.jsonl', 'components.json']
"""
            child_session.execute(setup_script)

            child_toolset = FunctionToolset[None]()
            @child_toolset.tool
            async def run_python_repl(python_code: str) -> str:
                """Execute Python code in your persistent REPL environment.
                
                You have access to `read_file(path)`, `list_dir(path)`, and `json_loads(data)`
                pre-loaded in your Python environment.
                """
                try:
                    result = child_session.execute(python_code)
                    if isinstance(result, dict) and 'result' in result:
                        return str(result['result'])
                    return str(result)
                except Exception as e:
                    return f"Error executing REPL code: {e}"
            @child_toolset.tool
            def clear_message_history(next_context: str) -> str:
                """Clear your conversation history to free up context window space. 
                Use this sparingly to preserve prompt caching efficiency! Only use it 
                when your context window is overflowing with massive tool outputs.
                Execution will restart with `next_context` as your new starting prompt.
                """
                raise ClearMessageHistoryException(next_context)
            @child_toolset.tool
            async def spawn_agent(instructions: str) -> str:
                return await _run_child_agent(instructions)

            system_prompt = (
                f"You are a recursive sub-agent exploring a sub-problem for trace analysis.\n"
                "Your python environment is persistent. Variables stay in memory.\n"
                "Use `run_python_repl` to read and parse `traces/traces.jsonl` using the built-in `read_file` function.\n"
                "Return your final answer to the parent agent.\n"
                "IMPORTANT: To leverage LLM prompt caching, you should build up state in your Python REPL.\n"
                "Only call `clear_message_history` sparingly when absolutely necessary to avoid context limits."
            )
            agent = Agent(reflection_model, system_prompt=system_prompt)
            
            loop_count = 0
            while True:
                loop_count += 1
                if loop_count > 20:
                    return "Error: Child agent exceeded maximum clear_message_history loops (20)."
                try:
                    result = await agent.run(current_prompt, toolsets=[child_toolset])
                    return result.data
                except ClearMessageHistoryException as e:
                    current_prompt = (
                        f"History cleared. You previously left yourself this note to continue:\n\n{e.next_context}\n\n"
                        "Your Python REPL state is intact."
                    )
                except Exception as e:
                    return f"Error running sub-agent: {e}"

    except ImportError:
        pass

    return toolset

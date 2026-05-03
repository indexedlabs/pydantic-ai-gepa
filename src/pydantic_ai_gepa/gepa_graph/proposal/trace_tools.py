from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, FunctionToolset


class ClearMessageHistoryException(Exception):
    def __init__(self, next_context: str):
        super().__init__("Clear message history requested")
        self.next_context = next_context


_MONTY_CONTEXT_ROOT = "/ctx"
_MONTY_LIMITS = {
    "max_duration_secs": 10.0,
    "max_memory": 128 * 1024 * 1024,
    "max_recursion_depth": 1000,
}
_MAX_HOST_LINE_LIMIT = 1000
_MAX_HOST_LINE_CHARS = 20_000
_MAX_HOST_BATCH_BYTES = 4 * 1024 * 1024
_LINE_COUNT_CHUNK_SIZE = 1024 * 1024
MONTY_REPL_PROMPT_GUIDANCE = """
### `run_python_repl` environment
- The REPL is `pydantic-monty`, a sandboxed Python subset, not CPython. It is persistent across calls within one reflection agent run, so variables and helper functions stay bound.
- Each call has a 10-second execution budget, plus memory and recursion limits. A timed-out call does not make the REPL unusable; preserve useful intermediate state in variables.
- Return values come from the final expression. `print(...)` writes to stdout and returns `None`, so end scripts with a bare value such as `summary`, `rows[:5]`, or `{"failures": failures}`.
- Supported syntax includes assignments, `if`/`else`, `for`/`while`, `def`, `lambda`, `try`/`except`, `raise`, comprehensions, and f-strings.
- Unsupported syntax includes `with` statements, `class` definitions, `match`, and `yield`. Do not use context managers, generators, or custom classes.
- Unsupported runtime/builtins include `globals()`, `locals()`, `eval()`, `exec()`, and `__import__()`.
- Imports are limited to a small standard-library subset such as `json`, `re`, `datetime`, `typing`, `sys`, and partial `os`. Third-party packages and most stdlib modules are unavailable. Prefer the pre-bound helpers below instead of filesystem imports or `os.getcwd()`.
- Pre-bound helpers: `read_file`, `file_info`, `file_size`, `line_count`, `read_lines`, `read_line_batch`, `tail_lines`, `find_lines`, `list_dir`, `json_loads`, plus `Path` and `json`.

### Trace file navigation
- `traces/traces.jsonl` can be large. Avoid `read_file('traces/traces.jsonl')` unless you already know it is small; returning the whole trace file can overflow the reflection model context.
- Start with `file_info('traces/traces.jsonl')` to understand size and line count.
- Use `find_lines('traces/traces.jsonl', query, limit=20)` for targeted search and `tail_lines(..., limit=20)` for recent spans or exceptions.
- Use `read_lines(path, start=n, limit=10)` for a small window around a known line number.
- For full-file scans, write one reducer-style script around `read_line_batch(path, offset=0, limit=1000)`. Advance with `offset = batch['next_offset']`, stop when `batch['eof']`, and return only compact aggregates.

Canonical full-scan pattern:
```python
offset = 0
failures = 0
examples = []
while True:
    batch = read_line_batch('traces/traces.jsonl', offset=offset, limit=1000)
    for line in batch['lines']:
        row = json_loads(line)
        if not row.get('success'):
            failures = failures + 1
            if len(examples) < 5:
                examples.append(row.get('feedback'))
    if batch['eof']:
        break
    offset = batch['next_offset']
{'failures': failures, 'examples': examples}
```
""".strip()
_MONTY_SETUP_SCRIPT = f"""
from pathlib import Path
import json

_CONTEXT_ROOT = Path({_MONTY_CONTEXT_ROOT!r})

def _resolve_context_path(path: str):
    if path.startswith({_MONTY_CONTEXT_ROOT!r}):
        return Path(path)
    if path.startswith('/'):
        raise PermissionError(f"Path outside mounted context: {{path}}")
    return _CONTEXT_ROOT / path

def read_file(path: str) -> str:
    return _resolve_context_path(path).read_text()

def file_size(path: str) -> int:
    return host_file_size(str(path))

def line_count(path: str) -> int:
    return host_line_count(str(path))

def file_info(path: str):
    return {{
        'size_bytes': file_size(path),
        'line_count': line_count(path),
    }}

def read_lines(path: str, start: int = 0, limit: int = 100):
    return host_read_lines(str(path), start, limit)

def read_line_batch(
    path: str,
    offset: int = 0,
    limit: int = 1000,
    max_bytes: int = {_MAX_HOST_BATCH_BYTES},
):
    return host_read_line_batch(str(path), offset, limit, max_bytes)

def tail_lines(path: str, limit: int = 100):
    return host_tail_lines(str(path), limit)

def find_lines(
    path: str,
    query: str,
    start: int = 0,
    limit: int = 100,
    case_sensitive: bool = False,
):
    return host_find_lines(str(path), str(query), start, limit, case_sensitive)

def list_dir(path: str):
    root = _resolve_context_path(path)
    prefix = str(_CONTEXT_ROOT) + '/'
    items = []
    for item in root.iterdir():
        item_path = str(item)
        if item_path.startswith(prefix):
            items.append(item_path[len(prefix):])
        else:
            items.append(item_path)
    return sorted(items)

def json_loads(data: str):
    return json.loads(data)
"""


def create_trace_toolset(
    run_id: str, candidate_idx: int, reflection_model: Any = "gpt-4o-mini"
) -> FunctionToolset[None]:
    toolset = FunctionToolset[None]()
    base_dir = Path(f".gepa_cache/runs/{run_id}/candidates/{candidate_idx}").resolve()

    def _read_file(path: str) -> str:
        safe_path = (base_dir / path).resolve()
        if not safe_path.is_relative_to(base_dir):
            return f"Error: Path {path} is outside the allowed directory."
        if not safe_path.exists():
            return f"Error: File {path} not found."
        try:
            return safe_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"

    def _list_dir(path: str) -> list[str]:
        safe_path = (base_dir / path).resolve()
        if not safe_path.is_relative_to(base_dir):
            return []
        if not safe_path.exists() or not safe_path.is_dir():
            return []
        try:
            return [str(p.relative_to(base_dir)) for p in safe_path.iterdir()]
        except Exception:
            return []

    def _resolve_host_path(path: str) -> Path:
        normalized = str(path)
        if normalized == _MONTY_CONTEXT_ROOT:
            normalized = ""
        elif normalized.startswith(f"{_MONTY_CONTEXT_ROOT}/"):
            normalized = normalized[len(_MONTY_CONTEXT_ROOT) + 1 :]
        elif normalized.startswith("/"):
            raise PermissionError(f"Path {path} is outside the allowed directory.")

        safe_path = (base_dir / normalized).resolve()
        if not safe_path.is_relative_to(base_dir):
            raise PermissionError(f"Path {path} is outside the allowed directory.")
        return safe_path

    def _coerce_non_negative_int(value: Any, *, name: str) -> int:
        try:
            coerced = int(value)
        except Exception as e:
            raise ValueError(f"{name} must be an integer.") from e
        if coerced < 0:
            raise ValueError(f"{name} must be non-negative.")
        return coerced

    def _coerce_line_limit(value: Any) -> int:
        limit = _coerce_non_negative_int(value, name="limit")
        return min(limit, _MAX_HOST_LINE_LIMIT)

    def _coerce_positive_int(value: Any, *, name: str) -> int:
        coerced = _coerce_non_negative_int(value, name=name)
        if coerced == 0:
            raise ValueError(f"{name} must be positive.")
        return coerced

    def _coerce_batch_bytes(value: Any) -> int:
        max_bytes = _coerce_positive_int(value, name="max_bytes")
        return min(max_bytes, _MAX_HOST_BATCH_BYTES)

    def _trim_line(line: str) -> str:
        line = line.rstrip("\n")
        if len(line) <= _MAX_HOST_LINE_CHARS:
            return line
        return f"{line[:_MAX_HOST_LINE_CHARS]}... [truncated]"

    def _decode_line(line: bytes) -> str:
        return line.decode("utf-8", errors="replace").rstrip("\n")

    def _host_file_size(path: str) -> int:
        return _resolve_host_path(path).stat().st_size

    def _host_line_count(path: str) -> int:
        safe_path = _resolve_host_path(path)
        count = 0
        has_content = False
        last_byte = b""
        with safe_path.open("rb") as f:
            while chunk := f.read(_LINE_COUNT_CHUNK_SIZE):
                has_content = True
                count += chunk.count(b"\n")
                last_byte = chunk[-1:]
        if has_content and last_byte != b"\n":
            count += 1
        return count

    def _host_read_lines(path: str, start: int = 0, limit: int = 100) -> list[str]:
        safe_path = _resolve_host_path(path)
        start = _coerce_non_negative_int(start, name="start")
        limit = _coerce_line_limit(limit)
        if limit == 0:
            return []

        lines: list[str] = []
        with safe_path.open("r", encoding="utf-8", errors="replace") as f:
            for line_number, line in enumerate(f):
                if line_number < start:
                    continue
                lines.append(_trim_line(line))
                if len(lines) >= limit:
                    break
        return lines

    def _host_read_line_batch(
        path: str,
        offset: int = 0,
        limit: int = 1000,
        max_bytes: int = _MAX_HOST_BATCH_BYTES,
    ) -> dict[str, Any]:
        safe_path = _resolve_host_path(path)
        file_size = safe_path.stat().st_size
        offset = _coerce_non_negative_int(offset, name="offset")
        limit = _coerce_line_limit(limit)
        max_bytes = _coerce_batch_bytes(max_bytes)

        if limit == 0:
            return {
                "lines": [],
                "offset": offset,
                "next_offset": offset,
                "eof": offset >= file_size,
                "bytes_read": 0,
            }

        lines: list[str] = []
        next_offset = offset
        bytes_read = 0
        with safe_path.open("rb") as f:
            f.seek(offset)
            while len(lines) < limit:
                line_offset = f.tell()
                line = f.readline()
                if not line:
                    next_offset = f.tell()
                    break

                if lines and bytes_read + len(line) > max_bytes:
                    next_offset = line_offset
                    break

                lines.append(_decode_line(line))
                bytes_read += len(line)
                next_offset = f.tell()
                if bytes_read >= max_bytes:
                    break

        return {
            "lines": lines,
            "offset": offset,
            "next_offset": next_offset,
            "eof": next_offset >= file_size,
            "bytes_read": bytes_read,
        }

    def _host_tail_lines(path: str, limit: int = 100) -> list[str]:
        safe_path = _resolve_host_path(path)
        limit = _coerce_line_limit(limit)
        if limit == 0:
            return []

        lines = deque(maxlen=limit)
        with safe_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                lines.append(_trim_line(line))
        return list(lines)

    def _host_find_lines(
        path: str,
        query: str,
        start: int = 0,
        limit: int = 100,
        case_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        safe_path = _resolve_host_path(path)
        start = _coerce_non_negative_int(start, name="start")
        limit = _coerce_line_limit(limit)
        if limit == 0:
            return []

        needle = query if case_sensitive else query.casefold()
        matches: list[dict[str, Any]] = []
        with safe_path.open("r", encoding="utf-8", errors="replace") as f:
            for line_number, line in enumerate(f, start=1):
                if line_number <= start:
                    continue
                haystack = line if case_sensitive else line.casefold()
                if needle not in haystack:
                    continue
                matches.append(
                    {
                        "line_number": line_number,
                        "text": _trim_line(line),
                    }
                )
                if len(matches) >= limit:
                    break
        return matches

    @toolset.tool_plain
    def read_file(path: str) -> str:
        """Read a file relative to the context directory."""
        return _read_file(path)

    @toolset.tool_plain
    def list_dir(path: str) -> list[str]:
        """List directory contents relative to the context directory."""
        return _list_dir(path)

    try:
        import pydantic_monty

        base_dir.mkdir(parents=True, exist_ok=True)
        mount = pydantic_monty.MountDir(
            _MONTY_CONTEXT_ROOT,
            base_dir,
            mode="read-only",
        )
        external_functions = {
            "host_file_size": _host_file_size,
            "host_line_count": _host_line_count,
            "host_read_lines": _host_read_lines,
            "host_read_line_batch": _host_read_line_batch,
            "host_tail_lines": _host_tail_lines,
            "host_find_lines": _host_find_lines,
        }

        def _reset_repl_timer(repl: Any) -> Any:
            # Monty's duration limit is cumulative for a REPL tracker. Loading a
            # snapshot preserves state but resets the tracker's start time.
            return pydantic_monty.MontyRepl.load(repl.dump())

        def _new_repl():
            repl = pydantic_monty.MontyRepl(
                script_name="trace_analysis.py",
                limits=_MONTY_LIMITS,
            )
            repl.feed_run(
                _MONTY_SETUP_SCRIPT,
                mount=mount,
                external_functions=external_functions,
            )
            return _reset_repl_timer(repl)

        def _execute_repl(session_state: dict[str, Any], python_code: str) -> str:
            repl = _reset_repl_timer(session_state["repl"])
            session_state["repl"] = repl
            try:
                result = repl.feed_run(
                    python_code,
                    mount=mount,
                    external_functions=external_functions,
                )
                return str(result)
            finally:
                try:
                    session_state["repl"] = _reset_repl_timer(repl)
                except Exception:
                    session_state["repl"] = repl

        session = {"repl": _new_repl()}
        session_lock = asyncio.Lock()

        @toolset.tool_plain
        async def run_python_repl(python_code: str) -> str:
            """Execute Python code in your persistent REPL environment.

            This is a stateful Jupyter-style REPL. Variables assigned here will persist
            in memory for future `run_python_repl` calls.

            This is pydantic-monty, not CPython. Use a practical Python subset:
            assignments, control flow, functions, comprehensions, exceptions, f-strings,
            and final-expression returns work. Do not use `with`, `class`, `match`,
            `yield`, generators, `globals()`, `locals()`, `eval()`, `exec()`, or broad
            stdlib/third-party imports. `print(...)` is not the return value; end with
            a bare expression containing the compact result you want.

            You have access to:
            - `read_file(path: str) -> str`: Reads a file relative to the context directory.
            - `file_size(path: str) -> int`: Returns file size in bytes.
            - `line_count(path: str) -> int`: Counts lines without reading the file into the REPL.
            - `file_info(path: str) -> dict`: Returns size and line count.
            - `read_lines(path, start=0, limit=100) -> list[str]`: Reads a bounded line slice.
            - `read_line_batch(path, offset=0, limit=1000) -> dict`: Reads complete
              lines from a byte cursor and returns `lines`, `next_offset`, `eof`,
              and `bytes_read`. Use this for full-file scans of large trace files.
            - `tail_lines(path, limit=100) -> list[str]`: Reads a bounded tail slice.
            - `find_lines(path, query, start=0, limit=100, case_sensitive=False) -> list[dict]`: Streams matching lines.
            - `list_dir(path: str) -> list[str]`: Lists directory contents.
            - `json_loads(data: str) -> Any`: Parses a JSON string into a Python object.

            Available structured files:
            - `components.json`: Contains the candidate components.
            - `traces/traces.jsonl`: Contains the execution traces.

            The script MUST return its output by returning the value from the last expression
            (or by assigning to a variable that is the last expression).
            """
            async with session_lock:
                try:
                    return await asyncio.to_thread(_execute_repl, session, python_code)
                except Exception as e:
                    return f"Error executing REPL code: {e}"

        @toolset.tool_plain
        def clear_message_history(next_context: str) -> str:
            """Clear your conversation history to free up context window space.
            Execution will restart with `next_context` as your new starting prompt.
            Because your Python REPL is stateful, any variables you declared previously
            will still be available in memory when you call `run_python_repl` again.
            """
            raise ClearMessageHistoryException(next_context)

        @toolset.tool_plain
        async def spawn_agent(instructions: str) -> str:
            """Spawn a recursive sub-agent with a fresh context window to investigate a sub-problem.
            It has access to its own isolated Python REPL session. Its message history does NOT affect
            your context window. It returns a string answer to your instructions.
            """
            return await _run_child_agent(instructions)

        async def _run_child_agent(current_prompt: str) -> str:
            child_session = {"repl": _new_repl()}
            child_session_lock = asyncio.Lock()

            child_toolset = FunctionToolset[None]()

            @child_toolset.tool_plain
            async def run_python_repl(python_code: str) -> str:
                """Execute Python code in your persistent REPL environment.

                You have access to file helpers including `read_file`, `file_size`,
                `line_count`, `file_info`, `read_lines`, `read_line_batch`,
                `tail_lines`, `find_lines`, `list_dir`, and `json_loads`.
                This is pydantic-monty, not CPython; avoid unsupported syntax such
                as `with`, `class`, `match`, and `yield`, and return a final expression.
                """
                async with child_session_lock:
                    try:
                        return await asyncio.to_thread(
                            _execute_repl, child_session, python_code
                        )
                    except Exception as e:
                        return f"Error executing REPL code: {e}"

            @child_toolset.tool_plain
            def clear_message_history(next_context: str) -> str:
                """Clear your conversation history to free up context window space.
                Use this sparingly to preserve prompt caching efficiency! Only use it
                when your context window is overflowing with massive tool outputs.
                Execution will restart with `next_context` as your new starting prompt.
                """
                raise ClearMessageHistoryException(next_context)

            @child_toolset.tool_plain
            async def spawn_agent(instructions: str) -> str:
                return await _run_child_agent(instructions)

            system_prompt = (
                "You are a recursive sub-agent exploring a sub-problem for trace analysis.\n"
                "Your python environment is persistent. Variables stay in memory.\n"
                "Use `run_python_repl` to parse `traces/traces.jsonl`; prefer `read_line_batch` for full-file scans.\n"
                "The REPL is pydantic-monty, not CPython: avoid `with`, `class`, `match`, `yield`, unsupported imports, and `print` as a return value.\n"
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
                    return result.output
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

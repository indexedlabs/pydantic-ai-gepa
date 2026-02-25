# RLM Architecture Upgrades & Fixes (ADR)

## Single Top-Level RLM
Instead of an intermediate `analyze_trace_with_llm` agent, we shifted to a single `InstructionProposalGenerator` reflector agent. This main agent manages its own persistent `run_python_repl` context. When deep, messy, or destructive exploration is needed, it can spawn child agents via `spawn_agent(instructions)`. This keeps the parent's context window and stateful memory pristine.

## Ouros Stateful Session
We migrated from the stateless `ouros.Sandbox` to the stateful `ouros.SessionManager`. 
This allows a Jupyter-style interactive Python environment where variables (like parsed JSON arrays) persist in memory between LLM tool calls. The model can natively self-compact its context window (`clear_message_history`) without losing its extracted data.

## Base64 Injection Bypass
`ouros.SessionManager` currently lacks support for passing `external_functions` (like `read_file`).
**Fix:** We bypassed the sandbox limitations by having the host read the massive `traces.jsonl` file, encode it as a Base64 string, chunk it via `textwrap.wrap` (to prevent Ouros Rust parser "Column number overflow" panics), and inject it natively as a Python string literal into the Ouros REPL during initialization. This provides near-instant synchronous filesystem access inside the stateful sandbox.

## PyO3 `!Send` Panic
The underlying Ouros `PySessionManager` object is not thread-safe (`!Send`), which caused panics when Pydantic AI automatically offloaded synchronous Python tools to `anyio` worker threads.
**Fix:** We declared the `run_python_repl` tool as an `async def`. Pydantic AI detects `async` functions and executes them directly on the main asyncio event loop, safely keeping the Ouros Rust session bound to its original thread.
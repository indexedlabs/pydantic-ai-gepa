# Repository Guidelines

## Mighty Workflow
- Use Mighty as the source of truth for spec/task/decision tracking and evidence links.
- Run `mt prime` at the start of each session and again after context loss (for example compaction, `/clear`, or session resume).
- Keep this file concise; use `mt prime` output for current workflow details and command guidance.

## Project Structure & Module Organization
- Core library lives in `src/pydantic_ai_gepa/`, organized by responsibility (`runner.py` exposes the high-level optimization API, `components.py` handles candidate transforms, `cache.py` manages GEPA caching, etc.).
- Example agents and walkthrough scripts sit in `examples/`, and experimental runs land in `optimization_results/` for posterity.
- Tests reside in `tests/` with fixtures in `tests/conftest.py`; favor mirroring module names (`test_signature_agent.py`, `test_cache.py`).
- Packaging metadata is defined in `pyproject.toml`; the repo uses the `uv` workflow (`uv.lock`) instead of ad‑hoc virtualenvs.

## Build, Test, and Development Commands
- `uv sync --all-extras` installs the project plus dev dependencies listed under `[dependency-groups.dev]`.
- `uv run pytest` executes the full test suite; add `-k pattern` to focus on a module (`uv run pytest -k signature`).
- `uv run python examples/classification.py` runs the end-to-end GEPA prompt optimization example; prefer `uv run` so dependencies resolve consistently.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents and snake_case for functions, module-level symbols, and filenames; keep classes in PascalCase.
- Preserve the existing type-hinted style—public APIs pass strongly-typed sequences (e.g., `Sequence[Case[InputT, OutputT, MetadataT]]`) and explicit `Model | KnownModelName` unions.
- Module docstrings summarize purpose; add short comments only where control flow is non-obvious (see `runner.py` contextmanagers for tone).

## Testing Guidelines
- Use `pytest` with inline snapshots where appropriate (dependency `inline-snapshot` is available); prefer async-aware tests via `pytest-asyncio` when touching async agents.
- Name new tests `test_<feature>.py` and group fixtures/utilities in `tests/conftest.py`.
- Run targeted coverage with `uv run pytest --cov=src/pydantic_ai_gepa --cov-report=term-missing` before large refactors; aim to keep coverage flat or higher.

## Commit & Pull Request Guidelines
- Commit messages follow a concise imperative style (e.g., "Fix cache key", "Include optimization results as an example"); keep to ~50 characters in the summary where possible.
- Pull requests should describe the agent scenario being improved, list key commands used for validation, and link related issues or experiment IDs under `optimization_results/`.
- Include screenshots or textual diffs for major GEPA runs, especially when changing prompt templates or optimization budgets.

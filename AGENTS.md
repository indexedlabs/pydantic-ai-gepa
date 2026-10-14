# Repository Guidelines

## Specs and tasks
- Specs and decisions for this library live in the Mighty Next graph, not in this repository. Before changing behavior, find the pages that own the code you are about to touch with `mighty invoke mighty.graph/governing --json '{"paths":["<repo-relative path>"]}'`, or browse the subtree starting at `specs/pydantic-ai-gepa/overview.md`. Tasks live in Linear (Indexed Labs); the orchestrator that dispatched you owns both.
- The v1 records under `.mighty/` (such as `.mighty/loro/` and `.mighty/config.json`) are a read-only historical record; do not write to them (`mt` is no longer installed for this repo). `.mighty/checks/` is active configuration for the hosted review checks and is edited like any other file.

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

## Review and merge
- Open pull requests with the `mighty-review` label (`gh pr create --label mighty-review`). It starts the two hosted reviews, `mighty review / claude` and `mighty review / codex`, on every pushed head.
- A pull request merges only when CI is green and both hosted review checks pass, or each of their findings is fixed or answered with a reason in a PR comment. Read the review bodies (`gh api repos/indexedlabs/pydantic-ai-gepa/pulls/<n>/reviews`) after every push before calling a PR ready.
- A hosted check that ended without a verdict (for example "Review did not finish") is re-run by removing and re-adding the `mighty-review` label, never waived.
- Squash-merge at the head you checked: `gh pr merge <n> --squash --match-head-commit <sha>`.

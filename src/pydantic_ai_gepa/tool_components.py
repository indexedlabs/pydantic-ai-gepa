"""Utilities for exposing and applying tool components to GEPA."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Iterable, Iterator, Mapping, cast

from pydantic_ai import RunContext
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.tools import ToolDefinition

from .gepa_graph.models import CandidateMap, ComponentValue, candidate_texts

ToolCandidate = dict[str, str]


def _unwrap_agent(agent: AbstractAgent[Any, Any]) -> AbstractAgent[Any, Any]:
    """Return the innermost agent (unwrap WrapperAgent layers)."""
    current = agent
    while isinstance(current, WrapperAgent):
        current = current.wrapped
    return current


def _description_key(tool_name: str) -> str:
    return f"tool:{tool_name}:description"


def _output_description_key(tool_name: str) -> str:
    return f"output:{tool_name}:description"


def _format_path(path: tuple[str, ...]) -> str:
    formatted: list[str] = []
    for segment in path:
        if segment == "[]":
            if not formatted:
                formatted.append("[]")
            else:
                formatted[-1] = f"{formatted[-1]}[]"
        else:
            formatted.append(segment)
    return ".".join(formatted)


def _parameter_key(tool_name: str, path: tuple[str, ...]) -> str:
    return f"tool:{tool_name}:param:{_format_path(path)}"


def _output_parameter_key(tool_name: str, path: tuple[str, ...]) -> str:
    return f"output:{tool_name}:param:{_format_path(path)}"


def _iter_schema_descriptions(
    schema: Any, path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], str]]:
    """Yield (path, description) pairs for a tool parameter schema."""
    if not isinstance(schema, dict):
        return

    description = schema.get("description")
    if isinstance(description, str) and description.strip() and path:
        yield path, description

    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties") or {}
        if isinstance(properties, dict):
            for name, subschema in properties.items():
                if isinstance(subschema, dict):
                    yield from _iter_schema_descriptions(subschema, path + (name,))
    if schema_type == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            yield from _iter_schema_descriptions(items, path + ("[]",))


def _set_schema_description(
    schema: dict[str, Any], path: tuple[str, ...], value: str
) -> bool:
    """Set a description on a copied schema, returning True if modified."""
    target: dict[str, Any] | None = schema
    for segment in path:
        if target is None:
            return False
        if segment == "[]":
            next_target = target.get("items")
            if not isinstance(next_target, dict):
                return False
            target = next_target
        else:
            properties = target.get("properties")
            if not isinstance(properties, dict):
                return False
            next_target = properties.get(segment)
            if not isinstance(next_target, dict):
                return False
            target = next_target

    if target is None:
        return False

    current = target.get("description")
    if current == value:
        return False
    target["description"] = value
    return True


@dataclass
class ToolMetadata:
    """Collected metadata for an individual tool."""

    name: str
    description_key: str | None
    parameter_paths: dict[str, tuple[str, ...]]


class ToolComponentCatalog:
    """Tracks tool component metadata and seed values."""

    def __init__(self) -> None:
        self._seed_components: dict[str, str] = {}
        self._metadata: dict[str, ToolMetadata] = {}

    def ingest(self, tool_defs: Iterable[ToolDefinition]) -> None:
        """Add tool definitions to the catalog."""
        for tool_def in tool_defs:
            metadata = self._describe_tool(tool_def)
            if metadata is None:
                continue
            self._metadata[metadata.name] = metadata

    def _describe_tool(self, tool_def: ToolDefinition) -> ToolMetadata | None:
        description_key: str | None = None
        parameter_paths: dict[str, tuple[str, ...]] = {}

        if isinstance(tool_def.description, str) and tool_def.description.strip():
            description_key = _description_key(tool_def.name)
            self._seed_components.setdefault(description_key, tool_def.description)

        for path, desc in _iter_schema_descriptions(tool_def.parameters_json_schema):
            key = _parameter_key(tool_def.name, path)
            parameter_paths[key] = path
            self._seed_components.setdefault(key, desc)

        if not description_key and not parameter_paths:
            return None

        return ToolMetadata(
            name=tool_def.name,
            description_key=description_key,
            parameter_paths=parameter_paths,
        )

    def seed_snapshot(self) -> dict[str, str]:
        """Return a copy of the seed component values."""
        return dict(self._seed_components)

    def component_keys(self) -> list[str]:
        """Return the ordered list of component keys."""
        return list(self._seed_components.keys())

    def metadata_for(self, tool_name: str) -> ToolMetadata | None:
        """Return metadata for a specific tool if available."""
        return self._metadata.get(tool_name)


class OutputToolComponentCatalog(ToolComponentCatalog):
    """Catalog for output tool components (mirrors ToolComponentCatalog)."""

    def _describe_tool(self, tool_def: ToolDefinition) -> ToolMetadata | None:  # type: ignore[override]
        # Reuse the same metadata structure but with output-prefixed keys
        description_key: str | None = None
        parameter_paths: dict[str, tuple[str, ...]] = {}

        if isinstance(tool_def.description, str) and tool_def.description.strip():
            description_key = _output_description_key(tool_def.name)
            self._seed_components.setdefault(description_key, tool_def.description)

        for path, desc in _iter_schema_descriptions(tool_def.parameters_json_schema):
            key = _output_parameter_key(tool_def.name, path)
            parameter_paths[key] = path
            self._seed_components.setdefault(key, desc)

        if not description_key and not parameter_paths:
            return None

        return ToolMetadata(
            name=tool_def.name,
            description_key=description_key,
            parameter_paths=parameter_paths,
        )


class ToolOptimizationManager:
    """Manage tool component extraction and candidate application."""

    def __init__(
        self,
        agent: AbstractAgent[Any, Any],
        *,
        allowed_tools: set[str] | None = None,
    ) -> None:
        self._base_agent = _unwrap_agent(agent)
        self._base_prepare = getattr(self._base_agent, "_prepare_tools", None)
        self._candidate_var: contextvars.ContextVar[ToolCandidate | None] = (
            contextvars.ContextVar("gepa_tool_candidate", default=None)
        )
        self._catalog = ToolComponentCatalog()
        self._latest_builtin_tools: list[AbstractBuiltinTool] = []
        # When None, all tools are optimized. When a set, only those tools are optimized.
        self._allowed_tools: set[str] | None = allowed_tools

        # Install wrapper only once.
        if getattr(self._base_agent, "_gepa_tool_prepare_wrapper", None) is None:
            setattr(
                self._base_agent, "_gepa_tool_prepare_wrapper", self._prepare_wrapper
            )
            self._base_agent._prepare_tools = self._prepare_wrapper  # type: ignore[assignment]

    def allow_tool(self, tool_name: str) -> None:
        """Add a tool to the allowed set for optimization.

        If the manager was created with allowed_tools=None (optimize all),
        this creates a new set with just this tool.
        """
        if self._allowed_tools is None:
            self._allowed_tools = {tool_name}
        else:
            self._allowed_tools.add(tool_name)

    def get_seed_components(self) -> dict[str, str]:
        """Return cached seed components, collecting them if necessary."""
        return self._catalog.seed_snapshot()

    def get_component_keys(self) -> list[str]:
        """Return all known component keys."""
        return self._catalog.component_keys()

    def record_model_request(
        self,
        *,
        function_tools: Iterable[ToolDefinition] | None = None,
        builtin_tools: Iterable[AbstractBuiltinTool] | None = None,
    ) -> None:
        """Update internal state from a completed ModelRequest."""
        if function_tools:
            self._catalog.ingest(function_tools)
        if builtin_tools:
            self._latest_builtin_tools = list(builtin_tools)

    def latest_builtin_tools(self) -> list[AbstractBuiltinTool]:
        """Return the most recent builtin tools observed."""
        return list(self._latest_builtin_tools)

    @contextmanager
    def candidate_context(
        self, candidate: CandidateMap | dict[str, str] | None
    ) -> Iterator[None]:
        """Context manager to apply a candidate during tool preparation."""
        candidate_map = self._normalize_candidate(candidate)
        candidate_text = candidate_texts(candidate_map)
        filtered = self._filter_candidate(candidate_text or None)
        token = self._candidate_var.set(filtered)
        try:
            yield
        finally:
            self._candidate_var.reset(token)

    def _normalize_candidate(
        self, candidate: CandidateMap | Mapping[str, ComponentValue | str] | None
    ) -> CandidateMap:
        if candidate is None:
            return {}
        if isinstance(candidate, dict) and all(
            isinstance(value, ComponentValue) for value in candidate.values()
        ):
            return cast(CandidateMap, candidate)

        normalized: CandidateMap = {}
        for name, value in candidate.items():
            if isinstance(value, ComponentValue):
                normalized[name] = value
            else:
                normalized[name] = ComponentValue(name=name, text=str(value))

        return normalized

    def _filter_candidate(
        self, candidate: dict[str, str] | None
    ) -> ToolCandidate | None:
        if not candidate:
            return None
        filtered: dict[str, str] = {}
        for key, value in candidate.items():
            if not key.startswith("tool:"):
                continue
            # Extract tool name from key like "tool:search_examples:description"
            parts = key.split(":", 2)
            if len(parts) < 2:
                continue
            tool_name = parts[1]
            # If allowed_tools is set, only include tools in that set
            if self._allowed_tools is not None and tool_name not in self._allowed_tools:
                continue
            filtered[key] = value
        return filtered or None

    async def _prepare_wrapper(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition] | None:
        prepared = tool_defs
        if self._base_prepare:
            prepared_result = await self._base_prepare(ctx, tool_defs)
            if prepared_result is None:
                return None
            prepared = prepared_result

        self._catalog.ingest(prepared)

        candidate = self._candidate_var.get()
        if not candidate:
            return prepared

        modified: list[ToolDefinition] = []
        changed = False
        for tool_def in prepared:
            new_def = self._apply_candidate_to_tool(tool_def, candidate)
            if new_def is not tool_def:
                changed = True
            modified.append(new_def)

        return modified if changed else prepared

    def _apply_candidate_to_tool(
        self, tool_def: ToolDefinition, candidate: ToolCandidate
    ) -> ToolDefinition:
        metadata = self._catalog.metadata_for(tool_def.name)
        if metadata is None:
            return tool_def

        updates: dict[str, Any] = {}

        if metadata.description_key:
            raw_value = candidate.get(metadata.description_key)
            if raw_value is not None:
                new_description = str(raw_value)
                if tool_def.description != new_description:
                    updates["description"] = new_description

        schema_copy: dict[str, Any] | None = None
        schema_changed = False

        for key, path in metadata.parameter_paths.items():
            if key not in candidate:
                continue
            raw_value = candidate[key]
            if raw_value is None:
                continue
            new_description = str(raw_value)
            if schema_copy is None:
                schema_copy = deepcopy(tool_def.parameters_json_schema)
            if _set_schema_description(schema_copy, path, new_description):
                schema_changed = True

        if schema_changed and schema_copy is not None:
            updates["parameters_json_schema"] = schema_copy

        if updates:
            return replace(tool_def, **updates)
        return tool_def


class OutputToolOptimizationManager:
    """Manage output tool component extraction and candidate application."""

    def __init__(self, agent: AbstractAgent[Any, Any]) -> None:
        self._base_agent = _unwrap_agent(agent)
        self._base_prepare = getattr(self._base_agent, "_prepare_output_tools", None)
        self._candidate_var: contextvars.ContextVar[ToolCandidate | None] = (
            contextvars.ContextVar("gepa_output_tool_candidate", default=None)
        )
        self._catalog = OutputToolComponentCatalog()

        if getattr(self._base_agent, "_gepa_output_tool_prepare_wrapper", None) is None:
            setattr(
                self._base_agent,
                "_gepa_output_tool_prepare_wrapper",
                self._prepare_wrapper,
            )
            self._base_agent._prepare_output_tools = self._prepare_wrapper  # type: ignore[assignment]

    def get_seed_components(self) -> dict[str, str]:
        return self._catalog.seed_snapshot()

    def get_component_keys(self) -> list[str]:
        return self._catalog.component_keys()

    def record_model_request(
        self,
        *,
        output_tools: Iterable[ToolDefinition] | None = None,
    ) -> None:
        if output_tools:
            self._catalog.ingest(output_tools)

    def candidate_context(self, candidate: CandidateMap | None):
        candidate_map = self._normalize_candidate(candidate)
        filtered_candidate = self._filter_candidate(candidate_texts(candidate_map))
        token = self._candidate_var.set(filtered_candidate)

        @contextmanager
        def _ctx():
            try:
                yield
            finally:
                self._candidate_var.reset(token)

        return _ctx()

    def _normalize_candidate(
        self, candidate: CandidateMap | Mapping[str, ComponentValue | str] | None
    ) -> CandidateMap:
        if candidate is None:
            return {}
        if isinstance(candidate, dict) and all(
            isinstance(v, ComponentValue) for v in candidate.values()
        ):
            return cast(CandidateMap, candidate)

        normalized: CandidateMap = {}
        for name, value in candidate.items():
            if isinstance(value, ComponentValue):
                normalized[name] = value
            else:
                normalized[name] = ComponentValue(name=name, text=str(value))
        return normalized

    def _filter_candidate(
        self, candidate: dict[str, str] | None
    ) -> ToolCandidate | None:
        if not candidate:
            return None
        filtered = {
            key: value for key, value in candidate.items() if key.startswith("output:")
        }
        return filtered or None

    async def _prepare_wrapper(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition] | None:
        prepared = tool_defs
        if self._base_prepare:
            prepared_result = await self._base_prepare(ctx, tool_defs)
            if prepared_result is None:
                return None
            prepared = prepared_result

        self._catalog.ingest(prepared)

        candidate = self._candidate_var.get()
        if not candidate:
            return prepared

        modified: list[ToolDefinition] = []
        changed = False
        for tool_def in prepared:
            new_def = self._apply_candidate_to_tool(tool_def, candidate)
            if new_def is not tool_def:
                changed = True
            modified.append(new_def)

        return modified if changed else prepared

    def _apply_candidate_to_tool(
        self, tool_def: ToolDefinition, candidate: ToolCandidate
    ) -> ToolDefinition:
        metadata = self._catalog.metadata_for(tool_def.name)
        if metadata is None:
            return tool_def

        updates: dict[str, Any] = {}

        if metadata.description_key:
            raw_value = candidate.get(metadata.description_key)
            if raw_value is not None:
                new_description = str(raw_value)
                if tool_def.description != new_description:
                    updates["description"] = new_description

        schema_copy: dict[str, Any] | None = None
        schema_changed = False

        for key, path in metadata.parameter_paths.items():
            if key not in candidate:
                continue
            raw_value = candidate[key]
            if raw_value is None:
                continue
            new_description = str(raw_value)
            if schema_copy is None:
                schema_copy = deepcopy(tool_def.parameters_json_schema)
            if _set_schema_description(schema_copy, path, new_description):
                schema_changed = True

        if schema_changed and schema_copy is not None:
            updates["parameters_json_schema"] = schema_copy

        if updates:
            return replace(tool_def, **updates)
        return tool_def


def get_tool_optimizer(
    agent: AbstractAgent[Any, Any],
) -> ToolOptimizationManager | None:
    """Return the installed tool optimization manager for an agent, if any."""
    base_agent = _unwrap_agent(agent)
    manager = getattr(base_agent, "_gepa_tool_optimizer", None)
    if isinstance(manager, ToolOptimizationManager):
        return manager
    return None


def get_or_create_tool_optimizer(
    agent: AbstractAgent[Any, Any],
    *,
    allowed_tools: set[str] | None = None,
) -> ToolOptimizationManager:
    """Retrieve or attach a tool optimization manager to an agent.

    Args:
        agent: The agent to get or create the optimizer for.
        allowed_tools: If provided, only these tools will be optimized.
            If None, all tools are optimized.
            If a manager already exists, these tools are added to the allowed set.
    """
    base_agent = _unwrap_agent(agent)
    manager = getattr(base_agent, "_gepa_tool_optimizer", None)
    if isinstance(manager, ToolOptimizationManager):
        # If additional allowed_tools specified, add them to existing manager
        if allowed_tools is not None:
            for tool_name in allowed_tools:
                manager.allow_tool(tool_name)
        return manager

    manager = ToolOptimizationManager(base_agent, allowed_tools=allowed_tools)
    setattr(base_agent, "_gepa_tool_optimizer", manager)
    initial = _collect_registered_tool_defs(base_agent)
    if initial:
        manager.record_model_request(function_tools=initial)
    return manager


def get_output_tool_optimizer(
    agent: AbstractAgent[Any, Any],
) -> OutputToolOptimizationManager | None:
    """Return the installed output-tool optimization manager for an agent, if any."""
    base_agent = _unwrap_agent(agent)
    manager = getattr(base_agent, "_gepa_output_tool_optimizer", None)
    if isinstance(manager, OutputToolOptimizationManager):
        return manager
    return None


def get_or_create_output_tool_optimizer(
    agent: AbstractAgent[Any, Any],
) -> OutputToolOptimizationManager:
    """Retrieve or attach an output-tool optimization manager to an agent."""
    base_agent = _unwrap_agent(agent)
    manager = getattr(base_agent, "_gepa_output_tool_optimizer", None)
    if isinstance(manager, OutputToolOptimizationManager):
        return manager

    manager = OutputToolOptimizationManager(base_agent)
    setattr(base_agent, "_gepa_output_tool_optimizer", manager)

    # Seed from existing output toolset when available
    toolset = getattr(base_agent, "_output_toolset", None)
    tool_defs = getattr(toolset, "_tool_defs", None)
    if tool_defs:
        manager.record_model_request(output_tools=tool_defs)
    return manager


def _collect_registered_tool_defs(
    agent: AbstractAgent[Any, Any],
) -> list[ToolDefinition]:
    """Return ToolDefinition objects for currently registered function tools."""
    toolset = getattr(agent, "_function_toolset", None)
    tools = getattr(toolset, "tools", None) if toolset is not None else None
    if not isinstance(tools, dict):
        return []

    definitions: list[ToolDefinition] = []
    for tool in tools.values():
        schema = getattr(tool, "function_schema", None)
        json_schema = getattr(schema, "json_schema", None)
        if not isinstance(json_schema, dict):
            continue
        description = getattr(tool, "description", None)
        name = getattr(tool, "name", getattr(tool, "__name__", "tool"))
        definitions.append(
            ToolDefinition(
                name=name,
                description=description,
                parameters_json_schema=json_schema,
            )
        )
    return definitions

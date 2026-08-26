"""Map between agent prompt components and GEPA candidates."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic_ai.agent.wrapper import WrapperAgent

from .gepa_graph.models import CandidateMap, ComponentValue
from .input_type import InputSpec, build_input_spec
from .signature_agent import SignatureAgent
from .tool_components import (
    build_candidate_capability,
    get_tool_optimizer,
    get_output_tool_optimizer,
)

if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent


class AppliedCandidateAgent(WrapperAgent[Any, Any]):
    """Agent view that injects an optimized candidate as a run capability."""

    def __init__(
        self,
        wrapped: AbstractAgent[Any, Any],
        candidate: CandidateMap,
    ) -> None:
        super().__init__(wrapped)
        self._candidate_capability = build_candidate_capability(wrapped, candidate)

    def iter(self, *args: Any, **kwargs: Any):
        """Forward a run while adding the candidate capability."""
        capabilities: list[Any] = list(kwargs.pop("capabilities", None) or ())
        if self._candidate_capability is not None:
            capabilities.append(self._candidate_capability)
        active_override = _active_instruction_override(self.wrapped)
        if active_override is not None:
            instructions = _normalized_instructions(active_override)
            for capability in capabilities:
                instructions.extend(
                    _normalized_instructions(capability.get_instructions())
                )
            instructions.extend(
                _normalized_instructions(kwargs.pop("instructions", None))
            )
            return self._iter_with_instruction_override(
                args,
                kwargs,
                tuple(capabilities) or None,
                instructions,
            )
        return self.wrapped.iter(
            *args,
            capabilities=tuple(capabilities) or None,
            **kwargs,
        )

    @asynccontextmanager
    async def _iter_with_instruction_override(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        capabilities: tuple[Any, ...] | None,
        instructions: list[Any],
    ):
        override: Any = (
            instructions[0] if len(instructions) == 1 else tuple(instructions)
        )
        target_agent = _base_agent(self.wrapped)
        with target_agent.override(instructions=override):
            async with self.wrapped.iter(
                *args,
                instructions=None,
                capabilities=capabilities,
                **kwargs,
            ) as run:
                yield run

    async def run_signature(self, *args: Any, **kwargs: Any):
        """Forward SignatureAgent's structured run API when available."""
        if not isinstance(self.wrapped, SignatureAgent):
            raise AttributeError("Wrapped agent does not support run_signature()")
        return await self.wrapped.run_signature(*args, **kwargs)

    def run_signature_sync(self, *args: Any, **kwargs: Any):
        """Forward SignatureAgent's synchronous structured run API."""
        if not isinstance(self.wrapped, SignatureAgent):
            raise AttributeError("Wrapped agent does not support run_signature_sync()")
        return self.wrapped.run_signature_sync(*args, **kwargs)


def _normalized_instructions(instructions: Any) -> list[Any]:
    if instructions is None:
        return []
    if isinstance(instructions, Sequence) and not isinstance(instructions, str):
        return list(instructions)
    return [instructions]


def _base_agent(agent: AbstractAgent[Any, Any]) -> AbstractAgent[Any, Any]:
    while isinstance(agent, WrapperAgent):
        agent = agent.wrapped
    return agent


def _active_instruction_override(agent: AbstractAgent[Any, Any]) -> Any | None:
    base_agent = _base_agent(agent)
    override_manager = getattr(base_agent, "_override_instructions", None)
    if override_manager is None:
        return None
    override = override_manager.get()
    if override is None:
        return None
    return override.value


def ensure_component_values(
    candidate: Mapping[str, ComponentValue | str] | None,
) -> CandidateMap:
    """Coerce raw values into ComponentValue instances."""
    if not candidate:
        return {}
    result: CandidateMap = {}
    for name, value in candidate.items():
        if isinstance(value, ComponentValue):
            result[name] = value
        else:
            result[name] = ComponentValue(name=name, text=str(value))
    return result


def _stringify_component_value(value: Any) -> str:
    """Render arbitrary component content as a string."""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "\n".join(str(part) for part in value)
    return str(value)


def extract_seed_candidate(
    agent: AbstractAgent[Any, Any], *, optimize_output_type: bool = False
) -> CandidateMap:
    """Extract the current prompts from an agent as a GEPA candidate.

    Args:
        agent: The agent to extract prompts from.

    Returns:
        A dictionary mapping component names to their text values.
        - 'instructions': The effective instructions (combining literal and functions)
    """
    candidate: CandidateMap = {}

    target_agent = agent
    if isinstance(agent, WrapperAgent):
        target_agent = agent.wrapped

    # Extract instructions
    # The candidate only carries literal string instructions — GEPA mutates
    # text, not callables. Function-based instructions registered via
    # `@agent.instructions` stay attached to the agent and are re-executed at
    # rollout time by SignatureAgent (or by pydantic-ai for plain agents),
    # so don't stringify them here (that would render `<function ... at 0x...>`
    # into the prompt).
    raw_instructions = getattr(target_agent, "_instructions", None)
    if raw_instructions:
        literal_parts = [item for item in raw_instructions if isinstance(item, str)]
        candidate["instructions"] = ComponentValue(
            name="instructions",
            text="\n".join(literal_parts),
        )
    else:
        candidate["instructions"] = ComponentValue(
            name="instructions",
            text="",
        )

    if isinstance(agent, SignatureAgent):
        if agent.optimize_tools:
            for key, text in agent.get_tool_components().items():
                candidate[key] = ComponentValue(
                    name=key, text=_stringify_component_value(text)
                )
        else:
            # SignatureAgent with optimize_tools=False: check if adapter installed
            # an optimizer on the wrapped agent
            optimizer = get_tool_optimizer(agent.wrapped)
            if optimizer:
                for key, text in optimizer.get_seed_components().items():
                    candidate[key] = ComponentValue(
                        name=key, text=_stringify_component_value(text)
                    )
    else:
        optimizer = get_tool_optimizer(agent)
        if optimizer:
            for key, text in optimizer.get_seed_components().items():
                candidate[key] = ComponentValue(
                    name=key, text=_stringify_component_value(text)
                )

    if optimize_output_type:
        output_optimizer = get_output_tool_optimizer(agent)
        if output_optimizer:
            for key, text in output_optimizer.get_seed_components().items():
                candidate[key] = ComponentValue(
                    name=key, text=_stringify_component_value(text)
                )

    return candidate


@contextmanager
def apply_candidate_to_agent(
    agent: AbstractAgent[Any, Any],
    candidate: CandidateMap | None,
) -> Iterator[None]:
    """Apply a GEPA candidate to an agent via override().

    This returns a context manager that temporarily applies the candidate
    prompts to the agent.

    Args:
        agent: The agent to apply prompts to.
        candidate: The candidate mapping component names to text.

    Returns:
        A context manager for the temporary override.
    """
    candidate_map: CandidateMap
    if candidate is None:
        candidate_map = {}
    elif isinstance(candidate, dict):
        candidate_map = candidate
    else:
        candidate_map = dict(candidate)

    instructions_value = candidate_map.get("instructions")
    instructions = instructions_value.text if instructions_value else None

    target_agent = agent
    if isinstance(agent, WrapperAgent):
        target_agent = agent.wrapped

    optimizer = get_tool_optimizer(agent)
    output_optimizer = get_output_tool_optimizer(agent)

    # `@agent.instructions` callbacks live alongside literal strings in
    # `_instructions`. When we override with the candidate text we'd otherwise
    # drop the callbacks; preserve them so per-run dynamic context still
    # reaches the model.
    raw_instructions = getattr(target_agent, "_instructions", None) or []
    instruction_callbacks: list[Any] = [
        item for item in raw_instructions if not isinstance(item, str)
    ]

    with ExitStack() as stack:
        if optimizer:
            stack.enter_context(optimizer.candidate_context(candidate_map))
        if output_optimizer:
            stack.enter_context(output_optimizer.candidate_context(candidate_map))
        override_value: list[Any] = []
        if instructions:
            override_value.append(instructions)
        override_value.extend(instruction_callbacks)
        override_value.extend(
            _normalized_instructions(target_agent.root_capability.get_instructions())
        )
        if override_value:
            override_payload: Any = (
                override_value[0] if len(override_value) == 1 else tuple(override_value)
            )
            stack.enter_context(target_agent.override(instructions=override_payload))
        yield


@contextmanager
def applied_candidate_agent(
    agent: AbstractAgent[Any, Any],
    candidate: CandidateMap | None,
) -> Iterator[AppliedCandidateAgent]:
    """Yield an agent view that applies every candidate component per run."""
    candidate_map = candidate or {}
    with apply_candidate_to_agent(agent, candidate_map):
        yield AppliedCandidateAgent(agent, candidate_map)


def get_component_names(
    agent: AbstractAgent[Any, Any], *, optimize_output_type: bool = False
) -> list[str]:
    """Get the list of optimizable component names for an agent.

    Args:
        agent: The agent to inspect.

    Returns:
        List of component names that can be optimized.
    """
    components: list[str] = ["instructions"]

    optimizer = get_tool_optimizer(agent)
    if isinstance(agent, SignatureAgent) and not agent.optimize_tools:
        optimizer = None

    if optimizer:
        components.extend(optimizer.get_component_keys())

    if optimize_output_type:
        output_optimizer = get_output_tool_optimizer(agent)
        if output_optimizer:
            components.extend(output_optimizer.get_component_keys())

    # Preserve order but ensure uniqueness
    seen: set[str] = set()
    deduped: list[str] = []
    for component in components:
        if component not in seen:
            deduped.append(component)
            seen.add(component)

    return deduped


def validate_components(
    agent: AbstractAgent[Any, Any],
    components: Sequence[str],
    *,
    optimize_output_type: bool = False,
) -> list[str]:
    """Validate that the requested components exist in the agent.

    Args:
        agent: The agent to check against.
        components: The requested component names.

    Returns:
        The validated list of component names.

    Raises:
        ValueError: If any component doesn't exist in the agent.
    """
    available = set(
        get_component_names(agent, optimize_output_type=optimize_output_type)
    )
    requested = set(components)

    invalid = requested - available
    if invalid:
        raise ValueError(
            f"Components {invalid} not found in agent. Available components: {sorted(available)}"
        )

    return list(components)


def extract_seed_candidate_with_input_type(
    agent: AbstractAgent[Any, Any],
    input_type: InputSpec[BaseModel] | None = None,
    *,
    optimize_output_type: bool = False,
) -> CandidateMap:
    """Extract prompts from an agent and optional input specification as a GEPA candidate.

    Args:
        agent: The agent to extract prompts from.
        input_type: Optional structured input specification to extract from.

    Returns:
        Combined dictionary of all components and their initial text.
    """
    candidate: CandidateMap = {}

    # Extract from agent
    candidate.update(
        extract_seed_candidate(agent, optimize_output_type=optimize_output_type)
    )

    # Extract from signature if provided
    if input_type:
        spec = build_input_spec(input_type)
        for key, text in spec.get_gepa_components().items():
            candidate[key] = ComponentValue(
                name=key, text=_stringify_component_value(text)
            )

    return candidate


@contextmanager
def apply_candidate_to_agent_and_input_type(
    candidate: CandidateMap | None,
    agent: AbstractAgent[Any, Any],
    input_type: InputSpec[BaseModel] | None = None,
) -> Iterator[None]:
    """Apply a GEPA candidate to an agent and optionally an input specification.

    This context manager temporarily applies the candidate to the agent
    (via override()) and optionally to a structured input specification.

    Args:
        candidate: The candidate mapping component names to text.
        agent: The agent to apply prompts to.
        input_type: Optional structured input specification to apply to.

    Yields:
        None while the candidate is applied.
    """
    from contextlib import ExitStack

    with ExitStack() as stack:
        # Apply to agent
        stack.enter_context(apply_candidate_to_agent(agent, candidate))

        # Apply to input specification if provided
        if input_type:
            spec = build_input_spec(input_type)
            candidate_map = candidate if candidate is not None else {}
            stack.enter_context(spec.apply_candidate(candidate_map))

        yield

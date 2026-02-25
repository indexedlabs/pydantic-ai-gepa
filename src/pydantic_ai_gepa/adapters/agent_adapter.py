"""GEPA adapter for pydantic-ai agents with single signature optimization."""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
import json
import os
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import logfire

from pydantic import BaseModel
from pydantic_ai import capture_run_messages, usage as _usage
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.tools import ToolDefinition

from ..cache import CacheManager
from ..gepa_graph.proposal.student_tools import create_example_search_tool
from ..gepa_graph.proposal.student_tools import create_skills_toolset
from ..exceptions import UsageBudgetExceeded
from ..components import (
    apply_candidate_to_agent,
    extract_seed_candidate_with_input_type,
)
from ..evaluation_models import EvaluationBatch
from ..gepa_graph.models import CandidateMap, candidate_texts
from ..inspection import InspectionAborted
from ..input_type import BoundInputSpec, InputSpec, build_input_spec
from ..signature_agent import SignatureAgent
from ..skill_components import apply_candidate_to_skills
from ..skills import SkillsFS
from ..skills.models import SkillCapability
from ..skills.search import SkillsSearchProvider
from ..tool_components import (
    get_or_create_output_tool_optimizer,
    get_or_create_tool_optimizer,
    get_output_tool_optimizer,
    get_tool_optimizer,
)
from ..types import (
    MetadataWithMessageHistory,
    MetricResult,
    RolloutOutput,
    Trajectory,
)
from ..adapter import Adapter, ReflectiveDataset, SharedReflectiveDataset
from pydantic_evals import Case


if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent
    from ..gepa_graph.example_bank import InMemoryExampleBank

from pydantic_ai.exceptions import ToolRetryError as _ToolRetryError
from pydantic_ai.exceptions import UsageLimitExceeded as _UsageLimitExceeded

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
MetadataT = TypeVar("MetadataT")

ErrorKind = Literal["tool", "system"]

_LIBRARY_MODULE_PREFIXES = ("pydantic_ai", "pydantic_ai_gepa", "pydantic_graph")
_LIBRARY_PATH_MARKERS = tuple(
    f"{os.sep}{name}{os.sep}"
    for name in ("pydantic_ai", "pydantic_ai_gepa", "pydantic_graph")
)
_TOOL_ERROR_TYPES: tuple[type[BaseException], ...] = (_ToolRetryError,)


def _traceback_originates_from_library(tb: TracebackType | None) -> bool:
    """Return True if the deepest frame in the traceback belongs to known library code."""
    last_tb = tb
    while last_tb and last_tb.tb_next:
        last_tb = last_tb.tb_next
    if last_tb is None:
        return False
    frame = last_tb.tb_frame
    module_name = frame.f_globals.get("__name__")
    if isinstance(module_name, str) and module_name.startswith(
        _LIBRARY_MODULE_PREFIXES
    ):
        return True
    filename = frame.f_code.co_filename
    if filename and any(marker in filename for marker in _LIBRARY_PATH_MARKERS):
        return True
    return False


def _classify_exception(exc: BaseException) -> ErrorKind:
    """Differentiate tool call failures from systemic/library issues."""
    if _TOOL_ERROR_TYPES and isinstance(exc, _TOOL_ERROR_TYPES):
        return "tool"

    module_name = type(exc).__module__
    if isinstance(module_name, str) and module_name.startswith(
        _LIBRARY_MODULE_PREFIXES
    ):
        return "system"

    if _traceback_originates_from_library(exc.__traceback__):
        return "system"
    return "tool"


def _truncate_text(value: str, limit: int = 2000) -> str:
    if len(value) <= limit:
        return value
    trimmed = value[:limit]
    omitted = len(value) - limit
    return f"{trimmed}... [truncated {omitted} chars]"


def _serialize_for_reflection(obj: Any) -> Any:
    """Serialize an object for inclusion in reflection records.

    Handles BaseModel, dataclasses, dicts, and other common types.
    """
    if obj is None:
        return None
    if isinstance(obj, (ImageUrl, AudioUrl, VideoUrl, DocumentUrl, BinaryContent)):
        return obj
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, Mapping):
        return {k: _serialize_for_reflection(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_serialize_for_reflection(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    # Fallback to string representation
    return str(obj)


def _compact_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _safe_getattr(obj: Any, attr: str) -> Any | None:
    try:
        return getattr(obj, attr)
    except AttributeError:
        return None
    except Exception:
        logfire.debug("Failed to read attribute", attr=attr, obj=obj, exc_info=True)
        return None


def _serialize_model_reference(model_obj: Any) -> str:
    if isinstance(model_obj, str):
        return model_obj

    parts: list[str] = []

    model_name = _safe_getattr(model_obj, "model_name")
    if model_name:
        parts.append(str(model_name))

    class_path = f"{model_obj.__class__.__module__}.{model_obj.__class__.__qualname__}"
    parts.append(class_path)

    provider_name = _safe_getattr(model_obj, "provider_name")
    if provider_name:
        parts.append(f"provider:{provider_name}")
    else:
        provider = _safe_getattr(model_obj, "provider")
        if provider is not None:
            provider_class = (
                f"{provider.__class__.__module__}.{provider.__class__.__qualname__}"
            )
            parts.append(f"provider:{provider_class}")

    settings = _safe_getattr(model_obj, "settings")
    if settings:
        parts.append(f"settings:{CacheManager._serialize_for_key(settings)}")

    profile = _safe_getattr(model_obj, "profile")
    if profile:
        parts.append(f"profile:{CacheManager._serialize_for_key(profile)}")

    return "|".join(parts)


def _derive_model_identifier(agent: "AbstractAgent[Any, Any]") -> str | None:
    model_obj = _safe_getattr(agent, "model")
    if model_obj is None:
        return None
    return _serialize_model_reference(model_obj)


def _timestamp_iso(timestamp: Any) -> str | None:
    return timestamp.isoformat() if hasattr(timestamp, "isoformat") else None


def _describe_binary_content(content: BinaryContent) -> str:
    size = len(content.data) if getattr(content, "data", None) else 0
    media_type = getattr(content, "media_type", "unknown")
    identifier = getattr(content, "identifier", None)
    identifier_str = f", id={identifier}" if identifier else ""
    return f"[binary media_type={media_type}{identifier_str}, bytes={size}]"


def _describe_user_content_item(item: UserContent) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, (ImageUrl, AudioUrl, DocumentUrl, VideoUrl)):
        meta: list[str] = [f"url={item.url}"]
        media_type = getattr(item, "media_type", None)
        if media_type:
            meta.append(f"media_type={media_type}")
        if getattr(item, "force_download", False):
            meta.append("force_download=true")
        return f"[{item.kind} {' '.join(meta)}]"
    if isinstance(item, BinaryContent):
        return _describe_binary_content(item)
    return repr(item)


def _serialize_user_prompt_content(content: str | Sequence[UserContent]) -> str:
    if isinstance(content, str):
        return content
    described = [_describe_user_content_item(item) for item in content]
    return "\n".join(described)


def _serialize_tool_return(
    part: ToolReturnPart | BuiltinToolReturnPart,
) -> dict[str, Any]:
    content_str = _truncate_text(part.model_response_str())
    serialized = {
        "type": "tool_return",
        "tool_name": part.tool_name,
        "content": content_str,
        "tool_call_id": part.tool_call_id,
        "timestamp": _timestamp_iso(part.timestamp),
    }
    metadata = getattr(part, "metadata", None)
    if metadata is not None:
        serialized["metadata"] = repr(metadata)
    provider_name = getattr(part, "provider_name", None)
    if provider_name:
        serialized["provider_name"] = provider_name
    return _compact_dict(serialized)


def _serialize_retry_part(part: RetryPromptPart) -> dict[str, Any]:
    if isinstance(part.content, str):
        reason = part.content
    else:
        reason = json.dumps(part.content, ensure_ascii=False, default=str)
    return _compact_dict(
        {
            "type": "retry_prompt",
            "tool_name": part.tool_name,
            "tool_call_id": part.tool_call_id,
            "content": _truncate_text(reason),
            "timestamp": _timestamp_iso(part.timestamp),
        }
    )


def _serialize_request_part(part: Any) -> dict[str, Any]:
    """Serialize a ModelRequest part."""
    if isinstance(part, SystemPromptPart):
        return _compact_dict(
            {
                "type": "system_prompt",
                "role": "system",
                "content": _truncate_text(part.content),
                "timestamp": _timestamp_iso(part.timestamp),
            }
        )
    if isinstance(part, UserPromptPart):
        return _compact_dict(
            {
                "type": "user_prompt",
                "role": "user",
                "content": _truncate_text(_serialize_user_prompt_content(part.content)),
                "timestamp": _timestamp_iso(part.timestamp),
            }
        )
    if isinstance(part, (ToolReturnPart, BuiltinToolReturnPart)):
        return _serialize_tool_return(part)
    if isinstance(part, RetryPromptPart):
        return _serialize_retry_part(part)
    return {
        "type": getattr(part, "part_kind", type(part).__name__),
        "repr": repr(part),
    }


def _serialize_response_part(part: Any) -> dict[str, Any]:
    """Serialize a ModelResponse part."""
    if isinstance(part, TextPart):
        return _compact_dict(
            {
                "type": "text",
                "role": "assistant",
                "content": _truncate_text(part.content),
                "id": part.id,
            }
        )
    if isinstance(part, ThinkingPart):
        return _compact_dict(
            {
                "type": "thinking",
                "role": "assistant",
                "content": _truncate_text(part.content),
                "id": part.id,
                "provider_name": part.provider_name,
            }
        )
    if isinstance(part, (ToolCallPart, BuiltinToolCallPart)):
        serialized = {
            "type": "tool_call",
            "role": "assistant",
            "tool_name": part.tool_name,
            "arguments": _truncate_text(part.args_as_json_str()),
            "tool_call_id": part.tool_call_id,
            "id": part.id,
        }
        provider_name = getattr(part, "provider_name", None)
        if provider_name:
            serialized["provider_name"] = provider_name
        return _compact_dict(serialized)
    if isinstance(part, BuiltinToolReturnPart):
        return _serialize_tool_return(part)
    if isinstance(part, FilePart):
        return _compact_dict(
            {
                "type": "file",
                "role": "assistant",
                "description": _describe_binary_content(part.content),
                "id": part.id,
                "provider_name": getattr(part, "provider_name", None),
            }
        )
    if hasattr(part, "content"):
        return _compact_dict(
            {
                "type": getattr(part, "part_kind", type(part).__name__),
                "role": "assistant",
                "content": _truncate_text(str(part.content)),
            }
        )
    return _compact_dict(
        {
            "type": getattr(part, "part_kind", type(part).__name__),
            "role": "assistant",
            "repr": repr(part),
        }
    )


def _serialize_model_message(
    message: ModelMessage,
    *,
    include_instructions: bool,
) -> dict[str, Any]:
    if isinstance(message, ModelRequest):
        data = {
            "kind": "request",
            "parts": [_serialize_request_part(part) for part in message.parts],
        }
        if include_instructions and message.instructions is not None:
            data["instructions"] = message.instructions
        return _compact_dict(data)
    if isinstance(message, ModelResponse):
        base: dict[str, Any] = {
            "kind": "response",
            "model_name": message.model_name,
            "provider_name": message.provider_name,
            "finish_reason": message.finish_reason,
            "timestamp": _timestamp_iso(message.timestamp),
            "parts": [_serialize_response_part(part) for part in message.parts],
        }
        if message.provider_response_id:
            base["provider_response_id"] = message.provider_response_id
        if message.provider_details:
            base["provider_details"] = message.provider_details
        if message.usage and hasattr(message.usage, "__dataclass_fields__"):
            base["usage"] = asdict(message.usage)
        return _compact_dict(base)
    return _compact_dict(
        {
            "kind": getattr(message, "kind", type(message).__name__),
            "repr": repr(message),
        }
    )


@dataclass
class AgentAdapterTrajectory(Trajectory):
    """Execution trajectory capturing the agent run for reflection."""

    messages: list[ModelMessage]
    final_output: Any
    instructions: str | None = None
    function_tools: list[ToolDefinition] | None = None
    output_tools: list[ToolDefinition] | None = None
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    case: Case[Any, Any, Any] | None = None
    metric_feedback: str | None = None
    metric_side_info: dict[str, Any] | None = None

    def _extract_user_content(self, part: UserPromptPart) -> str:
        if isinstance(part.content, str):
            return part.content
        if part.content:
            for content_item in part.content:
                if isinstance(content_item, str):
                    return content_item
            return "Multi-modal content"
        return "No content"

    def _extract_user_message(self) -> str | None:
        for msg in self.messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        return self._extract_user_content(part)
        return None

    def _extract_assistant_message(self) -> str | None:
        for msg in reversed(self.messages):
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        return part.content
        return None

    def _serialize_messages_with_instructions(self) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        instructions_recorded = False
        for message in self.messages:
            include_instructions = (
                isinstance(message, ModelRequest)
                and not instructions_recorded
                and getattr(message, "instructions", None) is not None
            )
            if include_instructions:
                instructions_recorded = True
            serialized.append(
                _serialize_model_message(
                    message,
                    include_instructions=include_instructions,
                )
            )
        return serialized

    def _serialize_tool_defs(
        self, tool_defs: list[ToolDefinition] | None
    ) -> list[dict[str, Any]] | None:
        """Serialize tool definitions (function or output) into JSON schema format."""
        if not tool_defs:
            return None

        serialized_tools: list[dict[str, Any]] = []
        for tool in tool_defs:
            tool_dict: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "parameters": tool.parameters_json_schema,
                },
            }
            if tool.description:
                tool_dict["function"]["description"] = tool.description
            if tool.kind != "function":
                tool_dict["kind"] = tool.kind
            serialized_tools.append(tool_dict)
        return serialized_tools

    @staticmethod
    def _serialized_tool_name(tool: Mapping[str, Any]) -> str | None:
        function_block = tool.get("function")
        if isinstance(function_block, Mapping):
            name = function_block.get("name")
            if isinstance(name, str) and name.strip():
                return name
        return None

    def _merge_serialized_tool_lists(
        self,
        primary: list[dict[str, Any]] | None,
        secondary: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        merged: dict[str, dict[str, Any]] = {}

        def _mutable_copy(tool: Mapping[str, Any]) -> dict[str, Any]:
            entry = dict(tool)
            fn_block = entry.get("function")
            if isinstance(fn_block, Mapping) and not isinstance(fn_block, dict):
                entry["function"] = dict(fn_block)
            return entry

        def _ingest(tool: Mapping[str, Any]) -> None:
            if not isinstance(tool, Mapping):
                return
            name = self._serialized_tool_name(tool)
            if not name:
                return
            existing = merged.get(name)
            if existing is None:
                merged[name] = _mutable_copy(tool)
                return

            if tool.get("kind") and not existing.get("kind"):
                existing["kind"] = tool["kind"]

            existing_fn = existing.get("function")
            candidate_fn = tool.get("function")
            if isinstance(existing_fn, dict) and isinstance(candidate_fn, Mapping):
                if not existing_fn.get("description") and candidate_fn.get(
                    "description"
                ):
                    existing_fn["description"] = candidate_fn["description"]
                if not existing_fn.get("parameters") and candidate_fn.get("parameters"):
                    existing_fn["parameters"] = candidate_fn["parameters"]

        for tool_list in (primary, secondary):
            if not tool_list:
                continue
            for tool in tool_list:
                _ingest(tool)

        return list(merged.values()) if merged else None

    def to_reflective_record(self) -> dict[str, Any]:
        user_msg = self._extract_user_message()
        assistant_msg = self._extract_assistant_message()

        if assistant_msg:
            response = assistant_msg
        elif self.final_output is not None:
            if isinstance(self.final_output, BaseModel):
                response = self.final_output.model_dump_json()
            else:
                response = str(self.final_output)
        else:
            response = "No output"

        record = {
            "user_prompt": user_msg or "No user message",
            "assistant_response": response,
            "error": self.error,
            "messages": self._serialize_messages_with_instructions(),
            "run_usage": self.usage or None,
        }

        # Add tools if available (function + output combined)
        tools = self._merge_serialized_tool_lists(
            self._serialize_tool_defs(self.function_tools),
            self._serialize_tool_defs(self.output_tools),
        )
        if tools:
            record["tools"] = tools

        return record


class _BaseAgentAdapter(
    Adapter[InputT, OutputT, MetadataT],
    Generic[InputT, OutputT, MetadataT],
    ABC,
):
    """Shared functionality for prompt and signature adapters."""

    def __init__(
        self,
        *,
        agent: "AbstractAgent[Any, Any]",
        metric: Callable[
            [Case[InputT, OutputT, MetadataT], RolloutOutput[OutputT]],
            MetricResult | Awaitable[MetricResult],
        ],
        input_spec: BoundInputSpec[BaseModel] | None = None,
        cache_manager: CacheManager | None = None,
        optimize_tools: bool | set[str] = False,
        optimize_output_type: bool = False,
        skills_fs: SkillsFS | None = None,
        skills_search_backend: SkillsSearchProvider | None = None,
        skills_capabilities: set[SkillCapability] | None = None,
        agent_usage_limits: _usage.UsageLimits | None = None,
        gepa_usage_limits: _usage.UsageLimits | None = None,
    ) -> None:
        self.agent = agent
        self.metric = metric
        self.input_spec = input_spec
        self.cache_manager = cache_manager
        self.skills_fs = skills_fs
        self.skills_search_backend = skills_search_backend
        self.skills_capabilities = skills_capabilities
        self._model_identifier = _derive_model_identifier(agent)
        if (
            self.cache_manager
            and self._model_identifier
            and not self.cache_manager.model_identifier
        ):
            self.cache_manager.set_model_identifier(self._model_identifier)
        existing_tool_optimizer = get_tool_optimizer(agent)
        existing_output_tool_optimizer = get_output_tool_optimizer(agent)

        # Store the original config for selective tool optimization
        self._optimize_tools_config: bool | set[str] = optimize_tools
        # For backwards compatibility, optimize_tools is True if any tools should be optimized
        self.optimize_tools = (
            bool(optimize_tools) or existing_tool_optimizer is not None
        )
        self.optimize_output_type = (
            optimize_output_type or existing_output_tool_optimizer is not None
        )
        self.agent_usage_limits = agent_usage_limits
        self._gepa_usage_limits = gepa_usage_limits
        self._gepa_usage = _usage.RunUsage()
        self._gepa_usage_lock = asyncio.Lock()
        if self.optimize_tools:
            self._configure_tool_optimizer()
        if self.optimize_output_type:
            self._configure_output_tool_optimizer()

    def _configure_tool_optimizer(self) -> None:
        """Install tool optimization support for plain agents when requested."""
        try:
            # If optimize_tools is a set, only allow those specific tools
            allowed_tools: set[str] | None = None
            if isinstance(self._optimize_tools_config, set):
                allowed_tools = self._optimize_tools_config
            get_or_create_tool_optimizer(self.agent, allowed_tools=allowed_tools)
        except Exception:
            pass  # Tool optimization not available for this agent type

    def _configure_output_tool_optimizer(self) -> None:
        """Install output tool optimization via prepare_output_tools when requested."""
        try:
            get_or_create_output_tool_optimizer(self.agent)
        except Exception:
            logfire.debug(
                "Output tool optimization not available for agent",
                agent_name=getattr(self.agent, "name", self.agent.__class__.__name__),
                exc_info=True,
            )

    def _usage_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for usage-limited agent invocations."""
        if self.agent_usage_limits is None:
            return {}
        return {"usage_limits": self.agent_usage_limits}

    async def _record_gepa_usage(self, run_usage: _usage.RunUsage | None) -> None:
        """Accumulate usage for the overall GEPA run and enforce limits."""
        if self._gepa_usage_limits is None or run_usage is None:
            return
        async with self._gepa_usage_lock:
            self._gepa_usage.incr(run_usage)
            self._check_gepa_usage_limits()

    def _check_gepa_usage_limits(self) -> None:
        """Raise if the aggregated usage exceeds configured limits."""
        limits = self._gepa_usage_limits
        if limits is None:
            return
        usage = self._gepa_usage

        if limits.request_limit is not None and usage.requests > limits.request_limit:
            raise UsageBudgetExceeded(
                f"Request limit exceeded: {usage.requests} > {limits.request_limit}"
            )

        if (
            limits.input_tokens_limit is not None
            and usage.input_tokens > limits.input_tokens_limit
        ):
            raise UsageBudgetExceeded(
                f"Input token limit exceeded: {usage.input_tokens} > {limits.input_tokens_limit}"
            )

        if (
            limits.output_tokens_limit is not None
            and usage.output_tokens > limits.output_tokens_limit
        ):
            raise UsageBudgetExceeded(
                f"Output token limit exceeded: {usage.output_tokens} > {limits.output_tokens_limit}"
            )

        if (
            limits.total_tokens_limit is not None
            and usage.total_tokens > limits.total_tokens_limit
        ):
            raise UsageBudgetExceeded(
                f"Total token limit exceeded: {usage.total_tokens} > {limits.total_tokens_limit}"
            )

    async def evaluate(
        self,
        batch: Sequence[Case[InputT, OutputT, MetadataT]],
        candidate: CandidateMap,
        capture_traces: bool,
        example_bank: "InMemoryExampleBank | None" = None,
    ) -> EvaluationBatch:
        """Evaluate a batch of cases asynchronously."""
        outputs: list[RolloutOutput[Any]] = []
        scores: list[float] = []
        trajectories: list[AgentAdapterTrajectory] | None = (
            [] if capture_traces else None
        )

        with self.apply_candidate(candidate):
            results = await asyncio.gather(
                *(
                    self.process_case(
                        case,
                        index,
                        capture_traces,
                        candidate,
                        example_bank=example_bank,
                    )
                    for index, case in enumerate(batch)
                )
            )

        for result in results:
            outputs.append(result["output"])
            scores.append(result["score"])
            if trajectories is not None and "trajectory" in result:
                trajectories.append(result["trajectory"])

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if trajectories else None,
        )

    def apply_candidate(self, candidate: CandidateMap | None):
        """Context manager to apply candidate to both agent and signature."""
        stack = ExitStack()
        candidate_map = candidate if candidate is not None else {}
        stack.enter_context(apply_candidate_to_agent(self.agent, candidate_map))
        if self.input_spec:
            stack.enter_context(self.input_spec.apply_candidate(candidate))
        return stack

    async def process_case(
        self,
        case: Case[InputT, OutputT, MetadataT],
        case_index: int,
        capture_traces: bool = False,
        candidate: CandidateMap | None = None,
        example_bank: "InMemoryExampleBank | None" = None,
    ) -> dict[str, Any]:
        """Process a single Case and return the metric evaluation."""
        metric_result: MetricResult | None = None
        case_name = self._case_identifier(case, case_index)
        try:
            if self.cache_manager and candidate:
                cached_agent_result = self.cache_manager.get_cached_agent_run(
                    case,
                    case_index,
                    candidate,
                    capture_traces,
                    model_identifier=self._model_identifier,
                )

                if cached_agent_result is not None:
                    trajectory, output = cached_agent_result
                else:
                    if capture_traces:
                        trajectory, output = await self._run_with_trace(
                            case, case_index, candidate, example_bank=example_bank
                        )
                    else:
                        output = await self._run_simple(
                            case, case_index, candidate, example_bank=example_bank
                        )
                        trajectory = None

                    self.cache_manager.cache_agent_run(
                        case,
                        case_index,
                        candidate,
                        trajectory,
                        output,
                        capture_traces,
                        model_identifier=self._model_identifier,
                    )
            else:
                if capture_traces:
                    trajectory, output = await self._run_with_trace(
                        case, case_index, candidate, example_bank=example_bank
                    )
                else:
                    output = await self._run_simple(
                        case, case_index, candidate, example_bank=example_bank
                    )
                    trajectory = None

            if self.cache_manager and candidate:
                cached_metric = self.cache_manager.get_cached_metric_result(
                    case,
                    case_index,
                    output,
                    candidate,
                    model_identifier=self._model_identifier,
                )

                if cached_metric is not None:
                    metric_result = cached_metric
                else:
                    maybe_metric_result = self.metric(case, output)
                    if inspect.isawaitable(maybe_metric_result):
                        metric_result = await maybe_metric_result
                    else:
                        metric_result = maybe_metric_result
                    self.cache_manager.cache_metric_result(
                        case,
                        case_index,
                        output,
                        candidate,
                        metric_result,
                        model_identifier=self._model_identifier,
                    )
            else:
                maybe_metric_result = self.metric(case, output)
                if inspect.isawaitable(maybe_metric_result):
                    metric_result = await maybe_metric_result
                else:
                    metric_result = maybe_metric_result

            if trajectory is not None:
                trajectory.metric_feedback = metric_result.feedback
                trajectory.metric_side_info = metric_result.side_info

            result: dict[str, Any] = {
                "output": output,
                "score": metric_result.score,
                "feedback": metric_result.feedback,
            }
            if trajectory is not None:
                result["trajectory"] = trajectory
            return result

        except InspectionAborted:
            raise
        except UsageBudgetExceeded:
            raise
        except Exception as exc:
            error_kind = _classify_exception(exc)
            logfire.error(
                "AgentAdapter failed to process case",
                case_id=case_name,
                error_kind=error_kind,
                capture_traces=capture_traces,
                candidate_keys=sorted(candidate.keys()) if candidate else None,
                exc_info=True,
            )
            output = RolloutOutput.from_error(exc, kind=error_kind)
            trajectory = (
                AgentAdapterTrajectory(
                    messages=[],
                    instructions=None,
                    function_tools=None,
                    output_tools=None,
                    final_output=None,
                    error=str(exc),
                    usage={},
                    case=case,
                )
                if capture_traces and error_kind == "tool"
                else None
            )
            error_result: dict[str, Any] = {
                "output": output,
                "score": 0.0,
            }
            if trajectory is not None:
                error_result["trajectory"] = trajectory
            return error_result

    def _extract_instructions_and_tools(
        self, messages: Sequence[ModelMessage]
    ) -> tuple[str | None, list[ToolDefinition] | None, list[ToolDefinition] | None]:
        """Return instruction text plus recorded tool definitions, if any."""
        instructions_text: str | None = None
        function_tools: list[ToolDefinition] | None = None
        output_tools: list[ToolDefinition] | None = None
        for message in messages:
            if not isinstance(message, ModelRequest):
                continue

            if instructions_text is None and isinstance(message.instructions, str):
                instructions_text = message.instructions

            params = message.model_request_parameters
            if params is None:
                continue

            if function_tools is None and params.function_tools:
                function_tools = list(params.function_tools)

            if output_tools is None and params.output_tools:
                output_tools = list(params.output_tools)

            if (
                instructions_text is not None
                and function_tools is not None
                and output_tools is not None
            ):
                break
        return instructions_text, function_tools, output_tools

    def _build_synthetic_request(
        self,
        case: Case[InputT, OutputT, MetadataT],
        candidate: CandidateMap | None,
    ) -> ModelRequest | None:
        return None

    def _message_history_for_case(
        self, case: Case[InputT, OutputT, MetadataT]
    ) -> list[ModelMessage] | None:
        metadata = case.metadata
        if isinstance(metadata, MetadataWithMessageHistory):
            return metadata.message_history
        return None

    @staticmethod
    def _case_identifier(
        case: Case[Any, Any, Any],
        case_index: int,
    ) -> str:
        return case.name or f"case-{case_index}"

    def _gather_messages(
        self,
        *,
        messages: Sequence[ModelMessage],
        captured_messages: Sequence[ModelMessage],
        case: Case[InputT, OutputT, MetadataT],
        case_index: int,
        candidate: CandidateMap | None,
    ) -> list[ModelMessage]:
        if messages:
            return list(messages)
        if captured_messages:
            return list(captured_messages)
        history = list(self._message_history_for_case(case) or [])
        synthetic = self._build_synthetic_request(case, candidate)
        if synthetic is not None:
            history.append(synthetic)
        return history

    async def _run_with_trace(
        self,
        case: Case[InputT, OutputT, MetadataT],
        case_index: int,
        candidate: CandidateMap | None,
        example_bank: "InMemoryExampleBank | None" = None,
    ) -> tuple[AgentAdapterTrajectory | None, RolloutOutput[Any]]:
        messages: list[ModelMessage] = []
        captured_messages: list[ModelMessage] = []
        run_result: AgentRunResult[Any] | None = None
        run_usage: _usage.RunUsage | None = None
        usage_kwargs = self._usage_kwargs()
        message_history = self._message_history_for_case(case)
        case_name = self._case_identifier(case, case_index)
        try:
            with capture_run_messages() as run_messages:
                captured_messages = run_messages
                run_result = await self._invoke_agent(
                    case,
                    candidate=candidate,
                    message_history=message_history,
                    usage_kwargs=usage_kwargs,
                    example_bank=example_bank,
                )
                messages = run_result.new_messages()
            run_usage = run_result.usage()
            await self._record_gepa_usage(run_usage)
        except InspectionAborted:
            raise
        except UsageBudgetExceeded:
            raise
        except _UsageLimitExceeded as exc:
            logfire.warn("Agent run usage limit reached", case_id=case_name)
            all_messages = self._gather_messages(
                messages=messages,
                captured_messages=captured_messages,
                case=case,
                case_index=case_index,
                candidate=candidate,
            )
            (
                instructions_text,
                function_tools,
                output_tools,
            ) = self._extract_instructions_and_tools(all_messages)
            trajectory = AgentAdapterTrajectory(
                messages=all_messages,
                instructions=instructions_text,
                function_tools=function_tools,
                output_tools=output_tools,
                final_output=None,
                error=str(exc),
                usage={},
                case=case,
            )
            output = RolloutOutput.from_error(exc, kind="system")
            return trajectory, output
        except Exception as exc:
            error_kind = _classify_exception(exc)
            logfire.error(
                "AgentAdapter run_with_trace failed",
                case_id=case_name,
                error_kind=error_kind,
                candidate_keys=sorted(candidate.keys()) if candidate else None,
            )
            all_messages = self._gather_messages(
                messages=messages,
                captured_messages=captured_messages,
                case=case,
                case_index=case_index,
                candidate=candidate,
            )
            trajectory = None
            if error_kind == "tool":
                (
                    instructions_text,
                    function_tools,
                    output_tools,
                ) = self._extract_instructions_and_tools(all_messages)
                trajectory = AgentAdapterTrajectory(
                    messages=all_messages,
                    instructions=instructions_text,
                    function_tools=function_tools,
                    output_tools=output_tools,
                    final_output=None,
                    error=str(exc),
                    usage={},
                    case=case,
                )
            output = RolloutOutput.from_error(exc, kind=error_kind)
            return trajectory, output

        assert run_result is not None
        final_messages = self._gather_messages(
            messages=messages,
            captured_messages=captured_messages,
            case=case,
            case_index=case_index,
            candidate=candidate,
        )
        final_output = run_result.output
        (
            instructions_text,
            function_tools,
            output_tools,
        ) = self._extract_instructions_and_tools(final_messages)
        trajectory = AgentAdapterTrajectory(
            messages=final_messages,
            instructions=instructions_text,
            function_tools=function_tools,
            output_tools=output_tools,
            final_output=final_output,
            error=None,
            usage=asdict(run_usage) if run_usage else {},
            case=case,
        )
        output = RolloutOutput.from_success(final_output, usage=run_usage)
        return trajectory, output

    async def _run_simple(
        self,
        case: Case[InputT, OutputT, MetadataT],
        case_index: int,
        candidate: CandidateMap | None,
        example_bank: "InMemoryExampleBank | None" = None,
    ) -> RolloutOutput[Any]:
        usage_kwargs = self._usage_kwargs()
        case_name = self._case_identifier(case, case_index)
        try:
            result = await self._invoke_agent(
                case,
                candidate=candidate,
                message_history=self._message_history_for_case(case),
                usage_kwargs=usage_kwargs,
                example_bank=example_bank,
            )
            run_usage = result.usage()
            await self._record_gepa_usage(run_usage)
            return RolloutOutput.from_success(result.output, usage=run_usage)
        except InspectionAborted:
            raise
        except UsageBudgetExceeded:
            raise
        except _UsageLimitExceeded as exc:
            logfire.warn("Agent run usage limit reached", case_id=case_name)
            return RolloutOutput.from_error(exc, kind="system")
        except Exception as exc:
            error_kind = _classify_exception(exc)
            logfire.error(
                "AgentAdapter run_simple failed",
                case_id=case_name,
                error_kind=error_kind,
                candidate_keys=sorted(candidate.keys()) if candidate else None,
            )
            return RolloutOutput.from_error(exc, kind=error_kind)

    def _build_toolsets(
        self,
        candidate: CandidateMap | None,
        example_bank: "InMemoryExampleBank | None",
    ) -> list[Any] | None:
        """Build toolsets for the agent run."""
        toolsets_list = []
        if example_bank is not None:
            toolsets_list.append(
                create_example_search_tool(
                    bank=example_bank,
                    instruction=example_bank.search_tool_instruction,
                    k=example_bank.retrieval_k,
                )
            )
        if self.skills_fs is not None:
            with apply_candidate_to_skills(self.skills_fs, candidate) as skills_view:
                toolsets_list.append(
                    create_skills_toolset(
                        skills_view,
                        search_backend=self.skills_search_backend,
                        candidate=candidate,
                        capabilities=self.skills_capabilities,
                    )
                )
        return toolsets_list if toolsets_list else None

    @abstractmethod
    async def _invoke_agent(
        self,
        case: Case[InputT, OutputT, MetadataT],
        *,
        candidate: CandidateMap | None,
        message_history: list[ModelMessage] | None,
        usage_kwargs: Mapping[str, Any],
        example_bank: "InMemoryExampleBank | None" = None,
    ) -> AgentRunResult[Any]:
        """Run the underlying agent for the provided case."""

    def make_reflective_dataset(
        self,
        *,
        candidate: CandidateMap,
        eval_batch: EvaluationBatch,
        components_to_update: Sequence[str],
        include_case_metadata: bool = False,
        include_expected_output: bool = False,
    ) -> ReflectiveDataset:
        trajectories = eval_batch.trajectories
        if not trajectories:
            return SharedReflectiveDataset(records=[])

        reflection_records: list[dict[str, Any]] = []
        for trajectory, output, score in zip(
            trajectories,
            eval_batch.outputs,
            eval_batch.scores,
        ):
            if trajectory is None:
                continue
            record: dict[str, Any] = trajectory.to_reflective_record()
            record["score"] = score
            record["success"] = output.success
            if output.error_message:
                record["error_message"] = output.error_message

            if trajectory.instructions:
                record["instructions"] = trajectory.instructions

            feedback_text = trajectory.metric_feedback
            if not feedback_text:
                if score >= 0.8:
                    feedback_text = "Good response"
                elif score >= 0.5:
                    feedback_text = "Adequate response, could be improved"
                else:
                    feedback_text = f"Poor response (score: {score:.2f})"
                    if output.error_message:
                        feedback_text += f" - Error: {output.error_message}"

            record["feedback"] = feedback_text
            if trajectory.metric_side_info:
                record["side_info"] = _serialize_for_reflection(
                    trajectory.metric_side_info
                )

            # Include case metadata and expected output based on config
            case = getattr(trajectory, "case", None)
            if case is not None:
                if include_case_metadata and case.metadata:
                    record["case_metadata"] = _serialize_for_reflection(case.metadata)
                if include_expected_output and case.expected_output:
                    record["expected_output"] = _serialize_for_reflection(
                        case.expected_output
                    )

            reflection_records.append(record)

        if not reflection_records:
            return SharedReflectiveDataset(records=[])
        return SharedReflectiveDataset(records=reflection_records)

    def get_components(self) -> CandidateMap:
        """Return the current components extracted from the agent and signature."""
        components = extract_seed_candidate_with_input_type(
            agent=self.agent,
            input_type=self.input_spec,
            optimize_output_type=self.optimize_output_type,
        )
        return components


class AgentAdapter(
    _BaseAgentAdapter[str, OutputT, MetadataT],
    Generic[OutputT, MetadataT],
):
    """Adapter for agents that accept string prompts."""

    def __init__(
        self,
        *,
        agent: "AbstractAgent[Any, Any]",
        metric: Callable[
            [Case[str, OutputT, MetadataT], RolloutOutput[OutputT]],
            MetricResult | Awaitable[MetricResult],
        ],
        cache_manager: CacheManager | None = None,
        optimize_tools: bool = False,
        optimize_output_type: bool = False,
        skills_fs: SkillsFS | None = None,
        skills_search_backend: SkillsSearchProvider | None = None,
        skills_capabilities: set[SkillCapability] | None = None,
        agent_usage_limits: _usage.UsageLimits | None = None,
        gepa_usage_limits: _usage.UsageLimits | None = None,
    ) -> None:
        super().__init__(
            agent=agent,
            metric=metric,
            input_spec=None,
            cache_manager=cache_manager,
            optimize_tools=optimize_tools,
            optimize_output_type=optimize_output_type,
            skills_fs=skills_fs,
            skills_search_backend=skills_search_backend,
            skills_capabilities=skills_capabilities,
            agent_usage_limits=agent_usage_limits,
            gepa_usage_limits=gepa_usage_limits,
        )

    async def _invoke_agent(
        self,
        case: Case[str, OutputT, MetadataT],
        *,
        candidate: CandidateMap | None,
        message_history: list[ModelMessage] | None,
        usage_kwargs: Mapping[str, Any],
        example_bank: "InMemoryExampleBank | None" = None,
    ) -> AgentRunResult[Any]:
        prompt = case.inputs
        if not isinstance(prompt, str):
            raise TypeError(
                "AgentAdapter expects Case.inputs to be a string prompt for prompt-based agents"
            )
        toolsets = self._build_toolsets(candidate, example_bank)
        return await self.agent.run(
            prompt,
            message_history=message_history,
            toolsets=toolsets,
            **usage_kwargs,
        )

    def _build_synthetic_request(
        self,
        case: Case[str, OutputT, MetadataT],
        candidate: CandidateMap | None,
    ) -> ModelRequest | None:
        prompt = case.inputs
        if not isinstance(prompt, str):
            return None
        instructions_text = None
        if candidate is not None:
            candidate_text = candidate_texts(candidate)
            instructions_text = candidate_text.get("instructions")
        if instructions_text is None:
            maybe_instructions = _safe_getattr(self.agent, "_instructions")
            if isinstance(maybe_instructions, str):
                instructions_text = maybe_instructions
        return ModelRequest(
            parts=[UserPromptPart(content=prompt)],
            instructions=instructions_text,
        )


class SignatureAgentAdapter(
    _BaseAgentAdapter[InputT, OutputT, MetadataT],
    Generic[InputT, OutputT, MetadataT],
):
    """Adapter for agents that accept structured inputs."""

    def __init__(
        self,
        *,
        agent: SignatureAgent[Any, OutputT],
        metric: Callable[
            [Case[InputT, OutputT, MetadataT], RolloutOutput[OutputT]],
            MetricResult | Awaitable[MetricResult],
        ],
        input_type: InputSpec[BaseModel] | None = None,
        cache_manager: CacheManager | None = None,
        optimize_tools: bool = False,
        optimize_output_type: bool = False,
        skills_fs: SkillsFS | None = None,
        skills_search_backend: SkillsSearchProvider | None = None,
        skills_capabilities: set[SkillCapability] | None = None,
        agent_usage_limits: _usage.UsageLimits | None = None,
        gepa_usage_limits: _usage.UsageLimits | None = None,
    ) -> None:
        bound_spec = (
            build_input_spec(input_type) if input_type is not None else agent.input_spec
        )
        self._input_model_cls: type[BaseModel] | None = (
            bound_spec.model_cls if bound_spec else None
        )
        self._signature_agent: SignatureAgent[Any, OutputT] = agent
        super().__init__(
            agent=agent,
            metric=metric,
            input_spec=bound_spec,
            cache_manager=cache_manager,
            optimize_tools=optimize_tools,
            optimize_output_type=optimize_output_type,
            skills_fs=skills_fs,
            skills_search_backend=skills_search_backend,
            skills_capabilities=skills_capabilities,
            agent_usage_limits=agent_usage_limits,
            gepa_usage_limits=gepa_usage_limits,
        )

    async def _invoke_agent(
        self,
        case: Case[InputT, OutputT, MetadataT],
        *,
        candidate: CandidateMap | None,
        message_history: list[ModelMessage] | None,
        usage_kwargs: Mapping[str, Any],
        example_bank: "InMemoryExampleBank | None" = None,
    ) -> AgentRunResult[Any]:
        inputs = self._validate_inputs(case.inputs)
        candidate_text = candidate_texts(candidate)
        toolsets = self._build_toolsets(candidate, example_bank)
        return await self._signature_agent.run_signature(
            inputs,
            message_history=message_history,
            candidate=candidate_text,
            toolsets=toolsets,
            deps=inputs,
            **usage_kwargs,
        )

    def _validate_inputs(self, inputs: InputT) -> BaseModel:
        if isinstance(inputs, BaseModel):
            return inputs
        if isinstance(inputs, Mapping):
            if self._input_model_cls is None:
                raise ValueError(
                    "SignatureAgentAdapter requires an input_type when cases provide dict inputs"
                )
            return self._input_model_cls.model_validate(inputs)
        raise ValueError("SignatureAgentAdapter requires BaseModel or dict inputs")


def create_adapter(
    *,
    agent: "AbstractAgent[Any, Any]",
    metric: Callable[
        [Case[Any, Any, MetadataT], RolloutOutput[Any]],
        MetricResult | Awaitable[MetricResult],
    ],
    input_type: InputSpec[BaseModel] | None = None,
    optimize_output_type: bool = False,
    **kwargs: Any,
) -> AgentAdapter[Any, MetadataT] | SignatureAgentAdapter[Any, Any, MetadataT]:
    """Create an adapter suited for the provided agent."""
    if isinstance(agent, SignatureAgent):
        return SignatureAgentAdapter(
            agent=agent,
            metric=metric,
            input_type=input_type,
            optimize_output_type=optimize_output_type,
            **kwargs,
        )
    if input_type is not None:
        raise TypeError(
            "input_type can only be provided when agent is a SignatureAgent"
        )
    return AgentAdapter(
        agent=agent,
        metric=metric,
        optimize_output_type=optimize_output_type,
        **kwargs,
    )

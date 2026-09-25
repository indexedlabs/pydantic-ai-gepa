"""Internal caching system for GEPA optimization to support resumable runs."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
from dataclasses import is_dataclass
from pathlib import Path
from collections.abc import Awaitable
from typing import Any, Callable, TypeVar

import cloudpickle
import logfire

from pydantic_ai.models import Model

from pydantic_evals import Case

from .gepa_graph.models import CandidateMap, candidate_texts
from .types import (
    MetadataWithMessageHistory,
    MetricResult,
    RolloutOutput,
    Trajectory,
)

CaseInputT = TypeVar("CaseInputT")
CaseOutputT = TypeVar("CaseOutputT")
CaseMetadataT = TypeVar("CaseMetadataT")

# Bump when the cache key shape changes so pre-existing entries miss instead
# of being silently reused.
_CACHE_KEY_SCHEMA_VERSION = 2


def metric_code_identity(metric: Callable[..., Any]) -> str:
    """Compute a code-identity hash for a metric callable.

    Returns a sha256 hex digest of the metric's qualified name plus its source
    code. ``functools.partial`` layers, bound methods and ``__wrapped__``
    decorators are unwrapped; each partial layer's ``args``/``keywords``, a
    bound method's ``__self__`` state, and a callable object's instance state
    are hashed as well, and for a callable object the source of its class is
    used. Pass the result as ``CacheManager(metric_identity=...)`` so cached
    scores are invalidated whenever the metric's code changes.

    This covers only the function's own source. It does **not** cover helpers
    the metric calls, judge prompts, or prompt files it reads. If your grader
    lives elsewhere, declare a version string or a freeze-manifest hash as the
    metric identity instead.

    Raises:
        ValueError: If the metric's source is unavailable (e.g. a lambda defined
            in a REPL or a compiled callable). Declare a version string as the
            metric identity instead.
    """
    target: Any = metric
    state_parts: list[str] = []
    while isinstance(target, functools.partial):
        # Partial arguments configure the grader; two partials of the same
        # function with different arguments must not share an identity.
        state_parts.append(
            f"partial-args:{CacheManager._serialize_for_key(target.args)}"
        )
        state_parts.append(
            f"partial-kwargs:{CacheManager._serialize_for_key(target.keywords)}"
        )
        target = target.func
    target = inspect.unwrap(target)
    if inspect.ismethod(target):
        # A bound method's behavior depends on the instance it is bound to.
        state_parts.append(
            f"bound-self:{CacheManager._serialize_for_key(target.__self__)}"
        )
        target = target.__func__

    if inspect.isfunction(target) or inspect.isclass(target):
        source_target = target
        module = getattr(target, "__module__", None) or ""
        qualname = getattr(target, "__qualname__", None) or repr(target)
    elif callable(target):
        # Callable object: hash the source of its class (covers __call__) plus
        # the instance state (same as a bound method's __self__), identified
        # by the class's module and qualname so the identity is stable across
        # processes (an instance repr would embed a memory address).
        source_target = type(target)
        module = getattr(source_target, "__module__", None) or ""
        qualname = getattr(source_target, "__qualname__", None) or "<unknown>"
        state_parts.append(f"instance-state:{CacheManager._serialize_for_key(target)}")
    else:
        raise ValueError(
            "metric_code_identity expected a callable; declare a version "
            "string as the metric identity instead."
        )

    try:
        source = inspect.getsource(source_target)
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"metric_code_identity could not read the source of "
            f"{module}:{qualname}; declare a version string or freeze-manifest "
            "hash as the metric identity instead."
        ) from exc

    payload = "\n".join([f"{module}:{qualname}", *state_parts, source])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CacheManager:
    """Manages caching of metric evaluation results for GEPA optimization.

    This cache allows optimization runs to be resumed without re-running
    expensive LLM calls that have already been completed.

    The cache fails closed: nothing is stored unless the caller explicitly opts
    in via ``cache_metric_results`` and/or ``cache_rollouts``. A cached score
    must never outlive a gold or grader change, so metric-result keys include
    the caller-declared ``metric_identity``, the case's ``expected_output``
    (gold) and its ``evaluators``.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        enabled: bool = True,
        verbose: bool = False,
        model_identifier: str | None = None,
        metric_identity: str | None = None,
        cache_metric_results: bool = False,
        cache_rollouts: bool = False,
    ):
        """Initialize the cache manager.

        Args:
            cache_dir: Directory to store cache files. If None, uses '.gepa_cache'
                      in the current working directory.
            enabled: Whether caching is enabled.
            verbose: Whether to log cache hits and misses.
            model_identifier: Optional string that scopes cache entries to a specific
                model (e.g., ``openai:gpt-4o``). When provided, cache keys include
                this identifier so different models never share cached artifacts.
            metric_identity: Caller-declared string that must change whenever the
                metric, its grader, judge prompts, or any code they call changes
                (a version string, a freeze-manifest hash, or
                ``metric_code_identity(metric)``). Included in every
                metric-result cache key.
            cache_metric_results: Explicit opt-in to cache metric results. Set this
                for a deterministic metric, or for a judge-model metric when you
                accept freezing the judge's first sample. Requires
                ``metric_identity``.
            cache_rollouts: Explicit opt-in to cache agent runs. A rollout that
                calls a model is sampled; caching freezes its first sample.
        """
        self.enabled = enabled
        self.verbose = verbose
        self.model_identifier = model_identifier
        self.metric_identity = metric_identity
        self.cache_metric_results = cache_metric_results
        self.cache_rollouts = cache_rollouts

        if self.enabled:
            if cache_metric_results and not (
                metric_identity and metric_identity.strip()
            ):
                raise ValueError(
                    "cache_metric_results=True requires a non-empty metric_identity "
                    "that changes whenever the metric, grader, or judge prompts "
                    "change (a version string, a freeze-manifest hash, or "
                    "metric_code_identity(metric))."
                )
            if not cache_metric_results and not cache_rollouts:
                raise ValueError(
                    "CacheManager is enabled but neither cache_metric_results nor "
                    "cache_rollouts is set, so the cache would store nothing. "
                    "Set cache_metric_results=True (requires metric_identity) "
                    "and/or cache_rollouts=True, or pass enabled=False."
                )

        if cache_dir is None:
            cache_dir = Path.cwd() / ".gepa_cache"

        self.cache_dir = Path(cache_dir)

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                logfire.info(
                    "Cache enabled",
                    cache_dir=str(self.cache_dir),
                )

    @staticmethod
    def _serialize_for_key(obj: Any) -> str:
        """Convert an object to a stable string representation for cache key generation.

        This handles various types including dataclasses, dicts, lists, and primitives.
        """
        if obj is None:
            return "None"
        elif isinstance(obj, (str, int, float, bool)):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return (
                f"[{','.join(CacheManager._serialize_for_key(item) for item in obj)}]"
            )
        elif isinstance(obj, dict):
            # Sort dict keys for stable serialization
            sorted_items = sorted(obj.items())
            return f"{{{','.join(f'{CacheManager._serialize_for_key(k)}:{CacheManager._serialize_for_key(v)}' for k, v in sorted_items)}}}"
        # Special handling for pydantic-ai message parts to exclude timestamp
        elif type(obj).__name__ in [
            "UserPromptPart",
            "SystemPromptPart",
            "ToolResponsePart",
            "ModelRequestPart",
            "ModelResponsePart",
            "RetryPromptPart",
            "ToolReturnPart",
            "TextPart",
        ]:
            # For message parts, exclude timestamp field for stable cache keys
            obj_dict = obj.__dict__.copy() if hasattr(obj, "__dict__") else {}
            obj_dict.pop("timestamp", None)  # Remove timestamp if present
            return CacheManager._serialize_for_key(obj_dict)
        elif isinstance(obj, Model):
            # A live model object (e.g. the judge model inside an LLMJudge
            # evaluator) holds non-copyable clients; identify it by system and
            # model name instead of walking its state. This must come before
            # the dataclass branch: pydantic-ai models are dataclasses, and
            # walking their fields recurses into the httpx client.
            system = getattr(obj, "system", None)
            model_name = getattr(obj, "model_name", None)
            if system is not None and model_name is not None:
                return f"model:{system}:{model_name}"
            return f"model:{type(obj).__module__}.{type(obj).__qualname__}"
        elif is_dataclass(obj):
            # Handle dataclass instances by walking fields with getattr rather
            # than dataclasses.asdict: asdict deep-copies field values, which
            # crashes on non-copyable leaves (e.g. a live model's httpx client
            # inside an LLMJudge evaluator).
            if not isinstance(obj, type):
                from dataclasses import fields

                field_values = {
                    field.name: getattr(obj, field.name) for field in fields(obj)
                }
                return CacheManager._serialize_for_key(field_values)
            else:
                # If it's a dataclass type (not instance), use its name
                return f"DataclassType:{obj.__name__}"
        elif inspect.isroutine(obj) or isinstance(obj, type):
            # Functions, methods, builtins and classes must not serialize to
            # their default repr, which embeds a memory address. Callable
            # *objects* deliberately fall through to their __dict__ so their
            # state stays in the key.
            qualname = getattr(obj, "__qualname__", None)
            module = getattr(obj, "__module__", None)
            if qualname is not None:
                prefix = f"{module}." if module else ""
                return f"callable:{prefix}{qualname}"
            return f"callable:{type(obj).__module__}.{type(obj).__qualname__}"
        elif hasattr(obj, "__dict__"):
            # For other objects, try to use their __dict__
            return CacheManager._serialize_for_key(obj.__dict__)
        else:
            # Fallback to string representation; a default repr embeds a
            # memory address, which is unstable across processes, so use the
            # type instead.
            text = str(obj)
            if " at 0x" in text:
                return f"object:{type(obj).__module__}.{type(obj).__qualname__}"
            return text

    @staticmethod
    def _fingerprint_for_key(obj: Any) -> str:
        """Serialize a value with explicit type tags for cache key generation.

        Unlike ``_serialize_for_key``, this encoding cannot collide across
        types: ``"1"`` vs ``1``, ``["a,b"]`` vs ``["a", "b"]`` and ``None``
        vs ``"None"`` all fingerprint differently. It is deterministic across
        processes: dict keys are sorted by their fingerprint and no memory
        addresses are embedded. Used for the gold and evaluator key parts,
        where a silent collision would reuse a stale score.
        """
        if obj is None:
            return "none:"
        elif isinstance(obj, bool):
            return f"bool:{json.dumps(obj)}"
        elif isinstance(obj, int):
            return f"int:{obj}"
        elif isinstance(obj, float):
            return f"float:{obj!r}"
        elif isinstance(obj, str):
            return f"str:{json.dumps(obj)}"
        elif isinstance(obj, (list, tuple)):
            tag = "list" if isinstance(obj, list) else "tuple"
            items = ",".join(CacheManager._fingerprint_for_key(item) for item in obj)
            return f"{tag}:[{items}]"
        elif isinstance(obj, dict):
            pairs = sorted(
                (
                    CacheManager._fingerprint_for_key(key),
                    CacheManager._fingerprint_for_key(value),
                )
                for key, value in obj.items()
            )
            inner = ",".join(f"{key}={value}" for key, value in pairs)
            return f"dict:{{{inner}}}"
        else:
            type_tag = f"{type(obj).__module__}.{type(obj).__qualname__}"
            return f"{type_tag}:{CacheManager._serialize_for_key(obj)}"

    @staticmethod
    def _extract_message_history(case: Case[Any, Any, Any]) -> list[Any] | None:
        metadata = case.metadata
        if isinstance(metadata, MetadataWithMessageHistory):
            return metadata.message_history
        return None

    @staticmethod
    def _case_label(case: Case[Any, Any, Any], case_index: int | None) -> str:
        if case.name:
            return case.name
        if case_index is not None:
            return f"case-{case_index}"
        return "case-unknown"

    def _generate_cache_key(
        self,
        case: Case[Any, Any, Any],
        case_index: int | None,
        output: RolloutOutput[Any] | None,
        candidate: dict[str, str],
        key_type: str = "metric",
        model_identifier: str | None = None,
    ) -> str:
        """Generate a unique cache key.

        The key is based on:
        - The key type ("metric" or "agent_run") and a key schema version
        - For metric keys only: the declared metric identity, the case's
          expected output (gold) and its evaluators
        - The case inputs/metadata/name (prompt or structured signature)
        - The output from the agent run (if provided, for metric caching)
        - The candidate prompts being evaluated
        """
        key_parts = [f"type:{key_type}", f"schema:{_CACHE_KEY_SCHEMA_VERSION}"]

        if key_type == "metric":
            # A cached score must never outlive a gold or grader change: bind
            # metric keys to the caller-declared metric identity, the case's
            # expected output (gold) and its evaluators. Agent-run keys do not
            # depend on the metric or the gold, so they omit these parts.
            key_parts.append(f"metric:{self.metric_identity}")
            expected_fingerprint = self._fingerprint_for_key(case.expected_output)
            expected_hash = hashlib.sha256(
                expected_fingerprint.encode("utf-8")
            ).hexdigest()
            key_parts.append(f"expected_output:{expected_hash}")
            key_parts.append(f"evaluators:{self._fingerprint_for_key(case.evaluators)}")

        resolved_model_identifier = model_identifier or self.model_identifier
        if resolved_model_identifier:
            key_parts.append(
                f"model:{self._serialize_for_key(resolved_model_identifier)}"
            )

        case_name = self._case_label(case, case_index)
        key_parts.append(f"case_name:{case_name}")

        serialized_inputs = self._serialize_for_key(case.inputs)
        key_parts.append(f"inputs:{serialized_inputs}")

        serialized_metadata = self._serialize_for_key(case.metadata)
        key_parts.append(f"metadata:{serialized_metadata}")

        message_history = self._extract_message_history(case)
        if message_history:
            key_parts.append(f"history:{self._serialize_for_key(message_history)}")

        # Add output information (only for metric caching)
        if output is not None:
            key_parts.append(f"result:{self._serialize_for_key(output.result)}")
            key_parts.append(f"success:{output.success}")
            key_parts.append(f"error:{output.error_message or 'None'}")

        # Add candidate prompts (sorted for stability)
        sorted_candidate = sorted(candidate.items())
        key_parts.append(f"candidate:{self._serialize_for_key(sorted_candidate)}")

        # Create a hash of all parts
        combined = "|".join(key_parts)
        hash_obj = hashlib.sha256(combined.encode("utf-8"))
        return hash_obj.hexdigest()

    def set_model_identifier(self, model_identifier: str | None) -> None:
        """Configure the default model identifier used for cache keys."""
        self.model_identifier = model_identifier

    def _try_generate_cache_key(
        self,
        key_type: str,
        case: Case[Any, Any, Any],
        case_index: int | None,
        output: RolloutOutput[Any] | None,
        candidate_text: dict[str, str],
        model_identifier: str | None = None,
    ) -> str | None:
        """Generate a cache key, failing safe.

        A key error (e.g. an exotic value in the case that cannot be
        serialized) is logged and treated as a cache miss: it must never
        fail the case being evaluated.
        """
        try:
            return self._generate_cache_key(
                case,
                case_index,
                output,
                candidate_text,
                key_type,
                model_identifier=model_identifier,
            )
        except Exception as e:
            logfire.warn(
                "Failed to generate cache key; treating as a miss",
                key_type=key_type,
                case_label=self._case_label(case, case_index),
                exception=e,
            )
            return None

    def get_cached_metric_result(
        self,
        case: Case[Any, Any, Any],
        case_index: int | None,
        output: RolloutOutput[Any],
        candidate: CandidateMap,
        model_identifier: str | None = None,
    ) -> MetricResult | None:
        """Get cached metric result if available.

        Args:
            case: The case being evaluated.
            case_index: Optional stable index for logging/cache keys when the case has no name.
            output: The output from the agent run.
            candidate: The candidate prompts being evaluated.
            model_identifier: Optional override for the model identifier to use when
                computing the cache key. Defaults to the manager-level identifier.

        Returns:
            Cached (score, feedback) tuple if found, None otherwise.
        """
        if not (self.enabled and self.cache_metric_results):
            return None

        case_label = self._case_label(case, case_index)
        candidate_text = candidate_texts(candidate)
        cache_key = self._try_generate_cache_key(
            "metric",
            case,
            case_index,
            output,
            candidate_text,
            model_identifier=model_identifier,
        )
        if cache_key is None:
            return None
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_result: MetricResult = cloudpickle.load(f)

                if self.verbose:
                    logfire.info(
                        "Cache hit for metric",
                        case_label=case_label,
                        score=cached_result.score,
                    )

                return cached_result
            except Exception as e:
                logfire.warn(
                    "Failed to load metric cache file",
                    cache_file=str(cache_file),
                    exception=e,
                )
                return None

        if self.verbose:
            logfire.debug("Cache miss for metric", case_label=case_label)

        return None

    def cache_metric_result(
        self,
        case: Case[Any, Any, Any],
        case_index: int | None,
        output: RolloutOutput[Any],
        candidate: CandidateMap,
        metric_result: MetricResult,
        model_identifier: str | None = None,
    ) -> None:
        """Cache a metric evaluation result.

        Args:
            case: The case that was evaluated.
            case_index: Optional index associated with the case.
            output: The output from the agent run.
            candidate: The candidate prompts that were evaluated.
            metric_result: The computed metric result.
        """
        if not (self.enabled and self.cache_metric_results):
            return

        candidate_text = candidate_texts(candidate)
        cache_key = self._try_generate_cache_key(
            "metric",
            case,
            case_index,
            output,
            candidate_text,
            model_identifier=model_identifier,
        )
        if cache_key is None:
            return
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                cloudpickle.dump(metric_result, f)

            if self.verbose:
                logfire.debug(
                    "Cached metric result",
                    case_label=self._case_label(case, case_index),
                    score=metric_result.score,
                )
        except Exception as e:
            logfire.warn("Failed to cache metric result", exception=e)

    def clear_cache(self) -> None:
        """Clear all cached results."""
        if not self.enabled:
            return

        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logfire.warn(
                        "Failed to delete cache file",
                        cache_file=str(cache_file),
                        exception=e,
                    )

            if self.verbose:
                logfire.info("Cache cleared", cache_dir=str(self.cache_dir))

    def get_cached_agent_run(
        self,
        case: Case[Any, Any, Any],
        case_index: int | None,
        candidate: CandidateMap,
        capture_traces: bool,
        model_identifier: str | None = None,
    ) -> tuple[Trajectory | None, RolloutOutput[Any]] | None:
        """Get cached agent run result if available.

        Args:
            case: The case being evaluated.
            case_index: Optional index associated with the case.
            candidate: The candidate prompts being evaluated.
            capture_traces: Whether traces were captured.

        Returns:
            Cached (trajectory, output) tuple if found, None otherwise.
        """
        if not (self.enabled and self.cache_rollouts):
            return None

        candidate_text = candidate_texts(candidate)
        cache_key = self._try_generate_cache_key(
            "agent_run",
            case,
            case_index,
            None,
            candidate_text,
            model_identifier=model_identifier,
        )
        if cache_key is None:
            return None
        # Add capture_traces to the key to differentiate
        cache_key = f"{cache_key}_traces_{capture_traces}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_result = cloudpickle.load(f)

                if self.verbose:
                    logfire.info(
                        "Cache hit for agent run",
                        case_label=self._case_label(case, case_index),
                    )

                return cached_result
            except Exception as e:
                logfire.warn(
                    "Failed to load agent run cache file",
                    cache_file=str(cache_file),
                    exception=e,
                )
                return None

        if self.verbose:
            logfire.debug(
                "Cache miss for agent run",
                case_label=self._case_label(case, case_index),
            )

        return None

    def cache_agent_run(
        self,
        case: Case[Any, Any, Any],
        case_index: int | None,
        candidate: CandidateMap,
        trajectory: Trajectory | None,
        output: RolloutOutput[Any],
        capture_traces: bool,
        model_identifier: str | None = None,
    ) -> None:
        """Cache an agent run result.

        Args:
            case: The case that was evaluated.
            case_index: Optional index associated with the case.
            candidate: The candidate prompts that were evaluated.
            trajectory: The execution trajectory (if captured).
            output: The output from the agent run.
            capture_traces: Whether traces were captured.
        """
        if not (self.enabled and self.cache_rollouts):
            return

        candidate_text = candidate_texts(candidate)
        cache_key = self._try_generate_cache_key(
            "agent_run",
            case,
            case_index,
            None,
            candidate_text,
            model_identifier=model_identifier,
        )
        if cache_key is None:
            return
        # Add capture_traces to the key to differentiate
        cache_key = f"{cache_key}_traces_{capture_traces}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                cloudpickle.dump((trajectory, output), f)

            if self.verbose:
                logfire.debug(
                    "Cached agent run",
                    case_label=self._case_label(case, case_index),
                )
        except Exception as e:
            logfire.warn("Failed to cache agent run", exception=e)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the cache.

        Returns:
            Dictionary with cache statistics.
        """
        if not self.enabled:
            return {"enabled": False}

        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "num_cached_results": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }


def create_cached_metric(
    metric: Callable[
        [Case[CaseInputT, CaseOutputT, CaseMetadataT], RolloutOutput[Any]], MetricResult
    ],
    cache_manager: CacheManager,
    candidate: CandidateMap,
    *,
    model_identifier: str | None = None,
) -> Callable[
    [Case[CaseInputT, CaseOutputT, CaseMetadataT], RolloutOutput[Any]],
    MetricResult | Awaitable[MetricResult],
]:
    """Create a cached version of a metric function.

    This wrapper function checks the cache before calling the actual metric,
    and caches the result afterward.

    Args:
        metric: The original metric function that accepts a Case.
        cache_manager: The cache manager to use.
        candidate: The current candidate being evaluated.
        model_identifier: Optional override for the model identifier. When not
            provided, the cache manager's configured identifier (if any) is used.

    Returns:
        A wrapped metric function that uses caching.
    """

    def cached_metric(
        case: Case[CaseInputT, CaseOutputT, CaseMetadataT],
        output: RolloutOutput[Any],
    ) -> MetricResult | Awaitable[MetricResult]:
        # Check cache first
        cached_result = cache_manager.get_cached_metric_result(
            case,
            None,
            output,
            candidate,
            model_identifier=model_identifier,
        )

        if cached_result is not None:
            return cached_result

        # Call the actual metric
        metric_result = metric(case, output)
        if inspect.isawaitable(metric_result):

            async def cache_and_return() -> MetricResult:
                awaited = await metric_result
                cache_manager.cache_metric_result(
                    case,
                    None,
                    output,
                    candidate,
                    awaited,
                    model_identifier=model_identifier,
                )
                return awaited

            return cache_and_return()

        # Cache the result
        cache_manager.cache_metric_result(
            case,
            None,
            output,
            candidate,
            metric_result,
            model_identifier=model_identifier,
        )

        return metric_result

    return cached_metric

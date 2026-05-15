"""Tests for the ``case_factory`` hook on ``SignatureAgentAdapter``.

The case factory is the eval-time extension point for converting a raw
dataset ``Case`` into the agent's fully-materialized input model — used
when dataset rows carry deferred references (file paths, Mighty file
ids, base64 blobs) that need to be loaded into ``BinaryContent`` (or
similar) before the agent runs.

These tests pin:

- The factory replaces the default ``_validate_inputs`` path when set.
- Sync and async factories are both honored.
- The factory's product is used as both the agent input and the deps.
- Non-``BaseModel`` returns raise ``TypeError`` instead of being passed
  through silently.
- ``create_adapter`` refuses ``case_factory`` for plain ``Agent``
  rollouts (no input model to materialize).
- ``GepaConfig`` parses + validates ``case_factory`` and
  ``resolve_case_factory`` imports the referenced callable.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

asyncio_mark = pytest.mark.asyncio(loop_scope="function")

from pydantic_ai_gepa import MetricResult, SignatureAgent
from pydantic_ai_gepa.adapters.agent_adapter import (
    SignatureAgentAdapter,
    create_adapter,
)
from pydantic_ai_gepa.cli.layout import (
    GepaConfig,
    GepaConfigError,
    resolve_case_factory,
)


class _Input(BaseModel):
    """Trivial input model used by the test agents."""

    payload: str = ""


class _Output(BaseModel):
    text: str = "ok"


def _build_signature_agent() -> SignatureAgent[Any, _Output]:
    """Construct a SignatureAgent with a TestModel-backed deterministic output."""
    inner = Agent(
        model=TestModel(custom_output_args=_Output(text="ok")),
        output_type=_Output,
        name="case-factory-test",
    )
    return SignatureAgent(inner, input_type=_Input, output_type=_Output)


def _scoring_metric(case: Case[Any, Any, Any], output: Any) -> MetricResult:
    """Trivial scoring callback — every rollout scores 1.0."""
    return MetricResult(score=1.0, feedback="ok")


async def _async_factory(case: Case[Any, Any, Any]) -> _Input:
    """Async factory branch — exercises ``inspect.isawaitable``."""
    await asyncio.sleep(0)
    return _Input(payload=f"async:{case.name}")


def _sync_factory(case: Case[Any, Any, Any]) -> _Input:
    return _Input(payload=f"sync:{case.name}")


# Module-level callable for the resolve_case_factory dotted-ref test.
def _public_case_factory(case: Case[Any, Any, Any]) -> _Input:
    return _Input(payload=case.name or "")


@asyncio_mark
async def test_materialize_inputs_uses_sync_case_factory() -> None:
    """When set, the factory replaces ``_validate_inputs`` entirely.

    The case's ``inputs`` field is intentionally a dict with NO
    ``payload`` key so the assertion fails loudly if the default
    validation path is taken instead of the factory.
    """
    agent = _build_signature_agent()
    adapter = SignatureAgentAdapter[Any, _Output, dict[str, Any]](
        agent=agent,
        metric=_scoring_metric,
        case_factory=_sync_factory,
    )
    case = Case(name="c1", inputs={"deferred_ref": "ignored-by-factory"})
    result = await adapter._materialize_inputs(case)
    assert isinstance(result, _Input)
    assert result.payload == "sync:c1"


@asyncio_mark
async def test_materialize_inputs_uses_async_case_factory() -> None:
    agent = _build_signature_agent()
    adapter = SignatureAgentAdapter[Any, _Output, dict[str, Any]](
        agent=agent,
        metric=_scoring_metric,
        case_factory=_async_factory,
    )
    case = Case(name="c2", inputs={"deferred_ref": "ignored-by-factory"})
    result = await adapter._materialize_inputs(case)
    assert isinstance(result, _Input)
    assert result.payload == "async:c2"


@asyncio_mark
async def test_materialize_inputs_falls_back_to_validate_inputs() -> None:
    """No factory → existing pass-through behavior must still work."""
    agent = _build_signature_agent()
    adapter = SignatureAgentAdapter[Any, _Output, dict[str, Any]](
        agent=agent,
        metric=_scoring_metric,
        case_factory=None,
    )
    case = Case(name="c3", inputs=_Input(payload="from-pass-through"))
    result = await adapter._materialize_inputs(case)
    assert isinstance(result, _Input)
    assert result.payload == "from-pass-through"


@asyncio_mark
async def test_materialize_inputs_rejects_non_basemodel_factory_return() -> None:
    """A factory that forgets to return a BaseModel must surface loudly,
    not silently pass through a dict that would explode later in
    ``run_signature``.
    """

    def _broken_factory(case: Case[Any, Any, Any]) -> Any:
        return {"payload": "not-a-pydantic-model"}

    agent = _build_signature_agent()
    adapter = SignatureAgentAdapter[Any, _Output, dict[str, Any]](
        agent=agent,
        metric=_scoring_metric,
        case_factory=_broken_factory,
    )
    case = Case(name="broken", inputs={})
    with pytest.raises(TypeError, match="case_factory must return a pydantic BaseModel"):
        await adapter._materialize_inputs(case)


def test_create_adapter_rejects_case_factory_for_plain_agent() -> None:
    """``case_factory`` only makes sense for SignatureAgentAdapter — plain
    ``Agent`` rollouts work on prompt-string inputs that don't need
    materialization.
    """
    plain_agent = Agent(
        model=TestModel(custom_output_text="ok"),
        output_type=str,
        name="plain-agent",
    )
    with pytest.raises(TypeError, match="case_factory can only be provided"):
        create_adapter(
            agent=plain_agent,
            metric=_scoring_metric,
            case_factory=_sync_factory,
        )


def test_gepa_config_parses_case_factory() -> None:
    cfg = GepaConfig.from_dict(
        {
            "agent": "pkg.mod:agent",
            "dataset": ".gepa/dataset.jsonl",
            "case_factory": "pkg.mod:factory",
        }
    )
    assert cfg.case_factory == "pkg.mod:factory"


def test_gepa_config_rejects_invalid_case_factory_ref() -> None:
    with pytest.raises(GepaConfigError, match="case_factory"):
        GepaConfig.from_dict(
            {"agent": "pkg.mod:agent", "case_factory": "missing_colon"},
        )


def test_gepa_config_case_factory_defaults_to_none() -> None:
    cfg = GepaConfig.from_dict({"agent": "pkg.mod:agent"})
    assert cfg.case_factory is None


def test_resolve_case_factory_imports_dotted_ref() -> None:
    cfg = GepaConfig(
        agent="pkg.mod:agent",
        case_factory=f"{__name__}:_public_case_factory",
    )
    resolved = resolve_case_factory(cfg)
    assert resolved is _public_case_factory


def test_resolve_case_factory_returns_none_when_absent() -> None:
    cfg = GepaConfig(agent="pkg.mod:agent")
    assert resolve_case_factory(cfg) is None

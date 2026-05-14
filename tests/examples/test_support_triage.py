"""Deterministic smoke test for the support-triage example.

The real demo runs against `openai:gpt-4o-mini`; in CI we patch the agent's
model to a `FunctionModel` that picks a tool based on simple keyword rules,
so the example exercises the full eval path without an API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals import Case

from pydantic_ai_gepa.cli.dataset import load_dataset
from pydantic_ai_gepa.evaluation import evaluate_candidate_dataset


# Repository root (this file at tests/examples/test_support_triage.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _keyword_router(messages, info: AgentInfo) -> ModelResponse:
    """Pick a tool based on keyword presence in the latest user prompt."""
    # Find the latest user prompt text. pydantic-ai's history shape is a list
    # of ModelMessage; the user prompt usually lives in the most recent
    # ModelRequest with UserPromptPart children.
    user_text = ""
    for msg in reversed(messages):
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    user_text = part.content
                    break
            if user_text:
                break

    # On the second call (after a tool returned) just echo the tool result.
    last = messages[-1] if messages else None
    if last is not None:
        from pydantic_ai.messages import ModelRequest, ToolReturnPart

        if isinstance(last, ModelRequest):
            for part in reversed(last.parts):
                if isinstance(part, ToolReturnPart):
                    return ModelResponse(parts=[TextPart(content=str(part.content))])

    text = user_text.lower()
    if "reset" in text or "password" in text or "forgot" in text or "locked" in text:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="send_reset_link", args={"email": "user@example.com"}
                )
            ]
        )
    if "tracking" in text or "package" in text or "shipment" in text:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="check_shipment_status", args={"tracking_number": "TRK"}
                )
            ]
        )
    if (
        "escalat" in text
        or "supervisor" in text
        or "manager" in text
        or "legal" in text
    ):
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name="escalate_to_human", args={"reason": "urgent"})
            ]
        )
    if (
        "ticket" in text
        or "complaint" in text
        or "damaged" in text
        or "billing error" in text
    ):
        return ModelResponse(
            parts=[ToolCallPart(tool_name="open_ticket", args={"summary": "issue"})]
        )
    if "order" in text or "eta" in text:
        return ModelResponse(
            parts=[ToolCallPart(tool_name="lookup_order", args={"order_id": "A-1"})]
        )
    # Fallback
    return ModelResponse(
        parts=[ToolCallPart(tool_name="open_ticket", args={"summary": "fallback"})]
    )


@pytest.fixture
def support_agent(monkeypatch: pytest.MonkeyPatch):
    """Import the example agent module and swap its model for a deterministic FunctionModel."""
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    # The example agent's default model is `openai:gpt-4o-mini`, which eagerly
    # constructs an OpenAI client (needs an API key) at module import time.
    # We're swapping the model out anyway, but the import has to succeed first,
    # so set a dummy key just for the import phase.
    monkeypatch.setenv("OPENAI_API_KEY", "test-sk-not-real")
    # Force a fresh import so any prior tests pick up the FunctionModel swap.
    for name in list(sys.modules):
        if name.startswith("examples.support_triage"):
            sys.modules.pop(name, None)
    from examples.support_triage.agent import agent
    from examples.support_triage import metric as metric_module

    monkeypatch.setattr(agent, "_model", FunctionModel(_keyword_router))
    return agent, metric_module.routed_to_expected_tool


def test_dataset_loads() -> None:
    cases = load_dataset(REPO_ROOT / "examples" / "support_triage" / "dataset.jsonl")
    assert len(cases) >= 15
    assert {case.expected_output for case in cases} == {
        "open_ticket",
        "lookup_order",
        "escalate_to_human",
        "send_reset_link",
        "check_shipment_status",
    }


@pytest.mark.asyncio
async def test_baseline_eval_with_deterministic_routing(support_agent) -> None:
    agent, metric = support_agent
    cases = load_dataset(REPO_ROOT / "examples" / "support_triage" / "dataset.jsonl")

    records = await evaluate_candidate_dataset(
        agent=agent,
        metric=metric,
        dataset=cases,
        candidate=None,
        concurrency=4,
    )

    mean = sum(r.score for r in records) / len(records)
    # The deterministic router should route most cases correctly; we want
    # high accuracy here to confirm the wiring works.
    assert mean >= 0.5, (
        f"Baseline mean was {mean}; per-case scores: {[r.score for r in records]}"
    )


def test_metric_unwraps_rollout_output() -> None:
    """Sanity check: metric handles RolloutOutput wrapping."""
    import asyncio

    from pydantic_ai_gepa.types import RolloutOutput
    from examples.support_triage.metric import routed_to_expected_tool

    case = Case(name="case-1", inputs="x", expected_output="open_ticket")
    output = RolloutOutput.from_success("open_ticket: opened a ticket — foo")
    result = asyncio.run(routed_to_expected_tool(case, output))
    assert result.score == 1.0


def test_metric_handles_wrong_route() -> None:
    import asyncio

    from pydantic_ai_gepa.types import RolloutOutput
    from examples.support_triage.metric import routed_to_expected_tool

    case = Case(name="case-1", inputs="x", expected_output="open_ticket")
    output = RolloutOutput.from_success("lookup_order: ...")
    result = asyncio.run(routed_to_expected_tool(case, output))
    assert result.score == 0.0

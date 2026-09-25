"""Regression tests for provider failures that must stop evaluation."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa.adapters.agent_adapter import AgentAdapter
from pydantic_ai_gepa.evaluation import evaluate_callable_dataset
from pydantic_ai_gepa.provider_errors import (
    ProviderStopError,
    is_provider_stop_error,
    is_provider_stop_message,
)
from pydantic_ai_gepa.types import MetricResult


@pytest.mark.parametrize(
    "body",
    [
        {"type": "insufficient_quota"},
        {
            "message": "You have no credits remaining.",
            "code": "credit_balance_exhausted",
        },
        {"code": "project_spend_limit_exceeded"},
        {"code": "billing_hard_limit_reached"},
        {"code": "organization_spend_limit_exceeded"},
    ],
)
def test_billing_quota_errors_require_operator(body: object) -> None:
    error = ModelHTTPError(status_code=429, model_name="test", body=body)

    assert is_provider_stop_error(error)


def test_credentials_require_operator_but_transient_rate_limits_do_not() -> None:
    credential_error = ModelHTTPError(
        status_code=401, model_name="test", body={"code": "invalid_api_key"}
    )
    rate_limit = ModelHTTPError(
        status_code=429, model_name="test", body={"code": "rate_limit_exceeded"}
    )

    assert is_provider_stop_error(credential_error)
    assert not is_provider_stop_error(rate_limit)


@pytest.mark.asyncio
async def test_agent_evaluation_propagates_billing_quota_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AgentAdapter(
        agent=Agent(TestModel()),
        metric=lambda case, output: MetricResult(score=0.0),
    )
    error = ModelHTTPError(
        status_code=429,
        model_name="test",
        body={"code": "credit_balance_exhausted"},
    )
    monkeypatch.setattr(adapter, "_run_simple", AsyncMock(side_effect=error))

    with pytest.raises(ModelHTTPError, match="credit_balance_exhausted"):
        await adapter.process_case(Case(name="case-1", inputs="x"), 0)


@pytest.mark.asyncio
async def test_plain_callable_evaluation_propagates_billing_quota_error() -> None:
    error = ModelHTTPError(
        status_code=429,
        model_name="test",
        body={"code": "insufficient_quota"},
    )

    async def evaluate(case: object) -> object:
        raise error

    with pytest.raises(ModelHTTPError, match="insufficient_quota"):
        await evaluate_callable_dataset(
            evaluate=evaluate,
            metric=lambda case, output: 0.0,
            dataset=[Case(name="case-1", inputs="x")],
        )


def test_provider_stop_error_stops_directly_and_through_a_cause() -> None:
    stop = ProviderStopError("child process: insufficient_quota")
    try:
        raise RuntimeError("case failed") from stop
    except RuntimeError as wrapped:
        chained = wrapped

    assert is_provider_stop_error(stop)
    assert is_provider_stop_error(chained)
    assert not is_provider_stop_error(RuntimeError("case failed"))


def test_stop_failures_are_found_in_provider_error_text() -> None:
    def text(status: int, body: object) -> str:
        return str(ModelHTTPError(status_code=status, model_name="m", body=body))

    assert is_provider_stop_message(text(429, {"code": "insufficient_quota"}))
    assert is_provider_stop_message("Error code: 429 - PROJECT_SPEND_LIMIT_EXCEEDED")
    assert is_provider_stop_message(text(401, {"code": "invalid_api_key"}))
    assert is_provider_stop_message(text(403, None))
    assert not is_provider_stop_message(text(429, {"code": "rate_limit_exceeded"}))
    assert not is_provider_stop_message(text(4010, None))


@pytest.mark.asyncio
async def test_plain_callable_evaluation_propagates_provider_stop_error() -> None:
    async def evaluate(case: object) -> object:
        raise ProviderStopError("child process: project_spend_limit_exceeded")

    with pytest.raises(ProviderStopError, match="project_spend_limit_exceeded"):
        await evaluate_callable_dataset(
            evaluate=evaluate,
            metric=lambda case, output: 0.0,
            dataset=[Case(name="case-1", inputs="x")],
        )


@pytest.mark.asyncio
async def test_a_stop_cancels_cases_already_in_flight() -> None:
    import asyncio

    started = asyncio.Event()
    cancelled: list[str] = []

    async def evaluate(case: Case) -> object:
        if case.name == "stops":
            await started.wait()
            raise ProviderStopError("insufficient_quota")
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.append(case.name or "")
            raise
        return "late"

    with pytest.raises(ProviderStopError):
        await evaluate_callable_dataset(
            evaluate=evaluate,
            metric=lambda case, output: 0.0,
            dataset=[Case(name="slow", inputs="x"), Case(name="stops", inputs="y")],
            concurrency=2,
        )
    # The slow case was cancelled and drained before the stop propagated.
    assert cancelled == ["slow"]

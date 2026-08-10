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
from pydantic_ai_gepa.provider_errors import is_provider_stop_error
from pydantic_ai_gepa.types import MetricResult


@pytest.mark.parametrize(
    "body",
    [
        {"type": "insufficient_quota"},
        {
            "message": "You have no credits remaining.",
            "code": "credit_balance_exhausted",
        },
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

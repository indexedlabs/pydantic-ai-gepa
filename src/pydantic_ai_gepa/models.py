from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.models import (
    KnownModelName,
    Model,
    ModelRequestParameters,
    StreamedResponse,
)
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai._run_context import RunContext


class OptimizableModel(WrapperModel):
    """Wraps an agent's model to capture request parameters for GEPA optimization.

    This captures the final `ModelRequestParameters` and `ModelSettings` (which include
    resolved tools and runtime configurations) and dynamically attaches them to the 
    corresponding `ModelRequest` message in the history.
    """

    def __init__(self, wrapped: Model | KnownModelName) -> None:
        super().__init__(wrapped)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        prepared_settings, prepared_parameters = self.wrapped.prepare_request(
            model_settings,
            model_request_parameters,
        )

        # Attach the prepared parameters to the most recent ModelRequest
        last_request = next(
            (msg for msg in reversed(messages) if isinstance(msg, ModelRequest)), None
        )
        if last_request is not None:
            setattr(last_request, "model_request_parameters", prepared_parameters)
            setattr(last_request, "model_settings", prepared_settings)

        return await self.wrapped.request(
            messages, model_settings, model_request_parameters
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        prepared_settings, prepared_parameters = self.wrapped.prepare_request(
            model_settings,
            model_request_parameters,
        )

        # Attach the prepared parameters to the most recent ModelRequest
        last_request = next(
            (msg for msg in reversed(messages) if isinstance(msg, ModelRequest)), None
        )
        if last_request is not None:
            setattr(last_request, "model_request_parameters", prepared_parameters)
            setattr(last_request, "model_settings", prepared_settings)

        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as stream:
            yield stream

"""Wrapper model that surfaces provider payloads prior to dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence, cast

from openai import NOT_GIVEN as OPENAI_NOT_GIVEN

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

MapMessagesCallback = Callable[
    [Model, list[ModelMessage], ModelSettings | None, ModelRequestParameters],
    Awaitable[list[Any]],
]


@dataclass(slots=True)
class InspectionSnapshot:
    """Captured request data for inspection."""

    model_name: str
    provider_name: str
    original_messages: Sequence[ModelMessage]
    provider_messages: list[Any]
    model_settings: dict[str, Any]
    request_parameters: ModelRequestParameters
    extra: dict[str, Any] = field(default_factory=dict)


class InspectionAborted(RuntimeError):
    """Raised when a request is intercepted for inspection."""

    def __init__(self, snapshot: InspectionSnapshot):
        super().__init__("Model request intercepted for inspection.")
        self.snapshot = snapshot


class InspectingModel(WrapperModel):
    """Wrap a model and raise with captured payloads instead of dispatching."""

    def __init__(
        self,
        wrapped: Model,
        *,
        map_messages: MapMessagesCallback | None = None,
    ) -> None:
        super().__init__(wrapped)
        self._openai: OpenAIChatModel | OpenAIResponsesModel | None = (
            self._maybe_openai_model(self.wrapped)
        )
        if self._openai is None and map_messages is None:
            raise TypeError(
                "InspectingModel requires either an OpenAIChatModel/OpenAIResponsesModel "
                "or a custom map_messages callback."
            )
        self._map_messages_override = map_messages

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ):
        snapshot = await self._build_snapshot(
            messages, model_settings, model_request_parameters
        )
        raise InspectionAborted(snapshot)

    async def _build_snapshot(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> InspectionSnapshot:
        if self._openai is not None:
            settings, prepared_params = self._openai.prepare_request(
                model_settings,
                model_request_parameters,
            )
            provider_messages = await self._map_openai_messages(
                messages, settings, prepared_params
            )
            settings_payload = dict(settings or {})
            model_name = self._openai.model_name
            provider_name = self._openai.system
        else:
            prepared_params = model_request_parameters
            if self._map_messages_override is None:
                raise RuntimeError(
                    "map_messages callback missing for non-OpenAI model."
                )
            provider_messages = await self._map_messages_override(
                self.wrapped,
                messages,
                model_settings,
                model_request_parameters,
            )
            settings_payload = _settings_to_dict(model_settings)
            model_name = getattr(
                self.wrapped, "model_name", self.wrapped.__class__.__name__
            )
            provider_name = getattr(
                self.wrapped, "system", self.wrapped.__class__.__name__
            )

        return InspectionSnapshot(
            model_name=model_name,
            provider_name=provider_name,
            original_messages=list(messages),
            provider_messages=provider_messages,
            model_settings=settings_payload,
            request_parameters=prepared_params,
        )

    async def _map_openai_messages(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> list[Any]:
        assert self._openai is not None
        if isinstance(self._openai, OpenAIResponsesModel):
            settings = cast(OpenAIResponsesModelSettings, model_settings or {})
            mapped = await self._openai._map_messages(
                messages, settings, model_request_parameters
            )
            if isinstance(mapped, tuple):
                _model_name, response_items = mapped
                return list(response_items)
            return mapped

        return await self._openai._map_messages(messages, model_request_parameters)

    @staticmethod
    def _maybe_openai_model(
        model: Model,
    ) -> OpenAIChatModel | OpenAIResponsesModel | None:
        if isinstance(model, (OpenAIChatModel, OpenAIResponsesModel)):
            return cast(OpenAIChatModel | OpenAIResponsesModel, model)
        return None


__all__ = [
    "InspectingModel",
    "InspectionAborted",
    "InspectionSnapshot",
]


def _settings_to_dict(settings: ModelSettings | None) -> dict[str, Any]:
    if settings is None:
        return {}
    if isinstance(settings, dict):
        return dict(settings)
    return dict(settings)  # type: ignore[arg-type]


def _json_safe(value: Any) -> Any:
    if value is OPENAI_NOT_GIVEN:
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value

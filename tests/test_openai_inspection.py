from __future__ import annotations

import pytest

from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from pydantic_ai_gepa.inspection import (
    InspectionAborted,
    InspectionSnapshot,
    InspectingModel,
)


class _EchoModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.requests = 0

    @property
    def model_name(self) -> str:
        return "test-model"

    @property
    def system(self) -> str:
        return "test-provider"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ):
        self.requests += 1
        return ModelResponse(parts=[TextPart("ok")])


async def _map_messages(
    model: Model,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": part.content
            if isinstance(part.content, str)
            else str(part.content),
        }
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]


def _make_messages() -> list[ModelMessage]:
    return [
        ModelRequest(
            parts=[UserPromptPart("2+2?")],
            instructions="calc it",
        )
    ]


@pytest.mark.asyncio
async def test_inspection_exception_includes_snapshot() -> None:
    fake = _EchoModel()
    wrapper = InspectingModel(fake, map_messages=_map_messages)

    with pytest.raises(InspectionAborted) as excinfo:
        await wrapper.request(
            _make_messages(),
            {"temperature": 0.5},
            ModelRequestParameters(),
        )

    snapshot: InspectionSnapshot = excinfo.value.snapshot
    assert snapshot.provider_messages[0]["content"] == "2+2?"
    assert snapshot.model_settings["temperature"] == 0.5
    assert fake.requests == 0
